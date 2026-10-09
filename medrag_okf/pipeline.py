"""The two pipelines compared in the review:

  baseline_rag : query -> dense top-k -> LLM/extractive generate -> answer           (what a vanilla RAG does)
  MedRAG_OKF       : query -> safety -> clarify -> hybrid retrieve -> re-rank -> evidence gate
                 -> generate -> verify claims -> safe answer + sources + evidence status
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from .config import settings
from .corpus import load_knowledge
from .generation import generate, strip_citations
from .retrieval import Hit, Index
from .safety import DISCLAIMER, Triage, assess_safety, needs_clarification
from .verification import Verification, verify

INSUFFICIENT = (
    "I could not find enough reliable evidence in my medical knowledge base to answer this safely, so I won't guess. "
    "Please rephrase with more detail, or ask a doctor or pharmacist."
)


@dataclass
class Source:
    rank: int
    title: str
    section: str
    source: str
    text: str
    score: float
    dense: float
    bm25: float
    page: int | None = None
    doc_id: str = ""
    chunk_id: str = ""
    concept_type: str = ""
    status: str = ""
    verified: bool = False
    stale: bool = False
    via_graph: bool = False        # retrieved by following an OKF link from the best-matching concept


@dataclass
class Result:
    mode: str                                  # "medrag_okf" | "baseline"
    question: str
    status: str                                # answered | insufficient_evidence | clarification | emergency | answered_with_flags
    answer: str = ""
    evidence_score: float | None = None
    evidence_label: str = ""
    supported_rate: float | None = None
    claims: list = field(default_factory=list)
    removed_claims: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    clarifying_questions: list = field(default_factory=list)
    safety_level: str = "none"
    safety_category: str = ""
    safety_message: str = ""
    generator: str = ""
    verifier: str = ""
    trace: list = field(default_factory=list)  # [{stage, ms, detail}]
    provenance: dict = field(default_factory=dict)   # OKF trust signals of the concepts the answer rests on
    related: list = field(default_factory=list)      # OKF graph neighbours worth reading next
    latency_ms: float = 0.0
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def _evidence_label(score: float) -> str:
    if score >= 0.70:
        return "Strong"
    if score >= settings.min_evidence_score:
        return "Moderate"
    return "Weak"


def _sources(hits: list[Hit], bundle=None) -> list[Source]:
    out = []
    for i, h in enumerate(hits, 1):
        c = bundle.concepts.get(h.chunk.doc_id) if bundle else None
        out.append(Source(i, h.chunk.title, h.chunk.section, h.chunk.source, h.chunk.text, round(h.score, 3), round(h.dense, 3),
                          round(h.bm25, 2), h.chunk.page, h.chunk.doc_id, h.chunk.chunk_id, h.chunk.concept_type,
                          c.status if c else "", bool(c and c.is_verified()), bool(c and c.is_stale()), bool(h.ranks.get("graph") and len(h.ranks) == 1)))
    return out


def okf_provenance(bundle, hits: list[Hit]) -> tuple[dict, list]:
    """OKF trust signals (status / verified / generated / stale_after) for the concepts behind an answer,
    plus the concepts they link to (graph neighbours) that were not already used as evidence."""
    paths = list(dict.fromkeys(h.chunk.doc_id for h in hits))
    concepts = [bundle.concepts[p] for p in paths if p in bundle.concepts]
    items = [{"path": c.path, "title": c.title, "type": c.type, "status": c.status, "verified": c.is_verified(),
              "verified_by": [v.get("by") for v in c.verified], "stale": c.is_stale(), "stale_after": c.stale_after_text,
              "generated_by": c.generated_by, "source": c.source_label()} for c in concepts]
    notes = []
    unverified = [i for i in items if not i["verified"]]
    if items and unverified:
        notes.append(f"{len(unverified)} of {len(items)} source concept(s) have no human verification yet (OKF status: "
                     f"{', '.join(sorted({i['status'] for i in unverified}))}).")
    stale = [i for i in items if i["stale"]]
    if stale:
        notes.append("Past their stale_after date, so may be outdated: " + ", ".join(i["title"] for i in stale) + ".")
    related, seen = [], set(paths)
    for c in concepts[:3]:
        for l in c.links:
            t = bundle.concepts.get(l)
            if t and l not in seen:
                seen.add(l)
                related.append({"path": l, "title": t.title, "type": t.type, "via": c.title})
    prov = {"concepts": items, "verified": len(items) - len(unverified), "total": len(items), "notes": notes}
    return prov, related[:5]


class MedRAG_OKF:
    def __init__(self, index: Index | None = None):
        self.bundle, chunks = load_knowledge()
        self.index = index or Index(chunks)
        self._link_graph()

    def _link_graph(self):
        self.index.set_graph({p: self.bundle.neighbors(p) for p in self.bundle.concepts})

    def rebuild(self):
        self.bundle, chunks = load_knowledge()
        self.index = Index(chunks, embedder=self.index.embedder, reranker=self.index.reranker)
        self._link_graph()

    # ------------------------------------------------------------------ baseline
    def baseline(self, question: str, top_k: int | None = None, use_llm: bool = True) -> Result:
        k = top_k or settings.top_k
        t0 = time.perf_counter()
        trace = []
        t = time.perf_counter()
        hits = self.index.baseline_retrieve(question, k)
        trace.append({"stage": "Dense retrieval", "ms": _ms(t), "detail": f"top-{k} by embedding similarity only"})
        t = time.perf_counter()
        answer, gen = generate(question, hits, use_llm)
        trace.append({"stage": "Generation", "ms": _ms(t), "detail": gen})
        if answer.strip() == "INSUFFICIENT_EVIDENCE":
            answer = "The model could not answer from the retrieved context."
        res = Result("baseline", question, "answered", answer=answer, sources=[asdict(s) for s in _sources(hits, self.bundle)], generator=gen, trace=trace)
        res.latency_ms = _ms(t0)
        return res

    # ------------------------------------------------------------------ MedRAG_OKF
    def ask(self, question: str, top_k: int | None = None, use_llm: bool = True, skip_clarification: bool = False,
            graph_expand: bool | None = None) -> Result:
        k = top_k or settings.top_k
        graph = settings.graph_expand if graph_expand is None else graph_expand
        t0 = time.perf_counter()
        trace: list[dict] = []

        # 1. safety / triage ------------------------------------------------
        t = time.perf_counter()
        triage: Triage = assess_safety(question)
        trace.append({"stage": "Safety / triage", "ms": _ms(t), "detail": f"{triage.level}" + (f": {triage.category}" if triage.category else "")})
        if triage.level == "emergency":
            res = Result("medrag_okf", question, "emergency", answer=triage.message, safety_level="emergency",
                         safety_category=triage.category, safety_message=triage.message, trace=trace)
            res.latency_ms = _ms(t0)
            return res

        # 2. query clarification -------------------------------------------
        t = time.perf_counter()
        if not skip_clarification:
            need, qs = needs_clarification(question)
            trace.append({"stage": "Query analysis", "ms": _ms(t), "detail": "ambiguous - asking follow-up" if need else "clear enough to retrieve"})
            if need:
                res = Result("medrag_okf", question, "clarification", answer="I need a little more detail before I search for evidence.",
                             clarifying_questions=qs, trace=trace, safety_level=triage.level,
                             safety_category=triage.category, safety_message=triage.message)
                res.latency_ms = _ms(t0)
                return res
        else:
            trace.append({"stage": "Query analysis", "ms": _ms(t), "detail": "follow-up merged with earlier question"})

        # 3. hybrid retrieval + re-ranking ---------------------------------
        t = time.perf_counter()
        hits = self.index.hybrid_retrieve(question, k, graph_expand=graph)
        n_graph = sum(1 for h in hits if h.ranks.get("graph") and len(h.ranks) == 1)
        trace.append({"stage": "Hybrid retrieval + re-rank", "ms": _ms(t),
                      "detail": f"BM25 + dense fused (RRF) -> {self.index.reranker.name}"
                                + (f"; OKF graph expansion on ({n_graph} passage(s) reached via links)" if graph else "")})

        ev = self.index.evidence_score(question, hits)
        label = _evidence_label(ev)
        trace.append({"stage": "Evidence score", "ms": 0.0, "detail": f"{ev:.2f} ({label}); threshold {settings.min_evidence_score}"})

        prov, related = okf_provenance(self.bundle, hits)
        base = dict(evidence_score=ev, evidence_label=label, sources=[asdict(s) for s in _sources(hits, self.bundle)],
                    provenance=prov, related=related,
                    safety_level=triage.level, safety_category=triage.category, safety_message=triage.message)

        # 4. evidence gate: insufficient evidence -> say so, don't guess -----
        if ev < settings.min_evidence_score:
            res = Result("medrag_okf", question, "insufficient_evidence", answer=INSUFFICIENT, trace=trace, **base)
            res.latency_ms = _ms(t0)
            return res

        # 5. generation -----------------------------------------------------
        t = time.perf_counter()
        answer, gen = generate(question, hits, use_llm)
        trace.append({"stage": "Generation", "ms": _ms(t), "detail": gen})
        if answer.strip().startswith("INSUFFICIENT_EVIDENCE"):
            res = Result("medrag_okf", question, "insufficient_evidence", answer=INSUFFICIENT, generator=gen, trace=trace, **base)
            res.latency_ms = _ms(t0)
            return res

        # 6. evidence verification ----------------------------------------
        t = time.perf_counter()
        v: Verification = verify(answer, hits)
        trace.append({"stage": "Evidence verification", "ms": _ms(t),
                      "detail": f"{sum(c.supported for c in v.claims)}/{len(v.claims)} claims supported ({v.method})"})
        claims = [asdict(c) for c in v.claims]

        if v.supported_rate < settings.min_supported_rate:
            res = Result("medrag_okf", question, "insufficient_evidence", answer=INSUFFICIENT, claims=claims, supported_rate=v.supported_rate,
                         generator=gen, verifier=v.method, trace=trace, **base)
            res.latency_ms = _ms(t0)
            return res

        status = "answered"
        removed = []
        if v.unsupported:
            # safer path: drop unsupported statements, keep the supported ones
            kept = [c for c in v.claims if c.supported]
            removed = [c.claim for c in v.unsupported]
            answer = _rebuild(answer, [c.claim for c in kept])
            status = "answered_with_flags"
        res = Result("medrag_okf", question, status, answer=answer, claims=claims, removed_claims=removed, supported_rate=v.supported_rate,
                     generator=gen, verifier=v.method, trace=trace, **base)
        res.latency_ms = _ms(t0)
        return res


def _rebuild(original: str, kept_claims: list[str]) -> str:
    """Keep only the original sentences (with their citations) whose text matches a supported claim."""
    from .textutil import split_sentences

    keep = set(kept_claims)
    return " ".join(s for s in split_sentences(original) if strip_citations(s) in keep)


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)
