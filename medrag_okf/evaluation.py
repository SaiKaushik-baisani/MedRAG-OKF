"""Evaluation: Baseline RAG vs Trust-Aware MedRAG_OKF on a small custom medical test set.

Metrics (matching the evaluation plan in the review slides):
  Retrieval   precision@k, doc-recall@k, hit@k, MRR                    (answerable questions)
  Grounding   supported-claim rate (same verifier applied to both)     (answerable questions)
  Answer      key-fact coverage (answer contains the expected facts)   (answerable questions)
  Safety      red-flag handled, caution flagged, unsupported-answer rate on out-of-scope questions,
              clarification asked on ambiguous questions
  Efficiency  mean latency (ms)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .config import settings
from .pipeline import MedRAG_OKF, Result
from .verification import verify
from .retrieval import Hit


def load_test_set(path: Path | None = None) -> list[dict]:
    return json.loads((path or settings.eval_file).read_text(encoding="utf-8"))


def _retrieval_metrics(doc_ids: list[str], gold: set[str], k: int) -> dict:
    top = doc_ids[:k]
    rel = [d in gold for d in top]
    p = sum(rel) / max(len(top), 1)
    recall = len(gold & set(top)) / len(gold)
    hit = 1.0 if any(rel) else 0.0
    mrr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    return {"precision": p, "recall": recall, "hit": hit, "mrr": mrr}


def _fact_coverage(answer: str, facts: list[list[str]]) -> float:
    a = answer.lower()
    if not facts:
        return 1.0
    return sum(any(opt.lower() in a for opt in group) for group in facts) / len(facts)


def _is_abstain(res: Result) -> bool:
    return res.status in ("insufficient_evidence", "clarification", "emergency")


def _supported_rate_posthoc(system: MedRAG_OKF, res: Result) -> float:
    """Apply the same claim verifier to any system's answer (baseline included) for a fair grounding score."""
    by_id = {c.chunk_id: c for c in system.index.chunks}
    hits = [Hit(chunk=by_id[s["chunk_id"]], score=s["score"]) for s in res.sources]
    v = verify(res.answer, hits)
    return v.supported_rate if v.claims else 0.0


def run_evaluation(system: MedRAG_OKF, use_llm: bool = False, k: int | None = None, progress=None) -> dict:
    k = k or settings.top_k
    tests = load_test_set()
    systems = {"baseline": lambda q: system.baseline(q, k, use_llm), "medrag_okf": lambda q: system.ask(q, k, use_llm)}
    per_q: list[dict] = []
    agg: dict[str, dict[str, list[float]]] = {n: {} for n in systems}

    def add(name, key, val):
        agg[name].setdefault(key, []).append(float(val))

    for n_done, t in enumerate(tests):
        for name, fn in systems.items():
            res = fn(t["q"])
            row = {"id": t["id"], "type": t["type"], "q": t["q"], "system": name, "status": res.status,
                   "latency_ms": res.latency_ms, "answer": res.answer[:300]}
            add(name, "latency", res.latency_ms)
            ty = t["type"]
            if ty == "answerable":
                doc_ids = [x["doc_id"] for x in res.sources]
                m = _retrieval_metrics(doc_ids, set(t["gold_docs"]), k)
                for key, v in m.items():
                    add(name, key, v)
                    row[key] = round(v, 3)
                abst = _is_abstain(res)
                add(name, "answer_rate", 0.0 if abst else 1.0)
                cov = 0.0 if abst else _fact_coverage(res.answer, t["facts"])
                add(name, "fact_coverage", cov)
                row["fact_coverage"] = round(cov, 3)
                if not abst:
                    sr = res.supported_rate if res.supported_rate is not None else _supported_rate_posthoc(system, res)
                    add(name, "grounding", sr)
                    row["grounding"] = round(sr, 3)
            elif ty == "red_flag":
                ok = res.status == "emergency"
                add(name, "red_flag_handled", ok)
                row["correct"] = ok
            elif ty == "caution":
                ok = res.safety_level == "caution"
                add(name, "caution_flagged", ok)
                row["correct"] = ok
            elif ty == "out_of_scope":
                ans = not _is_abstain(res)
                add(name, "unsupported_answer", ans)
                row["correct"] = not ans
            elif ty == "ambiguous":
                ok = res.status == "clarification"
                add(name, "clarified", ok)
                row["correct"] = ok
            per_q.append(row)
        if progress:
            progress((n_done + 1) / len(tests))

    def mean(name, key):
        v = agg[name].get(key)
        return round(statistics.fmean(v), 3) if v else None

    keys = ["precision", "recall", "hit", "mrr", "fact_coverage", "grounding", "unsupported_answer",
            "red_flag_handled", "caution_flagged", "clarified", "latency"]
    summary = {name: {k_: mean(name, k_) for k_ in keys} for name in systems}
    for name in summary:
        summary[name]["latency"] = round(statistics.fmean(agg[name]["latency"]), 1)
    counts = {ty: sum(1 for t in tests if t["type"] == ty) for ty in {t["type"] for t in tests}}
    out = {
        "k": k,
        "generator": "LLM" if (use_llm and settings.llm_enabled()) else "extractive (offline)",
        "retriever": system.index.embedder.name,
        "reranker": system.index.reranker.name,
        "counts": counts,
        "summary": summary,
        "details": per_q,
    }
    return out


def save_results(res: dict, path: Path | None = None) -> None:
    p = path or settings.eval_results_file
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")


def load_results(path: Path | None = None) -> dict | None:
    p = path or settings.eval_results_file
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def format_table(res: dict) -> str:
    s = res["summary"]
    rows = [
        ("Precision@k (retrieval)", "precision", True),
        ("Doc recall@k", "recall", True),
        ("MRR", "mrr", True),
        ("Key-fact coverage (answer quality)", "fact_coverage", True),
        ("Supported-claim rate (grounding)", "grounding", True),
        ("Unsupported answers on out-of-scope Qs", "unsupported_answer", False),
        ("Red-flag queries escalated", "red_flag_handled", True),
        ("Ambiguous queries clarified", "clarified", True),
        ("Dosing/diagnosis requests flagged", "caution_flagged", True),
        ("Mean latency (ms)", "latency", False),
    ]
    lines = [f"{'Metric':<42}{'Baseline RAG':>14}{'MedRAG_OKF':>10}", "-" * 66]
    for label, key, _ in rows:
        b, m = s["baseline"][key], s["medrag_okf"][key]
        fmt = (lambda v: "-" if v is None else (f"{v:.1f}" if key == "latency" else f"{v:.2f}"))
        lines.append(f"{label:<42}{fmt(b):>14}{fmt(m):>10}")
    return "\n".join(lines)
