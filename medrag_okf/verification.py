"""Evidence verification: after generation, check each claim against the retrieved passages.

Default verifier is lexical + numeric (fast, offline, explainable):
  * a claim is SUPPORTED when most of its topical terms occur together in ONE retrieved passage
  * any number in the claim (e.g. "140/90", "6.5") must also appear in that passage, which catches invented figures
Optional LLM-as-judge verifier (VERIFIER=llm) can be enabled for higher accuracy.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import settings
from .generation import llm_complete, strip_citations
from .retrieval import Hit
from .textutil import content_terms, numbers, split_sentences


@dataclass
class ClaimResult:
    claim: str
    supported: bool
    score: float
    source_rank: int | None   # 1-based index of best supporting passage
    reason: str = ""


@dataclass
class Verification:
    claims: list[ClaimResult]
    supported_rate: float
    method: str

    @property
    def unsupported(self) -> list[ClaimResult]:
        return [c for c in self.claims if not c.supported]


def _lexical_support(claim: str, hits: list[Hit]) -> ClaimResult:
    terms = set(content_terms(claim))
    nums = numbers(claim)
    if not terms:
        return ClaimResult(claim, True, 1.0, None, "no checkable content")
    best, best_rank = 0.0, None
    for rank, h in enumerate(hits, 1):
        ptxt = h.chunk.text
        pterms = set(content_terms(ptxt))
        score = len(terms & pterms) / len(terms)
        if nums and not nums <= numbers(ptxt):
            score *= 0.4   # figure not found in this passage
        if score > best:
            best, best_rank = score, rank
    ok = best >= settings.claim_support_threshold
    reason = "" if ok else ("figure or key terms not found in the retrieved evidence" if nums else "key terms not found together in any retrieved passage")
    return ClaimResult(claim, ok, round(best, 2), best_rank if ok else None, reason)


def _llm_support(claims: list[str], hits: list[Hit]) -> list[ClaimResult] | None:
    ctx = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, 1))
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    system = "You are a strict fact-checking judge. Judge each claim ONLY against the passages. Respond with JSON only."
    user = (
        f"Passages:\n{ctx}\n\nClaims:\n{numbered}\n\n"
        'Return a JSON list, one item per claim, like [{"claim":1,"supported":true,"passage":2}]. '
        '"supported" is true only if a passage states or clearly implies the claim.'
    )
    try:
        raw = llm_complete(system, user, max_tokens=400)
        data = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
        out = []
        for item in data:
            i = int(item["claim"]) - 1
            sup = bool(item["supported"])
            out.append(ClaimResult(claims[i], sup, 1.0 if sup else 0.0, item.get("passage") if sup else None, "" if sup else "judged unsupported by LLM"))
        if len(out) == len(claims):
            return out
    except Exception as exc:
        print(f"[verification] LLM judge failed ({type(exc).__name__}); falling back to lexical")
    return None


def verify(answer: str, hits: list[Hit]) -> Verification:
    claims = [strip_citations(s) for s in split_sentences(answer)]
    claims = [c for c in claims if len(c) > 15]
    if not claims:
        return Verification([], 0.0, "none")
    results = None
    method = "lexical + numeric check"
    if settings.verifier == "llm" and settings.llm_enabled():
        results = _llm_support(claims, hits)
        if results:
            method = "LLM-as-judge"
    if results is None:
        results = [_lexical_support(c, hits) for c in claims]
    rate = sum(r.supported for r in results) / len(results)
    return Verification(results, round(rate, 3), method)
