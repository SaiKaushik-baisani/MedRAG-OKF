"""Answer generation. Uses an LLM when configured, otherwise an offline extractive generator
so the whole pipeline (and the demo) works with no API key and no internet."""
from __future__ import annotations

import re

import requests

from .config import settings
from .retrieval import Hit
from .textutil import content_terms, expand_query, split_sentences, tokenize

SYSTEM_PROMPT = """You are MedRAG_OKF, a careful medical information assistant.
Rules:
1. Answer ONLY using the numbered context passages. Never use outside knowledge.
2. Write 2-5 short sentences in plain language. Stay close to the wording of the passages.
3. Put the passage number(s) in square brackets at the end of every sentence, e.g. [1] or [1][3].
4. Do not diagnose the user and do not give personal medicine doses.
5. If the passages do not contain enough information, reply with exactly: INSUFFICIENT_EVIDENCE
6. Mention when to see a doctor only if the passages say so."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{i}] ({h.chunk.title} - {h.chunk.section})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
    return f"Context passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"


def llm_complete(system: str, user: str, max_tokens: int | None = None) -> str:
    max_tokens = max_tokens or settings.llm_max_tokens
    if settings.llm_provider == "anthropic":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": settings.llm_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": settings.llm_model, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": user}], "temperature": 0.1},
            timeout=settings.llm_timeout,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()
    # OpenAI-compatible (OpenRouter, OpenAI, Groq, Gemini compat endpoint, Ollama, LM Studio ...)
    headers = {"content-type": "application/json"}
    if settings.llm_api_key:
        headers["authorization"] = f"Bearer {settings.llm_api_key}"
    r = requests.post(
        settings.llm_base_url.rstrip("/") + "/chat/completions",
        headers=headers,
        json={
            "model": settings.llm_model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=settings.llm_timeout,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 5) -> str:
    """Offline generator: pick the passage sentences that best match the question and cite them."""
    q_terms = set(content_terms(question)) | set(expand_query(question))
    q_all = set(tokenize(question))
    scored = []
    for rank, h in enumerate(hits, 1):
        # intent bonus: the passage section (Symptoms / Causes / Treatment...) matches what the user asked
        sec_bonus = 1.5 if q_all & set(tokenize(h.chunk.section)) else 0.0
        for pos, s in enumerate(split_sentences(h.chunk.text)):
            if len(s) < 25:
                continue
            terms = set(content_terms(s))
            overlap = len(q_terms & terms)
            # prefer relevant sentences from higher-ranked passages; mild bonus for the first sentence of a passage
            score = overlap + sec_bonus + 0.6 * h.score + (0.3 if pos == 0 else 0) - 0.05 * rank
            scored.append((score, rank, pos, s, overlap + sec_bonus))
    scored.sort(reverse=True)
    picked, seen = [], set()
    for score, rank, pos, s, overlap in scored:
        if overlap == 0 and len(picked) >= 2:
            continue
        key = s.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        picked.append((rank, pos, s))
        if len(picked) >= max_sentences:
            break
    picked.sort()
    return " ".join(f"{s.rstrip('.')}. [{rank}]" if not s.endswith((".", "!", "?")) else f"{s} [{rank}]" for rank, _, s in picked)


def generate(question: str, hits: list[Hit], use_llm: bool = True) -> tuple[str, str]:
    """Returns (answer_text, generator_name). Falls back to extractive on any LLM failure."""
    if use_llm and settings.llm_enabled():
        try:
            text = llm_complete(SYSTEM_PROMPT, build_prompt(question, hits))
            if text:
                return text, f"LLM ({settings.llm_model})"
        except Exception as exc:
            print(f"[generation] LLM call failed ({type(exc).__name__}: {exc}); using extractive fallback")
            return extractive_answer(question, hits), "extractive (LLM call failed)"
    return extractive_answer(question, hits), "extractive (offline)"


_CITE = re.compile(r"\s*(\[\d+\])+")


def strip_citations(text: str) -> str:
    return _CITE.sub("", text).strip()
