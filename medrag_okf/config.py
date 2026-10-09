"""Central configuration. Everything can be overridden from a .env file or environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # ---- paths -----------------------------------------------------------
    okf_dir: Path = field(default_factory=lambda: Path(_env("OKF_DIR", str(ROOT / "data" / "okf"))))   # the OKF bundle
    corpus_dir: Path = ROOT / "data" / "corpus"        # optional extra plain .md/.txt/.pdf documents
    upload_dir: Path = ROOT / "data" / "uploads"
    eval_file: Path = ROOT / "data" / "eval" / "test_set.json"
    eval_results_file: Path = ROOT / "data" / "eval" / "results.json"

    # ---- chunking --------------------------------------------------------
    chunk_chars: int = 650
    chunk_overlap_sentences: int = 1

    # ---- retrieval -------------------------------------------------------
    # "auto" = sentence-transformers if installed and the model can load, else TF-IDF char n-grams
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    # "auto" = cross-encoder if available, else a transparent heuristic re-ranker
    reranker_backend: str = field(default_factory=lambda: _env("RERANKER_BACKEND", "auto"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    # OKF: also consider the best passage of concepts linked to the top concept (set GRAPH_EXPAND=1 to default on)
    graph_expand: bool = field(default_factory=lambda: _env("GRAPH_EXPAND", "0").lower() in ("1", "true", "yes"))
    candidate_pool: int = 15   # candidates taken from each retriever before fusion
    top_k: int = 4             # passages sent to the LLM / shown to the user

    # ---- trust layer thresholds -----------------------------------------
    min_evidence_score: float = field(default_factory=lambda: float(_env("MIN_EVIDENCE_SCORE", "0.45")))
    claim_support_threshold: float = 0.55   # fraction of a claim's content terms found in one passage
    min_supported_rate: float = 0.50        # below this the whole answer is replaced by the safe fallback

    # ---- LLM -------------------------------------------------------------
    # none | openai_compat (OpenRouter, OpenAI, Groq, Gemini-compat, Ollama...) | anthropic
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "none").lower())
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free"))
    llm_timeout: int = 60
    llm_max_tokens: int = 500
    # "lexical" (fast, offline) or "llm" (LLM-as-judge, more accurate, slower)
    verifier: str = field(default_factory=lambda: _env("VERIFIER", "lexical").lower())

    def llm_enabled(self) -> bool:
        return self.llm_provider in {"openai_compat", "anthropic"} and bool(self.llm_api_key or "localhost" in self.llm_base_url)


settings = Settings()
