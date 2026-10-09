"""Hybrid retrieval (BM25 keyword + dense semantic, fused with Reciprocal Rank Fusion) and re-ranking.

Every component is swappable:
  * Embedder   -> sentence-transformers (semantic) or TF-IDF char n-grams (offline fallback)
  * Reranker   -> cross-encoder (if installed) or a transparent heuristic blend
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .corpus import Chunk
from .textutil import content_groups, content_terms, expand_query, tokenize


# --------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.df = df
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)

    def idf_of(self, term: str) -> float:
        return self.idf.get(term, self.max_idf)

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for t in set(query):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


# --------------------------------------------------------------------------- Embedders
class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> None: ...

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Offline fallback. Character n-grams give some tolerance to word forms (tired / tiredness)."""

    name = "tfidf-char-ngrams (offline fallback)"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)

    def fit(self, texts):
        self.vec.fit(texts)

    def encode(self, texts):
        m = self.vec.transform(texts)
        return m.toarray().astype(np.float32)  # already L2-normalised


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers: {model_name.split('/')[-1]}"

    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


def make_embedder() -> Embedder:
    backend = settings.embedding_backend
    if backend in ("auto", "st", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:
            if backend != "auto":
                raise
            print(f"[retrieval] sentence-transformers unavailable ({type(exc).__name__}); using TF-IDF fallback")
    return TfidfEmbedder()


# --------------------------------------------------------------------------- Rerankers
class Reranker:
    name = "base"

    def score(self, query: str, chunks: list[Chunk], feats: list[dict]) -> list[float]:
        raise NotImplementedError


class HeuristicReranker(Reranker):
    """Transparent blend: dense similarity, BM25, query-term coverage and a title/section match bonus."""

    name = "heuristic re-ranker (dense + BM25 + coverage + title match)"

    def score(self, query, chunks, feats):
        q_terms = set(content_terms(query)) | set(expand_query(query))
        out = []
        for c, f in zip(chunks, feats):
            title_terms = set(tokenize(c.title + " " + c.section))
            title_hit = 1.0 if q_terms & title_terms else 0.0
            out.append(0.40 * f["dense_norm"] + 0.30 * f["bm25_norm"] + 0.20 * f["coverage"] + 0.10 * title_hit)
        return out


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.name = f"cross-encoder: {model_name.split('/')[-1]}"

    def score(self, query, chunks, feats):
        raw = np.asarray(self.model.predict([(query, c.index_text) for c in chunks]), dtype=float)
        # squash logits to 0..1 and blend lightly with the heuristic so a bad model can't dominate
        sig = 1 / (1 + np.exp(-raw))
        heur = np.array([0.5 * f["dense_norm"] + 0.5 * f["coverage"] for f in feats])
        return list(0.7 * sig + 0.3 * heur)


def make_reranker() -> Reranker:
    if settings.reranker_backend in ("auto", "cross-encoder"):
        try:
            return CrossEncoderReranker(settings.reranker_model)
        except Exception as exc:
            if settings.reranker_backend != "auto":
                raise
            print(f"[retrieval] cross-encoder unavailable ({type(exc).__name__}); using heuristic re-ranker")
    return HeuristicReranker()


# --------------------------------------------------------------------------- Retrieved item
@dataclass
class Hit:
    chunk: Chunk
    score: float                      # final (re-ranked) score, 0..1-ish
    dense: float = 0.0                # raw cosine
    bm25: float = 0.0                 # raw BM25
    coverage: float = 0.0             # IDF-weighted fraction of query terms present
    rrf: float = 0.0
    ranks: dict = field(default_factory=dict)
    idx: int = -1


# --------------------------------------------------------------------------- Index + retrievers
class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, reranker: Reranker | None = None):
        if not chunks:
            raise ValueError("Corpus is empty - add documents to data/corpus")
        self.chunks = chunks
        self.embedder = embedder or make_embedder()
        self.reranker = reranker or make_reranker()
        texts = [c.index_text for c in chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.encode(texts)             # vector store (in-memory, cosine)
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.chunk_terms = [set(tokenize(t)) for t in texts]
        self.graph: dict[str, set[str]] = {}                   # OKF concept graph: doc_id -> neighbouring doc_ids
        self.doc_chunks: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.doc_chunks.setdefault(c.doc_id, []).append(i)

    def set_graph(self, neighbors: dict[str, list[str]]) -> None:
        self.graph = {k: set(v) for k, v in neighbors.items()}

    @property
    def doc_count(self) -> int:
        return len({c.doc_id for c in self.chunks})

    # ---- individual retrievers
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query])[0]
        sims = self.vectors @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def bm25_search(self, query: str, k: int, expand: bool = True) -> list[tuple[int, float]]:
        toks = expand_query(query) if expand else tokenize(query)
        sc = self.bm25.scores(toks)
        order = np.argsort(-sc)[:k]
        return [(int(i), float(sc[i])) for i in order if sc[i] > 0]

    def _group_idf(self, g: set[str]) -> float:
        return max(self.bm25.idf_of(t) for t in g)

    def coverage(self, query: str, idxs: list[int]) -> float:
        """IDF-weighted share of the query's topical terms (or their synonyms) found in the given passages."""
        groups = content_groups(query)
        if not groups:
            return 0.0
        covered = set()
        for i in idxs:
            covered |= self.chunk_terms[i]
        w_all = sum(self._group_idf(g) for g in groups)
        w_hit = sum(self._group_idf(g) for g in groups if g & covered)
        return w_hit / w_all if w_all else 0.0

    # ---- baseline: plain dense top-k (what a vanilla RAG does)
    def baseline_retrieve(self, query: str, k: int) -> list[Hit]:
        res = self.dense_search(query, k)
        return [Hit(self.chunks[i], score=s, dense=s, idx=i) for i, s in res]

    # ---- MedRAG_OKF: hybrid + re-rank
    def hybrid_retrieve(self, query: str, k: int, pool: int | None = None, graph_expand: bool = False) -> list[Hit]:
        pool = pool or settings.candidate_pool
        dense = self.dense_search(query, pool)
        sparse = self.bm25_search(query, pool)
        rrf: dict[int, float] = {}
        ranks: dict[int, dict] = {}
        for name, lst in (("dense", dense), ("bm25", sparse)):
            for rank, (i, _) in enumerate(lst, 1):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank)
                ranks.setdefault(i, {})[name] = rank
        cand = sorted(rrf, key=rrf.get, reverse=True)[: pool]
        if not cand:
            return []

        q_vec = self.embedder.encode([query])[0]
        bm_all = self.bm25.scores(expand_query(query))

        # OKF graph expansion: concepts linked to the best-matching concept contribute their best passage as an
        # extra candidate (e.g. "tired all the time" -> fatigue -> anemia / thyroid). The re-ranker still decides.
        if graph_expand and self.graph:
            dense_all = self.vectors @ q_vec
            d_top, b_top = max(float(dense_all.max()), 1e-9), max(float(bm_all.max()), 1e-9)
            for nb in sorted(self.graph.get(self.chunks[cand[0]].doc_id, ())):
                idxs = self.doc_chunks.get(nb, [])
                if not idxs:
                    continue
                best = max(idxs, key=lambda j: max(float(dense_all[j]), 0) / d_top + float(bm_all[j]) / b_top)
                if best not in cand:
                    cand.append(best)
                    rrf.setdefault(best, 0.0)
                    ranks.setdefault(best, {})["graph"] = 1

        dense_raw = {i: float(self.vectors[i] @ q_vec) for i in cand}
        bm_raw = {i: float(bm_all[i]) for i in cand}
        d_max = max(max(dense_raw.values()), 1e-9)
        b_max = max(max(bm_raw.values()), 1e-9)

        feats = []
        for i in cand:
            feats.append(
                {
                    "dense_norm": max(dense_raw[i], 0) / d_max,
                    "bm25_norm": bm_raw[i] / b_max,
                    "coverage": self.coverage(query, [i]),
                }
            )
        scores = self.reranker.score(query, [self.chunks[i] for i in cand], feats)
        hits = [
            Hit(self.chunks[i], score=float(s), dense=dense_raw[i], bm25=bm_raw[i], coverage=f["coverage"], rrf=rrf[i], ranks=ranks.get(i, {}), idx=i)
            for i, s, f in zip(cand, scores, feats)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def oov_share(self, query: str) -> float:
        """IDF-weighted share of the query's topical terms that never occur anywhere in the corpus."""
        groups = content_groups(query)
        if not groups:
            return 0.0
        w_all = sum(self._group_idf(g) for g in groups)
        w_oov = sum(self._group_idf(g) for g in groups if not (g & self.bm25.idf.keys()))
        return w_oov / w_all if w_all else 0.0

    def evidence_score(self, query: str, hits: list[Hit]) -> float:
        """0..1 confidence that the corpus really contains evidence for this query.

        = (0.7 * IDF-weighted query-term coverage in the top-3 passages + 0.3 * best re-rank score)
          reduced when key query terms do not exist anywhere in the knowledge base (e.g. "kidney stones").
        """
        if not hits:
            return 0.0
        idxs = [h.idx for h in hits[:3]]
        cov = self.coverage(query, idxs)
        top = min(1.0, hits[0].score)
        base = 0.7 * cov + 0.3 * top
        return round(base * (1 - 0.6 * self.oov_share(query)), 3)
