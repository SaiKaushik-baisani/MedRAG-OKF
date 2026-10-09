"""Knowledge loading and chunking.

The knowledge base is an OKF bundle (data/okf): every concept file becomes a set of passages, one per
section (OKF: "section headings as chunking boundaries"), each carrying the concept's identity (file path),
type, tags and trust metadata. Plain .md/.txt/.pdf files dropped into data/corpus or uploaded in the UI are
wrapped as draft `Document` concepts so they flow through the same pipeline.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import settings
from .okf import Bundle, Concept, load_bundle, parse_frontmatter
from .textutil import split_sentences

NON_EVIDENCE_SECTIONS = {"related concepts"}   # link sections feed the knowledge graph, not retrieval


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str                 # OKF concept path, e.g. "conditions/type2_diabetes"
    title: str
    section: str
    source: str
    text: str
    page: int | None = None
    concept_type: str = ""
    tags: list = field(default_factory=list)

    @property
    def index_text(self) -> str:
        """Text used for retrieval: concept title + section + passage + OKF tags give topical context."""
        return f"{self.title}. {self.section}. {self.text} {' '.join(self.tags)}".strip()

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pack(sentences: list[str], limit: int, overlap: int) -> list[str]:
    """Greedy sentence packing with a small sentence overlap between neighbouring chunks."""
    out, cur = [], []
    for s in sentences:
        if cur and sum(len(x) + 1 for x in cur) + len(s) > limit:
            out.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_text(c: Concept, body: str, source: str, page: int | None = None, start: int = 0) -> list[Chunk]:
    """Split on markdown '## Section' headings, then pack sentences up to chunk_chars."""
    body = _clean(body)
    sections: list[tuple[str, str]] = []
    current, buf = "Overview", []
    for line in body.splitlines():
        if line.startswith("## "):
            if buf:
                sections.append((current, "\n".join(buf)))
            current, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        else:
            buf.append(line)
    if buf:
        sections.append((current, "\n".join(buf)))

    chunks: list[Chunk] = []
    for section, text in sections:
        text = text.strip()
        if not text or section.lower() in NON_EVIDENCE_SECTIONS:
            continue
        for piece in _pack(split_sentences(text), settings.chunk_chars, settings.chunk_overlap_sentences):
            cid = f"{c.path}#{start + len(chunks)}" if page is None else f"{c.path}#p{page}-{start + len(chunks)}"
            chunks.append(Chunk(cid, c.path, c.title, section, source, piece, page, c.type, c.tags))
    return chunks


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def chunk_concept(c: Concept) -> list[Chunk]:
    source = c.source_label()
    if c.file.suffix.lower() == ".pdf":
        out: list[Chunk] = []
        for page_no, text in _read_pdf(c.file):
            out.extend(chunk_text(c, text, source, page=page_no, start=len(out)))
        return out
    return chunk_text(c, c.body, source)


def _wrap_plain_file(path: Path, prefix: str) -> Concept | None:
    """Wrap a non-OKF file as a draft `Document` concept (or honour its own OKF frontmatter if it has a type)."""
    suffix = path.suffix.lower()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta, body = {}, ""
    if suffix in (".md", ".txt"):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8", errors="ignore"))
    elif suffix != ".pdf":
        return None
    if not meta.get("type"):
        meta = {
            **meta,
            "type": "Document",
            "title": meta.get("title") or path.stem.replace("_", " ").title(),
            "status": "draft",
            "timestamp": now,
            "generated": {"by": "user-upload", "at": now},
            "sources": [{"title": f"Uploaded file: {path.name}"}],
        }
    return Concept(f"{prefix}/{path.stem}", path, meta, body.strip())


def load_knowledge() -> tuple[Bundle, list[Chunk]]:
    """Load the OKF bundle plus any plain/uploaded documents; return (bundle, passages)."""
    bundle = load_bundle(settings.okf_dir)
    for directory, prefix in ((settings.corpus_dir, "corpus"), (settings.upload_dir, "uploads")):
        if not directory.exists():
            continue
        for f in sorted(directory.iterdir()):
            if f.name.startswith(".") or f.name.lower().startswith("readme"):
                continue
            try:
                c = _wrap_plain_file(f, prefix)
                if c:
                    bundle.concepts[c.path] = c
            except Exception as exc:  # a bad upload must not break the demo
                print(f"[corpus] skipped {f.name}: {exc}")
    chunks: list[Chunk] = []
    for c in bundle.concepts.values():
        try:
            chunks.extend(chunk_concept(c))
        except Exception as exc:
            print(f"[corpus] could not chunk {c.path}: {exc}")
    return bundle, chunks
