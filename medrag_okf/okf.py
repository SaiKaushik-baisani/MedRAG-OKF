"""Open Knowledge Format (OKF) consumer, validator and helpers.

OKF (Google Cloud, v0.1 June 2026) represents knowledge as a directory ("bundle") of markdown files:
  * one concept per file; the file path IS the concept's identity (e.g. conditions/type2_diabetes.md)
  * YAML frontmatter: only `type` is required; title, description, resource, tags, timestamp are recommended;
    unknown keys must be tolerated (backward-compatible growth)
  * concepts link to each other with ordinary markdown links -> the directory becomes a graph
  * optional reserved files: index.md (progressive disclosure) and log.md (change history)

Trust fields used here (sources, generated, verified, status, stale_after, not) follow the v0.2 trust-signal
proposals described in public write-ups of OKF; they are plain extra frontmatter keys, so any v0.1
consumer simply ignores them.

Identity convention in this code base: a concept's `path` is its bundle-relative file path WITHOUT ".md"
(e.g. "conditions/asthma"); that string is also the `doc_id` of its retrieval chunks.
"""
from __future__ import annotations

import datetime as dt
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path

RESERVED = {"index.md", "log.md"}
_FRONT = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s#]+\.md)(?:#[^)]*)?\)")


# =============================================================================== YAML (small subset)
def _scalar(v: str):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1].replace('\\"', '"')
    if v.startswith("[") and v.endswith("]"):
        return [_scalar(x) for x in _split_top(v[1:-1])]
    if v.startswith("{") and v.endswith("}"):
        out = {}
        for part in _split_top(v[1:-1]):
            k, _, val = part.partition(":")
            out[k.strip().strip("'\"")] = _scalar(val)
        return out
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    return v


def _split_top(s: str) -> list[str]:
    out, depth, cur, quote = [], 0, [], ""
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [x.strip() for x in out]


def _mini_yaml(text: str) -> dict:
    """Small YAML subset: scalars, flow lists/maps, nested mappings and lists of mappings by indentation."""
    lines = [(len(r) - len(r.lstrip(" ")), r.strip()) for r in text.splitlines() if r.strip() and not r.strip().startswith("#")]

    def block(i: int, indent: int):
        if i >= len(lines):
            return None, i
        if lines[i][1].startswith("- ") or lines[i][1] == "-":
            out = []
            while i < len(lines) and lines[i][0] == indent and (lines[i][1].startswith("- ") or lines[i][1] == "-"):
                item = lines[i][1][1:].strip()
                if item == "":
                    if i + 1 < len(lines) and lines[i + 1][0] > indent:
                        val, i = block(i + 1, lines[i + 1][0])
                    else:
                        val, i = "", i + 1
                    out.append(val)
                elif re.match(r"""^[^\s'"\[{][^:]*:(\s|$)""", item):          # "- key: value": a mapping item
                    lines[i] = (indent + 2, item)
                    val, i = block(i, indent + 2)
                    out.append(val)
                else:
                    out.append(_scalar(item))
                    i += 1
            return out, i
        out = {}
        while i < len(lines) and lines[i][0] == indent and not lines[i][1].startswith("- "):
            key, _, rest = lines[i][1].partition(":")
            key, rest = key.strip(), rest.strip()
            if rest == "":
                nxt = lines[i + 1] if i + 1 < len(lines) else None
                if nxt and (nxt[0] > indent or (nxt[0] == indent and nxt[1].startswith("- "))):
                    out[key], i = block(i + 1, nxt[0])
                else:
                    out[key], i = "", i + 1
            else:
                out[key] = _scalar(rest)
                i += 1
        return out, i

    if not lines:
        return {}
    val, _ = block(0, lines[0][0])
    return val if isinstance(val, dict) else {}


def parse_yaml(text: str) -> dict:
    try:
        import yaml                                  # optional; used when installed

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except ImportError:
        pass
    except Exception:
        pass
    return _mini_yaml(text)


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONT.match(raw)
    if not m:
        return {}, raw
    return parse_yaml(m.group(1)), raw[m.end():]


def _as_date(v) -> dt.datetime | None:
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day, tzinfo=dt.timezone.utc)
    if not v:
        return None
    s = str(v).strip().replace("Z", "+00:00")
    try:
        d = dt.datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _list(v) -> list:
    if v is None or v == "":
        return []
    return v if isinstance(v, list) else [v]


# =============================================================================== model
@dataclass
class Concept:
    path: str                    # identity: bundle-relative path without ".md", e.g. "conditions/asthma"
    file: Path
    meta: dict
    body: str
    links: list[str] = field(default_factory=list)          # resolved outgoing concept paths
    link_text: dict = field(default_factory=dict)           # target path -> anchor text
    broken_links: list[str] = field(default_factory=list)

    @property
    def type(self) -> str:
        return str(self.meta.get("type") or "")

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.path.rsplit("/", 1)[-1].replace("_", " ").title())

    @property
    def description(self) -> str:
        return str(self.meta.get("description") or "")

    @property
    def tags(self) -> list[str]:
        return [str(t) for t in _list(self.meta.get("tags"))]

    @property
    def status(self) -> str:
        return str(self.meta.get("status") or "unspecified")

    @property
    def verified(self) -> list[dict]:
        return [v if isinstance(v, dict) else {"by": str(v)} for v in _list(self.meta.get("verified"))]

    @property
    def generated_by(self) -> str:
        g = self.meta.get("generated")
        return str(g.get("by", "")) if isinstance(g, dict) else ""

    @property
    def sources(self) -> list:
        return _list(self.meta.get("sources"))

    @property
    def stale_after_text(self) -> str:
        d = _as_date(self.meta.get("stale_after"))
        return d.strftime("%Y-%m-%d") if d else str(self.meta.get("stale_after") or "")

    @property
    def not_covered(self) -> list[str]:
        """OKF "negative knowledge": what this concept explicitly does NOT cover."""
        n = self.meta.get("not")
        return [str(x) for x in _list(n)] if not isinstance(n, dict) else [str(x) for x in n.values()]

    def source_label(self) -> str:
        """Short human-readable provenance string for UI display."""
        if self.sources:
            s = self.sources[0]
            if isinstance(s, dict):
                title = str(s.get("title") or "")
                extra = str(s.get("ref") or s.get("url") or "")
                return f"{title} ({extra})" if title and extra else (title or extra or f"OKF concept: {self.path}.md")
            return str(s)
        if self.meta.get("resource"):
            return str(self.meta["resource"])
        return f"OKF concept: {self.path}.md"

    def is_stale(self, now: dt.datetime | None = None) -> bool:
        d = _as_date(self.meta.get("stale_after"))
        return bool(d and d < (now or dt.datetime.now(dt.timezone.utc)))

    def is_verified(self) -> bool:
        return any(str(v.get("by", "")).startswith("human:") for v in self.verified)

    def sections(self) -> list[str]:
        return [ln[3:].strip() for ln in self.body.splitlines() if ln.startswith("## ")]

    def trust(self) -> dict:
        """Compact trust summary (status / verification / freshness) used by the API and UI."""
        return {
            "path": self.path, "title": self.title, "type": self.type, "tags": self.tags, "status": self.status,
            "verified": self.is_verified(), "verified_by": [v.get("by", "") for v in self.verified],
            "stale": self.is_stale(), "stale_after": self.stale_after_text,
            "generated_by": self.generated_by, "source": self.source_label(), "not": self.not_covered,
            "timestamp": str(self.meta.get("timestamp") or ""),
        }


@dataclass
class Issue:
    level: str       # error | warning
    path: str
    message: str


@dataclass
class Bundle:
    root: Path
    concepts: dict[str, Concept] = field(default_factory=dict)
    index: str = ""                                           # root index.md
    log: str = ""                                             # root log.md
    indexes: dict[str, str] = field(default_factory=dict)     # directory -> index.md text (progressive disclosure)

    def backlinks(self) -> dict[str, list[str]]:
        back: dict[str, list[str]] = {}
        for c in self.concepts.values():
            for t in c.links:
                back.setdefault(t, []).append(c.path)
        return back

    def edges(self) -> list[tuple[str, str]]:
        return [(c.path, t) for c in self.concepts.values() for t in c.links]

    def types(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.concepts.values():
            out[c.type or "(none)"] = out.get(c.type or "(none)", 0) + 1
        return out

    def neighbors(self, path: str) -> list[str]:
        """Concepts directly linked to or from `path` (undirected one-hop graph neighbourhood)."""
        c = self.concepts.get(path)
        out = list(c.links) if c else []
        out += [p for p in self.backlinks().get(path, []) if p not in out]
        return out


def resolve_link(from_path: str, href: str) -> str:
    """OKF links are ordinary markdown links to other concept files: bundle-root-relative ('/tables/x.md')
    or relative to the current file. Returns the target concept path without '.md'."""
    if href.startswith("/"):
        target = posixpath.normpath(href[1:])
    else:
        target = posixpath.normpath(posixpath.join(posixpath.dirname(from_path), href))
    return target[:-3] if target.endswith(".md") else target


def load_bundle(root: Path) -> Bundle:
    root = Path(root)
    b = Bundle(root)
    if not root.exists():
        return b
    for f in sorted(root.rglob("*.md")):
        rel = f.relative_to(root).as_posix()
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if f.name in RESERVED:
            if f.parent == root:
                if f.name == "index.md":
                    b.index = text
                else:
                    b.log = text
            elif f.name == "index.md":
                b.indexes[f.parent.relative_to(root).as_posix()] = text
            continue
        meta, body = parse_frontmatter(text)
        path = rel[:-3]
        b.concepts[path] = Concept(path, f, meta, body.strip())
    for c in b.concepts.values():
        for text_, href in _LINK.findall(c.body):
            if re.match(r"^[a-z][a-z0-9+.-]*://", href):
                continue
            target = resolve_link(c.path, href)
            if target in b.concepts:
                if target not in c.links and target != c.path:
                    c.links.append(target)
                c.link_text[target] = text_
            else:
                c.broken_links.append(target)
    return b


def validate(b: Bundle) -> list[Issue]:
    """Conformance (v0.1): every concept has a `type`. Everything else is advisory."""
    issues: list[Issue] = []
    if not b.concepts:
        issues.append(Issue("error", "", "bundle contains no concept files"))
    for c in b.concepts.values():
        if not c.meta:
            issues.append(Issue("error", c.path, "no YAML frontmatter"))
            continue
        if not c.type:
            issues.append(Issue("error", c.path, "missing required `type` field"))
        for key in ("title", "description", "tags"):
            if not c.meta.get(key):
                issues.append(Issue("warning", c.path, f"recommended field `{key}` is missing"))
        for t in c.broken_links:
            issues.append(Issue("warning", c.path, f"link to missing concept `{t}.md`"))
        if c.is_stale():
            issues.append(Issue("warning", c.path, f"content is past its stale_after date ({c.stale_after_text})"))
        if c.status == "stable" and not c.is_verified():
            issues.append(Issue("warning", c.path, "status is `stable` but no human verification is recorded"))
    order = {"error": 0, "warning": 1}
    return sorted(issues, key=lambda i: (order.get(i.level, 2), i.path))


def verify_concept(file: Path, by: str, when: dt.datetime | None = None) -> dict:
    """Stamp a concept as human-verified: adds a `verified` entry and sets status: stable (text-level edit).
    Earlier verification entries are kept."""
    file = Path(file)
    when = when or dt.datetime.now(dt.timezone.utc)
    raw = file.read_text(encoding="utf-8")
    m = _FRONT.match(raw)
    if not m:
        raise ValueError(f"{file} has no frontmatter")
    fm, rest = m.group(1), raw[m.end():]
    meta = parse_yaml(fm)
    entry = {"by": by if by.startswith(("human:", "agent:")) else f"human:{by}", "at": when.strftime("%Y-%m-%dT%H:%M:%SZ")}
    previous = [v if isinstance(v, dict) else {"by": str(v)} for v in _list(meta.get("verified"))]
    for v in previous:                       # PyYAML turns timestamps into datetimes; write them back as ISO text
        for k, val in list(v.items()):
            if isinstance(val, (dt.datetime, dt.date)):
                v[k] = val.strftime("%Y-%m-%dT%H:%M:%SZ")
    fm = re.sub(r"^verified:.*(?:\n[ \t]+.*)*\n?", "", fm, flags=re.M).rstrip("\n")
    if re.search(r"^status:", fm, flags=re.M):
        fm = re.sub(r"^status:.*$", "status: stable", fm, flags=re.M)
    else:
        fm += "\nstatus: stable"
    block = "\nverified:"
    for v in previous + [entry]:
        first = True
        for k, val in v.items():
            block += f"\n  {'- ' if first else '  '}{k}: {val}"
            first = False
    file.write_text("---\n" + fm + block + "\n---\n" + rest, encoding="utf-8")
    return entry


def import_markdown(src: Path, dst: Path, concept_type: str = "Document") -> list[Path]:
    """Wrap plain .md/.txt files as OKF concepts (type + provenance fields), written to `dst`."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    written = []
    for f in sorted(Path(src).iterdir()):
        if f.suffix.lower() not in (".md", ".txt") or f.name in RESERVED:
            continue
        meta, body = parse_frontmatter(f.read_text(encoding="utf-8", errors="ignore"))
        title = str(meta.get("title") or f.stem.replace("_", " ").title()).replace('"', "'")
        first = next((ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")), "")
        desc = str(meta.get("description") or first[:160]).replace('"', "'")
        out = dst / (f.stem + ".md")
        out.write_text(
            f'---\ntype: {meta.get("type") or concept_type}\ntitle: "{title}"\ndescription: "{desc}"\ntags: []\n'
            f"timestamp: {now}\nstatus: draft\ngenerated:\n  by: okf-import\n  at: {now}\n---\n\n{body.strip()}\n",
            encoding="utf-8",
        )
        written.append(out)
    return written
