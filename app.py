"""MedRAG_OKF web app - zero extra dependencies (Python standard library HTTP server + one HTML page).

    python app.py                 -> http://localhost:8000
    python app.py --port 8501
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from medrag_okf.config import ROOT, settings
from medrag_okf.evaluation import load_results, run_evaluation, save_results
from medrag_okf.okf import validate, verify_concept
from medrag_okf.pipeline import MedRAG_OKF

STATIC = ROOT / "static"
_lock = threading.Lock()
SYSTEM: MedRAG_OKF | None = None
EVAL_STATE = {"running": False, "error": ""}


def get_system() -> MedRAG_OKF:
    global SYSTEM
    if SYSTEM is None:
        SYSTEM = MedRAG_OKF()
    return SYSTEM


def status_payload() -> dict:
    s = get_system()
    docs: dict[str, dict] = {}
    for c in s.index.chunks:
        con = s.bundle.concepts.get(c.doc_id)
        d = docs.setdefault(c.doc_id, {
            "doc_id": c.doc_id, "title": c.title, "source": c.source, "chunks": 0, "type": c.concept_type,
            "status": con.status if con else "", "verified": bool(con and con.is_verified()),
            "stale": bool(con and con.is_stale()), "uploaded": c.doc_id.startswith("uploads/")})
        d["chunks"] += 1
    return {
        "documents": sorted(docs.values(), key=lambda d: d["title"]),
        "chunks": len(s.index.chunks),
        "embedder": s.index.embedder.name,
        "reranker": s.index.reranker.name,
        "llm_enabled": settings.llm_enabled(),
        "llm_model": settings.llm_model if settings.llm_enabled() else None,
        "verifier": settings.verifier,
        "top_k": settings.top_k,
        "min_evidence_score": settings.min_evidence_score,
        "graph_expand": settings.graph_expand,
        "okf": {**okf_stats(s.bundle)},
    }


def okf_stats(b) -> dict:
    cs = list(b.concepts.values())
    return {"concepts": len(cs), "types": b.types(), "links": len(b.edges()),
            "verified": sum(c.is_verified() for c in cs), "stale": sum(c.is_stale() for c in cs),
            "drafts": sum(c.status == "draft" for c in cs)}


def okf_payload() -> dict:
    b = get_system().bundle
    back = b.backlinks()
    concepts = []
    for c in b.concepts.values():
        t = c.trust()
        concepts.append({**t, "description": c.description, "dir": c.path.rsplit("/", 1)[0] if "/" in c.path else "",
                         "links": c.links, "backlinks": back.get(c.path, []), "broken_links": c.broken_links})
    issues = [{"level": i.level, "path": i.path, "message": i.message} for i in validate(b)]
    return {"root": str(b.root), "stats": okf_stats(b), "concepts": sorted(concepts, key=lambda c: (c["dir"], c["title"])),
            "issues": issues, "index": b.index, "log": b.log, "indexes": b.indexes}


def concept_payload(path: str) -> dict:
    b = get_system().bundle
    c = b.concepts.get(path)
    if not c:
        raise KeyError(path)
    back = b.backlinks().get(path, [])
    ref = lambda p: {"path": p, "title": b.concepts[p].title, "type": b.concepts[p].type}  # noqa: E731
    return {**c.trust(), "description": c.description, "meta": c.meta, "body": c.body, "sections": c.sections(),
            "links": [{**ref(p), "text": c.link_text.get(p, "")} for p in c.links if p in b.concepts],
            "backlinks": [ref(p) for p in back if p in b.concepts], "broken_links": c.broken_links,
            "file": c.file.name}


class Handler(BaseHTTPRequestHandler):
    server_version = "MedRAG_OKF/0.1"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else ""):
            print("  ", args[0])

    # ---------------------------------------------------------------- helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                data = (STATIC / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/status":
                self._json(status_payload())
            elif path == "/api/okf":
                self._json(okf_payload())
            elif path == "/api/okf/concept":
                from urllib.parse import parse_qs, urlparse

                q = parse_qs(urlparse(self.path).query).get("path", [""])[0]
                self._json(concept_payload(q))
            elif path == "/api/eval":
                self._json({"results": load_results(), **EVAL_STATE})
            else:
                self._json({"error": "not found"}, 404)
        except KeyError as exc:
            self._json({"error": f"unknown concept {exc}"}, 404)
        except Exception as exc:  # noqa
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            body = self._body()
            s = get_system()
            if path == "/api/ask":
                q = (body.get("question") or "").strip()
                if not q:
                    return self._json({"error": "empty question"}, 400)
                k = int(body.get("top_k") or settings.top_k)
                use_llm = bool(body.get("use_llm", True))
                with _lock:
                    if body.get("mode") == "baseline":
                        res = s.baseline(q, k, use_llm)
                    else:
                        res = s.ask(q, k, use_llm, skip_clarification=bool(body.get("skip_clarification")), graph_expand=body.get("graph_expand"))
                return self._json(res.to_dict())
            if path == "/api/compare":
                q = (body.get("question") or "").strip()
                if not q:
                    return self._json({"error": "empty question"}, 400)
                k = int(body.get("top_k") or settings.top_k)
                use_llm = bool(body.get("use_llm", True))
                with _lock:
                    return self._json({"baseline": s.baseline(q, k, use_llm).to_dict(), "medrag_okf": s.ask(q, k, use_llm, graph_expand=body.get("graph_expand")).to_dict()})
            if path == "/api/upload":
                name = re.sub(r"[^A-Za-z0-9._-]", "_", body.get("filename") or "upload.txt")
                if not name.lower().endswith((".pdf", ".txt", ".md")):
                    return self._json({"error": "Only .pdf, .txt and .md files are supported"}, 400)
                settings.upload_dir.mkdir(parents=True, exist_ok=True)
                (settings.upload_dir / name).write_bytes(base64.b64decode(body["content_b64"]))
                with _lock:
                    s.rebuild()
                return self._json(status_payload())
            if path == "/api/okf/verify":
                by = (body.get("by") or "").strip()
                if not by:
                    return self._json({"error": "reviewer name is required"}, 400)
                c = s.bundle.concepts.get(body.get("path", ""))
                if not c or c.path.startswith(("uploads/", "corpus/")) or c.file.suffix != ".md":
                    return self._json({"error": "only bundle concept files can be verified"}, 400)
                with _lock:
                    verify_concept(c.file, by)
                    s.rebuild()
                return self._json(concept_payload(c.path))
            if path == "/api/delete_upload":
                name = Path(body.get("doc_id", "")).name
                for p in settings.upload_dir.glob(name + ".*"):
                    p.unlink()
                with _lock:
                    s.rebuild()
                return self._json(status_payload())
            if path == "/api/eval/run":
                if EVAL_STATE["running"]:
                    return self._json({"running": True})
                EVAL_STATE.update(running=True, error="")

                def job():
                    try:
                        with _lock:
                            res = run_evaluation(s, use_llm=bool(body.get("use_llm", False)))
                        save_results(res)
                    except Exception as exc:  # noqa
                        EVAL_STATE["error"] = str(exc)
                    finally:
                        EVAL_STATE["running"] = False

                threading.Thread(target=job, daemon=True).start()
                return self._json({"running": True})
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # noqa
            import traceback

            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    print("Loading knowledge base and models ...")
    s = get_system()
    print(f"  {len(s.index.chunks)} passages from {s.index.doc_count} documents")
    print(f"  retriever : {s.index.embedder.name}")
    print(f"  re-ranker : {s.index.reranker.name}")
    print(f"  generator : {'LLM ' + settings.llm_model if settings.llm_enabled() else 'offline extractive (no LLM key configured)'}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}"
    print(f"\nMedRAG_OKF is running at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
