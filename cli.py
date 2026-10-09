"""Command-line helpers.

    python cli.py ask "What causes persistent fatigue?"
    python cli.py eval            # baseline vs MedRAG_OKF table (offline extractive generator)
    python cli.py eval --llm      # same, but using the configured LLM
    python cli.py warmup          # download/cached models once, BEFORE the demo (needs internet)

    # Open Knowledge Format (OKF) bundle tools
    python cli.py okf-validate                       # conformance + trust checks on data/okf
    python cli.py okf-graph                          # concepts, types and links
    python cli.py okf-verify conditions/asthma --by "Dr. Rao"   # record a HUMAN review (verified + status: stable)
    python cli.py okf-import some_folder --to data/okf/imported  # wrap plain .md/.txt files as draft OKF concepts
"""
import argparse
import json

from pathlib import Path

from medrag_okf.config import settings
from medrag_okf.evaluation import format_table, run_evaluation, save_results
from medrag_okf.okf import import_markdown, load_bundle, validate, verify_concept
from medrag_okf.pipeline import MedRAG_OKF


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--baseline", action="store_true")
    a.add_argument("--json", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--llm", action="store_true")
    sub.add_parser("warmup")
    v = sub.add_parser("okf-validate")
    v.add_argument("--dir", default=str(settings.okf_dir))
    g = sub.add_parser("okf-graph")
    g.add_argument("--dir", default=str(settings.okf_dir))
    vf = sub.add_parser("okf-verify")
    vf.add_argument("path", help="concept path, e.g. conditions/asthma")
    vf.add_argument("--by", required=True, help="reviewer name (stored as human:<name>)")
    vf.add_argument("--dir", default=str(settings.okf_dir))
    im = sub.add_parser("okf-import")
    im.add_argument("src")
    im.add_argument("--to", default=str(settings.okf_dir / "imported"))
    im.add_argument("--type", default="Document")
    args = ap.parse_args()

    if args.cmd.startswith("okf-"):
        return okf_command(args)

    system = MedRAG_OKF()
    if args.cmd == "warmup":
        print("Retriever :", system.index.embedder.name)
        print("Re-ranker :", system.index.reranker.name)
        print("Models are ready (cached for offline use).")
    elif args.cmd == "ask":
        r = system.baseline(args.question) if args.baseline else system.ask(args.question)
        if args.json:
            print(json.dumps(r.to_dict(), indent=2))
            return
        print(f"[{r.status}]  evidence={r.evidence_score}  supported={r.supported_rate}  {r.latency_ms:.0f} ms\n")
        print(r.answer)
        for s in r.sources:
            print(f"\n  [{s['rank']}] {s['title']} - {s['section']}  (score {s['score']})")
    elif args.cmd == "eval":
        res = run_evaluation(system, use_llm=args.llm)
        save_results(res)
        print(f"retriever: {res['retriever']} | re-ranker: {res['reranker']} | generator: {res['generator']}\n")
        print(format_table(res))
        print(f"\nSaved to {settings.eval_results_file}")


def okf_command(args):
    if args.cmd == "okf-import":
        files = import_markdown(Path(args.src), Path(args.to), args.type)
        print(f"Wrote {len(files)} draft OKF concepts to {args.to}")
        for f in files:
            print("  ", f)
        print("Review them, then run: python cli.py okf-validate")
        return
    bundle = load_bundle(Path(args.dir))
    if args.cmd == "okf-validate":
        issues = validate(bundle)
        errs = [i for i in issues if i.level == "error"]
        for i in issues:
            print(f"{i.level.upper():8} {i.path or '-':45} {i.message}")
        cs = bundle.concepts.values()
        print(f"\n{len(bundle.concepts)} concepts, {len(bundle.edges())} links | "
              f"{sum(c.is_verified() for c in cs)} human-verified, {sum(c.status == 'draft' for c in cs)} draft, "
              f"{sum(c.is_stale() for c in cs)} stale | {len(errs)} errors, {len(issues) - len(errs)} warnings")
        print("CONFORMANT (every concept has a `type`)" if not errs else "NOT CONFORMANT")
        raise SystemExit(1 if errs else 0)
    if args.cmd == "okf-graph":
        print("types:", bundle.types())
        back = bundle.backlinks()
        for c in sorted(bundle.concepts.values(), key=lambda c: c.path):
            print(f"{c.path:45} [{c.type}] -> {len(c.links)} out, {len(back.get(c.path, []))} in")
            for t in c.links:
                print(f"      -> {t}")
    if args.cmd == "okf-verify":
        path = args.path[:-3] if args.path.endswith(".md") else args.path
        c = bundle.concepts.get(path)
        if not c:
            raise SystemExit(f"unknown concept {path!r}; known: {', '.join(sorted(bundle.concepts))}")
        entry = verify_concept(c.file, args.by)
        print(f"{path}: verified by {entry['by']} at {entry['at']} (status -> stable)")


if __name__ == "__main__":
    main()
