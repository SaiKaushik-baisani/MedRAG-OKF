"""Run with:  python -m unittest discover -s tests -v
Covers retrieval, safety, clarification, evidence gate, verification and the LLM path (using a local mock LLM server)."""
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrag_okf.config import settings  # noqa: E402
from medrag_okf.pipeline import MedRAG_OKF  # noqa: E402
from medrag_okf.safety import assess_safety, needs_clarification  # noqa: E402

settings.embedding_backend = "tfidf"      # deterministic, no downloads in tests
settings.reranker_backend = "heuristic"
SYSTEM = MedRAG_OKF()


class MockLLM(BaseHTTPRequestHandler):
    reply = ""

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": MockLLM.reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class TestCore(unittest.TestCase):
    def test_retrieval_finds_right_document(self):
        hits = SYSTEM.index.hybrid_retrieve("What HbA1c level indicates diabetes?", 3)
        self.assertEqual(hits[0].chunk.doc_id, "conditions/type2_diabetes")

    def test_synonym_expansion_bridges_everyday_words(self):
        hits = SYSTEM.index.hybrid_retrieve("my bp is high, what should it be", 3)
        self.assertEqual(hits[0].chunk.doc_id, "conditions/hypertension")

    def test_emergency_short_circuits_before_retrieval(self):
        r = SYSTEM.ask("I have crushing chest pain spreading to my left arm")
        self.assertEqual(r.status, "emergency")
        self.assertEqual(r.sources, [])

    def test_self_harm_message_has_helpline(self):
        t = assess_safety("I want to end my life")
        self.assertEqual(t.level, "emergency")
        self.assertIn("14416", t.message)

    def test_dosing_flagged_as_caution(self):
        self.assertEqual(assess_safety("How many paracetamol tablets can I take?").level, "caution")

    def test_vague_query_triggers_clarification(self):
        self.assertTrue(needs_clarification("pain")[0])
        self.assertFalse(needs_clarification("What causes persistent fatigue?")[0])

    def test_out_of_scope_gets_insufficient_evidence(self):
        for q in ["What is the best treatment for kidney stones?", "What is the capital of France?", "Can turmeric cure cancer?"]:
            self.assertEqual(SYSTEM.ask(q).status, "insufficient_evidence", q)

    def test_baseline_answers_out_of_scope_anyway(self):
        self.assertEqual(SYSTEM.baseline("What is the best treatment for kidney stones?").status, "answered")

    def test_answer_has_sources_and_scores(self):
        r = SYSTEM.ask("What causes persistent fatigue?")
        self.assertIn(r.status, ("answered", "answered_with_flags"))
        self.assertTrue(r.sources and r.evidence_score >= settings.min_evidence_score)
        self.assertEqual(r.supported_rate, 1.0)


class TestOKF(unittest.TestCase):
    """Open Knowledge Format: parsing, links, conformance, trust signals, graph, verification."""

    def setUp(self):
        import shutil
        import tempfile

        from medrag_okf import okf

        self.okf = okf
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "bundle"
        shutil.copytree(settings.okf_dir, self.root)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shipped_bundle_is_conformant(self):
        b = self.okf.load_bundle(settings.okf_dir)
        self.assertGreaterEqual(len(b.concepts), 15)
        self.assertEqual([i for i in self.okf.validate(b) if i.level == "error"], [])
        self.assertTrue(all(c.type for c in b.concepts.values()))

    def test_identity_is_path_and_links_resolve(self):
        b = self.okf.load_bundle(settings.okf_dir)
        asthma = b.concepts["conditions/asthma"]
        self.assertIn("emergency/emergency_warning_signs", asthma.links)
        self.assertEqual(asthma.broken_links, [])
        self.assertIn("conditions/asthma", b.backlinks()["emergency/emergency_warning_signs"])

    def test_relative_and_absolute_links(self):
        r = self.okf.resolve_link
        self.assertEqual(r("conditions/asthma", "/emergency/x.md"), "emergency/x")
        self.assertEqual(r("conditions/asthma", "../emergency/x.md"), "emergency/x")
        self.assertEqual(r("conditions/asthma", "dehydration.md"), "conditions/dehydration")

    def test_unknown_frontmatter_keys_are_preserved_not_rejected(self):
        c = self.root / "conditions" / "custom.md"
        c.write_text("---\ntype: Condition\ntitle: Custom\nfuture_field: {a: 1}\nweird: hello\n---\n\n## Overview\nSome text here that is long enough to chunk.\n")
        b = self.okf.load_bundle(self.root)
        self.assertEqual(b.concepts["conditions/custom"].meta["weird"], "hello")
        self.assertEqual([i for i in self.okf.validate(b) if i.level == "error"], [])

    def test_missing_type_is_the_one_spec_error(self):
        (self.root / "conditions" / "bad.md").write_text("---\ntitle: No type\n---\n\nbody\n")
        b = self.okf.load_bundle(self.root)
        errs = [i for i in self.okf.validate(b) if i.level == "error"]
        self.assertEqual([(e.path, e.message) for e in errs], [("conditions/bad", "missing required `type` field")])

    def test_broken_link_and_stale_are_flagged(self):
        f = self.root / "conditions" / "asthma.md"
        txt = f.read_text().replace("stale_after: 2027-10-07T00:00:00Z", "stale_after: 2020-01-01T00:00:00Z")
        f.write_text(txt + "\nSee [ghost](/conditions/ghost.md).\n")
        b = self.okf.load_bundle(self.root)
        msgs = [i.message for i in self.okf.validate(b)]
        self.assertTrue(any("ghost" in m for m in msgs))
        self.assertTrue(any("stale_after" in m for m in msgs))
        self.assertTrue(b.concepts["conditions/asthma"].is_stale())

    def test_only_human_verification_counts(self):
        b = self.okf.load_bundle(self.root)
        c = b.concepts["conditions/asthma"]
        self.assertFalse(c.is_verified())
        self.okf.verify_concept(c.file, "Dr Rao")
        self.okf.verify_concept(c.file, "Dr Mehta")
        c2 = self.okf.load_bundle(self.root).concepts["conditions/asthma"]
        self.assertTrue(c2.is_verified())
        self.assertEqual(c2.status, "stable")
        self.assertEqual([v["by"] for v in c2.verified], ["human:Dr Rao", "human:Dr Mehta"])   # history kept
        c2.meta["verified"] = [{"by": "agent:gemini"}]
        self.assertFalse(c2.is_verified())          # an agent stamp is not human verification

    def test_answer_carries_okf_provenance_and_related(self):
        r = SYSTEM.ask("What causes persistent fatigue?")
        self.assertTrue(r.provenance["total"] >= 1)
        self.assertTrue(any("verification" in n for n in r.provenance["notes"]))
        self.assertTrue(r.related)                                   # graph neighbours suggested
        self.assertTrue(all(s["concept_type"] and s["doc_id"] in SYSTEM.bundle.concepts for s in r.sources))

    def test_stale_source_is_flagged_in_answer(self):
        from medrag_okf.pipeline import okf_provenance

        b = self.okf.load_bundle(self.root)
        f = b.concepts["conditions/asthma"].file
        f.write_text(f.read_text().replace("stale_after: 2027-10-07T00:00:00Z", "stale_after: 2020-01-01T00:00:00Z"))
        b = self.okf.load_bundle(self.root)
        hits = SYSTEM.index.hybrid_retrieve("When should I get emergency help during an asthma attack?", 3)
        prov, _ = okf_provenance(b, hits)
        self.assertTrue(any("stale_after" in n for n in prov["notes"]))

    def test_graph_expansion_adds_linked_concepts_only_via_links(self):
        q = "What causes persistent fatigue?"
        plain = SYSTEM.index.hybrid_retrieve(q, 8, graph_expand=False)
        graph = SYSTEM.index.hybrid_retrieve(q, 8, graph_expand=True)
        self.assertGreaterEqual(len(graph), len(plain))
        reached = [h for h in graph if h.ranks.get("graph") and len(h.ranks) == 1]
        for h in reached:
            self.assertIn(h.chunk.doc_id, SYSTEM.index.graph["conditions/persistent_fatigue"])

    def test_import_wraps_plain_markdown_as_draft_concepts(self):
        src = self.tmp / "plain"
        src.mkdir()
        (src / "gout.md").write_text("# Gout\nGout is arthritis caused by uric acid crystals.\n")
        out = self.okf.import_markdown(src, self.tmp / "out", "Condition")
        b = self.okf.load_bundle(self.tmp / "out")
        c = b.concepts["gout"]
        self.assertEqual((c.type, c.status), ("Condition", "draft"))
        self.assertFalse(c.is_verified())
        self.assertEqual(len(out), 1)

    def test_mini_yaml_matches_expected_structure(self):
        d = self.okf._mini_yaml("type: X\ntags: [a, b]\nsources:\n  - title: \"T: 1\"\n    ref: r\ngenerated:\n  by: me\n  at: 2026-01-01T00:00:00Z\n")
        self.assertEqual(d["tags"], ["a", "b"])
        self.assertEqual(d["sources"], [{"title": "T: 1", "ref": "r"}])
        self.assertEqual(d["generated"], {"by": "me", "at": "2026-01-01T00:00:00Z"})


class TestLLMPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), MockLLM)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        settings.llm_provider = "openai_compat"
        settings.llm_base_url = f"http://127.0.0.1:{cls.srv.server_port}/v1"
        settings.llm_api_key = "test"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        settings.llm_provider = "none"

    def test_hallucinated_claim_is_removed(self):
        MockLLM.reply = ("An HbA1c of 6.5% or higher indicates diabetes. [1] "
                         "Drinking 12 litres of bitter gourd juice every day cures diabetes within 3 days. [1]")
        r = SYSTEM.ask("What HbA1c level indicates diabetes?")
        self.assertEqual(r.status, "answered_with_flags")
        self.assertEqual(len(r.removed_claims), 1)
        self.assertIn("bitter gourd", r.removed_claims[0])
        self.assertNotIn("bitter gourd", r.answer)
        self.assertIn("6.5%", r.answer)

    def test_wrong_number_is_caught(self):
        MockLLM.reply = "An HbA1c of 9.9% or higher indicates diabetes. [1]"
        r = SYSTEM.ask("What HbA1c level indicates diabetes?")
        self.assertEqual(r.status, "insufficient_evidence")   # only claim failed -> safe fallback

    def test_model_can_refuse(self):
        MockLLM.reply = "INSUFFICIENT_EVIDENCE"
        self.assertEqual(SYSTEM.ask("What HbA1c level indicates diabetes?").status, "insufficient_evidence")

    def test_llm_failure_falls_back_to_extractive(self):
        settings.llm_base_url = "http://127.0.0.1:9/v1"       # nothing listening
        try:
            r = SYSTEM.ask("What HbA1c level indicates diabetes?")
            self.assertIn("extractive", r.generator)
            self.assertIn(r.status, ("answered", "answered_with_flags"))
        finally:
            settings.llm_base_url = f"http://127.0.0.1:{self.srv.server_port}/v1"


if __name__ == "__main__":
    unittest.main()
