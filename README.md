# MedRAG_OKF - A Trust-Aware Medical RAG Chatbot

Retrieve → Rank → Generate → **Verify → Explain → Safety-check**

Implements the pipeline from the review slides: hybrid retrieval + re-ranking, evidence verification,
confidence/evidence score, source attribution, query clarification, a safety/triage layer, and an
evaluation dashboard comparing **Baseline RAG vs MedRAG_OKF**.

The knowledge base is a **Google [Open Knowledge Format (OKF)](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing) bundle**: one markdown concept per file with YAML
frontmatter (type, tags, sources, generated, verified, status, stale_after) and ordinary links between concepts.
Those trust signals travel with every cited passage (see *OKF integration* below).

## Run it (2 minutes)

```bash
# Windows: double-click run.bat      macOS/Linux: ./run.sh
# or manually:
pip install -r requirements.txt
python app.py            # opens http://localhost:8000
```

It works **with no API key and no internet** (offline extractive generator + TF-IDF retriever).
For the best demo, add an LLM and (optionally) real embeddings:

1. `copy .env.example .env` and uncomment one LLM option (OpenRouter, Gemini, Groq, Ollama or Anthropic).
2. Optional: `pip install sentence-transformers`, then run `python cli.py warmup` **once while online** so the
   embedding model and cross-encoder are cached. After that everything also runs offline.

The sidebar shows which retriever / re-ranker / generator is actually active.

## What is in the UI

| Tab | Purpose |
|---|---|
| **Chat** | Ask questions. Each answer has clickable `[n]` citations, source passages (query terms highlighted), an evidence score, claim-by-claim verification and a per-stage pipeline trace. |
| **Baseline vs MedRAG_OKF** | Same question through both pipelines side by side (slide 8's example, live). |
| **Knowledge (OKF)** | The OKF bundle as an interactive concept graph: click a concept to see its frontmatter, trust signals, links and backlinks; validation results; bundle tree; `log.md`. Concepts used in your last answer are ringed. |
| **Evaluation** | Runs the built-in test set and shows the Baseline-vs-MedRAG_OKF table (slide 10's metrics). |
| **Architecture** | One-page explanation of modules A-G from the slides. |

The sidebar also lets you toggle MedRAG_OKF/Baseline, change top-k, and **upload your own PDF/TXT/MD documents** (re-indexed instantly).

## Code map (matches the slides)

| Slide module | File |
|---|---|
| Open Knowledge Format: parser, bundle loader, validator, verifier, importer | `medrag_okf/okf.py` |
| Corpus → cleaning → chunking (one passage per section) → metadata | `medrag_okf/corpus.py` |
| A. Hybrid retrieval (BM25 + dense, RRF) + re-ranking | `medrag_okf/retrieval.py` |
| D. Query clarification, E. Safety / triage | `medrag_okf/safety.py` |
| LLM generation (OpenAI-compatible / Anthropic / offline extractive) | `medrag_okf/generation.py` |
| B. Evidence verification (claim-level, numeric check, optional LLM judge) | `medrag_okf/verification.py` |
| C. Explainable output, evidence gate, full pipeline | `medrag_okf/pipeline.py` |
| Evaluation (retrieval, grounding, answer quality, safety, latency) | `medrag_okf/evaluation.py`, `data/eval/test_set.json` |
| Web server + UI | `app.py`, `static/index.html` |

Every component is replaceable (embedder, re-ranker, vector store, LLM), as the "keep modules replaceable" design principle says.

## How the trust layer works

1. **Safety / triage** - regex rules for red flags (chest pain, stroke signs, breathing difficulty, self-harm, overdose, seizure, heavy bleeding, infant fever) short-circuit to emergency guidance (112 / Tele-MANAS 14416). Dosing, diagnosis and "stop my medicine" requests get a caution banner.
2. **Clarification** - vague input ("pain", "I feel sick") returns focused follow-up questions before any retrieval.
3. **Hybrid retrieval** - BM25 (with clinical synonym expansion: tired→fatigue, bp→blood pressure...) + dense similarity, fused with Reciprocal Rank Fusion, then re-ranked.
4. **Evidence score / gate** - 0-1 score from IDF-weighted query-term coverage of the top passages and the best re-rank score, reduced when key terms don't exist in the knowledge base at all. Below the threshold the bot says *insufficient evidence* instead of guessing.
5. **Generation** - prompt restricts the LLM to the numbered passages, requires citations, forbids diagnosis/dosing, and allows `INSUFFICIENT_EVIDENCE`.
6. **Claim verification** - each answer sentence must have most of its topical terms in one retrieved passage, and any number in the claim (e.g. `6.5`, `140/90`) must appear there. Unsupported sentences are removed ("safer path"); if too many fail the whole answer is replaced by the safe fallback.

## OKF integration

What was implemented, mapped to the spec's concepts:

| OKF idea | In MedRAG_OKF |
|---|---|
| Bundle = directory of markdown files, one concept per file, **path is identity** | `data/okf/` (15 concepts in `conditions/`, `infections/`, `emergency/`). A concept's path (e.g. `conditions/asthma`) is also the id of its retrieval passages and is shown on every source. |
| Only `type` is required; unknown keys must be tolerated | Validator reports a missing `type` as the only spec error; every other frontmatter key is preserved (`Concept.meta`). |
| Section headings as chunking boundaries | `corpus.py` makes one passage per `## Section`; the `Related concepts` section is navigation, not evidence, so it is not indexed. |
| Ordinary markdown links form a graph | Links are resolved (absolute `/a/b.md` or relative) into a concept graph with backlinks. Shown in the Knowledge tab; used for **related-concept suggestions** under each answer and an optional **graph-expanded retrieval** toggle. |
| `index.md` and `log.md` | Read and displayed in the Knowledge tab. |
| v0.2 trust fields: `sources`, `generated`, `verified`, `status`, `stale_after`, `not` | Shown as badges on every cited source (`draft · unverified`, `human-verified`, `stale`), summarised in a **Knowledge verified n/m** tile and a trust note on each answer. `verified` only counts entries stamped `human:<name>` (an `agent:` stamp does not count). |
| Producer / consumer independence | Consumer: everything above. Producer helpers: `okf-import` wraps plain `.md`/`.txt` as draft concepts; `okf-verify` records a human review. |

Honest status of the knowledge: all 15 concepts are `status: draft` and **unverified**: they are AI-written prototype
summaries, and the UI says so on every answer. Nothing is marked verified until a real person reviews it:

```bash
python cli.py okf-validate                       # spec conformance + trust checks
python cli.py okf-graph                          # types and links
python cli.py okf-verify conditions/asthma --by "Dr. Rao"   # writes verified + status: stable (history is kept)
python cli.py okf-import my_plain_docs --to data/okf/imported  # wrap plain files as draft concepts
```
(The Knowledge tab has the same "Record human verification" button.) To demo staleness, set a concept's `stale_after`
to a past date and reload: its sources get a red `stale` badge and the answer carries an "may be outdated" note.

Notes: (1) the OKF spec text used here is Google's June 2026 announcement plus public write-ups of v0.2; the v0.2 trust
fields are implemented as plain extra frontmatter keys, so any v0.1 consumer ignores them. (2) Graph-expanded retrieval is
**off by default**: on the 36-question test set it gave identical precision/recall/MRR to normal hybrid retrieval, so I
am not claiming it improves accuracy; it is there as an experiment (sidebar toggle, or `GRAPH_EXPAND=1`).

## Evaluation (offline mode, included in `data/eval/results.json`)

36-question custom test set: 20 answerable, 5 red-flag, 2 dosing/diagnosis, 6 out-of-scope, 3 ambiguous.

| Metric | Baseline RAG | MedRAG_OKF |
|---|---|---|
| Precision@4 | 0.84 | 0.90 |
| MRR | 0.93 | 1.00 |
| Out-of-scope questions answered anyway (lower is better) | 100% | 0% |
| Red-flag queries escalated | 0% | 100% |
| Ambiguous queries clarified | 0% | 100% |
| Mean latency | ~1 ms | ~2 ms |

**Read these numbers honestly.** (1) The test set is small and was used while tuning thresholds, so results are optimistic; say "preliminary" in the review. (2) In offline mode the generator is extractive, so supported-claim rate is 1.00 for both systems by construction; grounding differences only show up with a real LLM (run the Evaluation tab with *use LLM generator* on). (3) Grounding is scored by the same verifier MedRAG_OKF uses, so it is self-assessed; RAGAS and a larger expert-reviewed test set are the planned final evaluation.

## The knowledge base

`data/okf/` holds 15 short **demo concepts** of standard patient-education content (fatigue, diabetes, hypertension, anemia, asthma, cold/flu, migraine, dehydration, UTI, vitamin D, thyroid, reflux, dengue, sleep, emergency signs). They were AI-drafted for this prototype and are **not** copies of WHO/NHS/CDC documents (each concept's `sources` says so). Before you describe the corpus as "trusted", replace or extend it with real documents and have a reviewer verify them. You can drop PDFs/TXT/MD into `data/corpus/`, upload them in the UI (they become draft `Document` concepts), or write proper OKF concepts under `data/okf/<folder>/<name>.md`.

## Commands

```bash
python app.py --port 8501        # change port
python cli.py ask "What causes persistent fatigue?"
python cli.py ask "kidney stones treatment" --baseline
python cli.py eval               # Baseline vs MedRAG_OKF table (add --llm to use the LLM)
python cli.py okf-validate       # OKF conformance + trust checks (see OKF integration)
python -m unittest discover -s tests -v
```

## Next review (not implemented yet)

FAISS/ChromaDB as the vector store (currently an in-memory cosine index, behind the `Index` class), RAGAS evaluation, larger expert-reviewed test set, multilingual queries.
