# 5-minute demo script for tomorrow's review

**Before the review (tonight):**
1. `python app.py` once and click through everything. Check the sidebar chips.
2. Optional but better: add an LLM key in `.env` (see `.env.example`) and `pip install sentence-transformers` + `python cli.py warmup` while you have internet.
3. Open the app and keep it running. Don't depend on Wi-Fi in the room (offline mode works).

**Opening line (slide 13):** "My novelty is not RAG. My novelty is making the RAG pipeline trust-aware."

## Walkthrough

| Step | Do this | Say this |
|---|---|---|
| 1. Normal answer | Chat tab → *What causes persistent fatigue?* → click a `[1]` citation, open **Sources** and **Claim-by-claim verification** | "Answer, source document and section, relevance score, and every claim checked against the evidence." |
| 2. Pipeline trace | Open **Pipeline trace** | "Safety → query analysis → hybrid retrieval and re-rank → evidence score → generation → verification. This is slide 6 running live." |
| 3. Baseline vs MedRAG_OKF | **Baseline vs MedRAG_OKF** tab → *What is the best treatment for kidney stones?* | "My knowledge base has no kidney-stone document. Baseline RAG still answers from irrelevant text. MedRAG_OKF sees weak evidence and refuses instead of guessing." (slide 8) |
| 4. Clarification | Chat → type `pain`, then reply `in my head, worse with bright light` | "Ambiguous queries get focused follow-up questions before retrieval." |
| 5. Safety | Chat → *I have crushing chest pain spreading to my left arm* | "High-risk patterns skip the LLM entirely and escalate." Then *How many paracetamol tablets can I take at once?* → caution banner (no personal dosing). |
| 6. Live knowledge update | Sidebar → upload a short kidney-stone `.txt`, ask the kidney-stone question again | "Add a document and the same question is now answered with a citation. This addresses knowledge drift." |
| 6b. OKF knowledge base | **Knowledge (OKF)** tab → click *Asthma* | "Our knowledge base is a Google Open Knowledge Format bundle: one markdown file per concept, the path is its identity, links form a graph. Each concept carries its own trust signals: who generated it, whether a human verified it, and when it goes stale." Point at *Verified by: nobody yet*. |
| 6c. Trust travels with answers | Chat → *What causes persistent fatigue?* → open **Sources** | "Every cited source shows its OKF type and status. These are AI-drafted and unverified, and the answer says so instead of hiding it. Related concepts come from the OKF links." |
| 6d. Human verification | Knowledge tab → type a real reviewer name → *Record human verification* (only if someone actually reviewed it) | "Verification is recorded in the file as `verified: human:<name>`; the badge and the Knowledge-verified tile update." |
| 7. Evaluation | **Evaluation** tab → Run evaluation | "Baseline vs MedRAG_OKF on retrieval, safety, and latency. Overhead is a few milliseconds." Say "preliminary, small test set". |

## Questions the panel may ask

- **"Isn't this just RAG?"** No: RAG is the base. The contribution is the trust layer (evidence gate, claim verification, source visibility, safety triage) plus a measured comparison with baseline.
- **"How do you verify claims?"** Each generated sentence must find most of its topical terms in one retrieved passage and every number must appear there. An LLM-as-judge option exists (`VERIFIER=llm`). Planned: NLI model / RAGAS faithfulness.
- **"Why is grounding 1.00 for both?"** In offline mode the generator is extractive, so it can only quote evidence. The grounding gap appears with an LLM, which is why the dashboard has a "use LLM generator" switch.
- **"Is the corpus trusted?"** Currently 15 demo summaries; the plan is to ingest WHO/NHS/CDC/ICMR documents. Uploading a PDF already works.
- **"Is the test set big enough?"** No. It's a development set (36 questions). The final evaluation will use a larger expert-reviewed set and RAGAS.
- **"Why OKF?"** It is a vendor-neutral, markdown-based format for curated knowledge with built-in provenance and freshness fields. It lets the trust layer see where each passage came from, whether a human verified it, and whether it is stale, and the knowledge stays human-readable and version-controllable. OKF is a format, not a retrieval method; retrieval is still our hybrid BM25 + dense pipeline over passages from the bundle.
- **"Did OKF improve accuracy?"** Not measured as such: it improves transparency and maintainability. Graph-expanded retrieval over OKF links showed no gain on our test set, so it is off by default.
- **"Are the concepts verified?"** No. They are AI-drafted prototype summaries marked `draft`; a human review step is built in (`okf-verify`) but has not been done.
- **"What's next?"** FAISS/Chroma, real WHO/NHS/ICMR sources with expert verification, multilingual input, RAGAS.
