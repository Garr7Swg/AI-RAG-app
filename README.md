# Northwind RAG — Production-Grade RAG with an Evaluation Gate

A retrieval-augmented generation system built the way it has to be built when people actually
rely on the answers: **hybrid retrieval, citation-verified generation, a hard "I don't know"
guardrail, and an automated evaluation pipeline that fails the build when quality regresses.**

It runs entirely on free tiers: local embeddings and re-ranking, ChromaDB (or Pinecone free),
Ollama for generation, Ragas for scoring, Streamlit Community Cloud for the UI, GitHub Actions
for evaluation.

```
                 ┌─────────────────────────────────────────────────────────────┐
  PDF · DOCX ·   │  INGEST                                                     │
  MD · TXT ·  ──▶│  validate → extract pages → semantic chunk → embed          │
  HTML (≤10 MB)  │            │                             │                  │
                 │            ▼                             ▼                  │
                 │      chunks.jsonl                  Chroma / Pinecone        │
                 │            │                             │                  │
                 └────────────┼─────────────────────────────┼──────────────────┘
                              ▼                             ▼
   question ──▶ rewrite ──▶ BM25 ──┐                   dense top-20
                                   ├──▶ RRF fuse ──▶ cross-encoder re-rank ──┐
                                   ┘                                         │
                                                                             ▼
                        cite-or-abstain ◀── verify ◀── generate (fenced context)
                              │                │
                              │                ├─ numeric grounding
                              │                └─ citation validation
                              ▼
                    answer + citations   |   "I don't know."
                              │
                              ▼
                 Ragas eval → report → gate → commit to git
```

---

## 1. What problem this solves

Basic RAG is a weekend project. What separates a demo from something a company can deploy:

| Demo | This system |
| --- | --- |
| Semantic search only | Dense **+ BM25** fused with Reciprocal Rank Fusion |
| No re-ranking | Cross-encoder re-ranking with a calibrated score floor |
| Model answers from memory | Context is fenced, cited, and **verified against evidence** |
| Hallucinated numbers | Every number in the answer must appear in the cited text, or it abstains |
| "Seems good" | 20-question golden set, Ragas metrics, regression gate in CI |
| Secrets in code | Supabase JWT verified server-side, uploads validated, no tokens in URLs |
| Silent regressions | Reports committed to git; a metric drop fails the pipeline |

The evaluation gate is the point: **quality can go up, it cannot quietly go down.**

---

## 2. Stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python **3.12 / 3.13** | Both supported; CI tests both |
| Orchestration | LangChain core (`ChatOpenAI`, text splitters, embeddings interface) | Needed for Ragas' judge and provider abstraction |
| Vector DB | **ChromaDB** (default) or **Pinecone** free tier | One env var switches; same interface |
| Keyword search | `rank-bm25` | Exact identifiers, codes, dollar amounts |
| Embeddings | `BAAI/bge-small-en-v1.5` (33M, 384-dim) | Fits a 6 GB GPU; runs fine on CPU |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M) | Biggest quality-per-millisecond gain |
| Generation | Ollama `qwen2.5:3b-instruct` (default) or any OpenAI-compatible endpoint | Free, ~2 GB quantised, strong at grounded extraction |
| Evaluation | **Ragas** + deterministic fallback | Ragas for real scoring; fallback keeps CI honest with no API key |
| UI | **Streamlit** | Free Community Cloud hosting |
| Auth | **Supabase** (email/password), JWT verified locally | Free tier, RLS for data |
| CI | GitHub Actions | Runs eval, commits reports back |

---

## 3. Repository layout

```
.
├── app/
│   ├── streamlit_app.py        # the whole UI (auth, chat, library, evaluation, system)
│   └── theme.py                # design system: CSS, splash animation, metric cards
├── src/ragkit/
│   ├── config.py               # env-driven settings (frozen dataclass)
│   ├── loaders.py              # PDF/DOCX/MD/TXT/HTML → pages
│   ├── chunking.py             # semantic chunking (sentence-distance boundaries)
│   ├── embeddings.py           # bge-small wrapper + offline test embedder
│   ├── bm25.py                 # keyword index, persisted
│   ├── vectorstore.py          # Chroma + Pinecone behind one interface
│   ├── rerank.py               # cross-encoder + lexical fallback
│   ├── hybrid.py               # RRF fusion + context builder
│   ├── llm.py                  # Ollama / OpenAI-compatible adapters
│   ├── prompts.py              # grounded, injection-resistant prompt
│   ├── rag.py                  # the pipeline: retrieve → generate → verify → cite
│   ├── ingest.py               # idempotent ingestion + manifest
│   ├── security.py             # upload validation, injection scan, rate limiting
│   ├── auth.py                 # Supabase auth + JWKS/HS256 token verification
│   └── eval/                   # dataset, runner, metrics, report, gate
├── scripts/                    # ingest.py, evaluate.py, check_env.py, bootstrap.sh
├── data/corpus/                # 3-document demo corpus (annual report, credit agreement, manual)
├── data/eval/golden_set.jsonl  # 20 questions (16 answerable + 4 controls)
├── eval/                       # baseline + generated reports (committed by CI)
├── .github/workflows/          # ci.yml (tests) + rag-eval.yml (eval + publish)
├── docs/                       # architecture, evaluation, security, Supabase, runbook
└── tests/                      # 52 offline unit tests
```

---

## 4. Quickstart in GitHub Codespaces (step by step)

### Step 1 — Open the repo in a Codespace

Push this repository to GitHub, click **Code → Codespaces → Create codespace on main**.
The devcontainer (`.devcontainer/devcontainer.json`, Python 3.12) installs dependencies
automatically via `scripts/bootstrap.sh`. That takes 3-6 minutes the first time.

If you prefer a plain terminal:

```bash
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

### Step 2 — Check what works

```bash
python scripts/check_env.py
```

You should see `ok` for streamlit, chromadb, sentence-transformers, rank-bm25, pypdf,
python-docx, ragas, supabase and PyJWT. The LLM line will say `warn` until you do step 3.

### Step 3 — Get a model (choose one)

**Option A — Groq (free tier, recommended).** No GPU, no local model, and the only
option that works when the UI is hosted (Cloud, Codespaces, Streamlit Community Cloud).

```bash
# .env
RAG_LLM_PROVIDER=openai_compat
RAG_OPENAI_BASE_URL=https://api.groq.com/openai/v1
RAG_OPENAI_MODEL=openai/gpt-oss-20b
RAG_OPENAI_API_KEY=your_key_here     # free key: https://console.groq.com/keys
```

> Groq retired `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` for free and
> developer tiers on 2026-08-16. Use `openai/gpt-oss-20b` (fast) or
> `openai/gpt-oss-120b` (stronger answers, still free-tier). Free-plan limits are
> roughly 30 requests/min, 1,000/day and 8,000 tokens/min.

Do not commit `.env` — it is gitignored.

**Option B — Ollama (free, local, no API key).** Best when you want zero network calls
and have the GPU: a 3B Q4 model fits an RTX 2060 6 GB.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b-instruct      # ~2 GB
```

`RAG_LLM_PROVIDER` unset means **Ollama** — local use stays exactly as it was, and Groq is
never contacted unless you ask for it. Set `RAG_LLM_PROVIDER=auto` if you would rather have
Groq picked automatically whenever a key is present.

### Step 4 — Ingest the corpus

```bash
make ingest
# or: python scripts/ingest.py --rebuild -v
```

Expected output: `{"documents": 3, "pages": 3, "chunks": ~30, ...}`. The first run downloads
the embedding model (~130 MB) and the re-ranker (~90 MB) into `~/.cache/huggingface`.

### Step 5 — Run the app

```bash
make serve
```

Open the forwarded port 8501. You will see the loading animation for the first few seconds
(models + index warm-up), then the console.

Citations appear under every answer. Expand **Evidence & scores** to see the chunk id,
page number and re-ranker score. Ask something the documents do not cover — for example
*"What was Q3 FY2025 revenue?"* — and the answer will be exactly **"I don't know."**

### Step 6 — Run the evaluation

```bash
make eval-offline   # deterministic judge, no model needed, seconds
make eval           # Ragas judge (needs Ollama or an API key), slower
```

`make eval` writes:

```
eval/reports/latest.json      full report (metrics, records, config, git sha)
eval/reports/latest.md        human-readable table
eval/reports/summary.json     compact metrics for badges/dashboards
eval/reports/history/*.jsonl  one line per question, per run
```

Exit code is `0` when the quality gate passes and `1` when it fails.

Reference runs on the bundled corpus (CPU, Python 3.13, 40 chunks, 20 questions):

| Judge | `context_recall` | `hit_rate` | `faithfulness` | `answer_relevancy` | `abstention_accuracy` |
| --- | --- | --- | --- | --- | --- |
| Ragas + `qwen2.5:3b-instruct` (gate **PASS**) | 1.000 | 1.000 | 0.733 | 0.912 | 0.950 |
| deterministic extractive (no model) | 0.943 | 1.000 | n/a | n/a | 0.900 |

These are two different judges, so do not compare them metric-by-metric. Full per-question
breakdown and caveats: [`docs/EVALUATION.md`](docs/EVALUATION.md#8-reference-runs).

### Step 7 — Run the tests

```bash
make test
```

52 tests, no network, no model downloads, ~4 seconds.

---

## 5. Using your own documents

**In the UI:** *Library → Upload documents*. Files are validated (size, extension, magic
bytes, zip-bomb check), parsed, semantically chunked and indexed immediately.

**From the CLI:**

```bash
python scripts/ingest.py --file ~/reports/10-K-2024.pdf
python scripts/ingest.py --path ~/manuals --rebuild
python scripts/ingest.py --stats
```

Limits: **10 MB per file** (configurable), PDF / DOCX / MD / TXT / HTML. Scanned PDFs
without a text layer are rejected with `no_text` — run OCR first.

Re-ingesting the same file replaces its vectors (content-addressed `doc_id`), so nothing is
duplicated.

---

## 6. Authentication with your Supabase project

1. Create a project at [supabase.com](https://supabase.com) (free tier).
2. **Authentication → Providers → Email**: enable it. For a demo, turn *Confirm email* off;
   for anything real, leave it on.
3. **Authentication → URL Configuration**: *Site URL* = your app's public origin
   (`https://<your-app>.streamlit.app` once deployed, `http://localhost:8501` otherwise), and
   add every origin you use to *Redirect URLs*. Then set `RAG_PUBLIC_URL` in `.env`/Secrets so
   confirmation emails return to the right place (see §8).
4. Copy the project URL and the **anon/publishable** key into `.env` (or
   `.streamlit/secrets.toml`):

```bash
SUPABASE_URL=https://YOUR-PROJECT.supabase.co
SUPABASE_ANON_KEY=sb_publishable_...
RAG_ADMIN_EMAILS=you@example.com
```

5. Optional but recommended: run the SQL in [`docs/SUPABASE_SETUP.md`](docs/SUPABASE_SETUP.md)
   to create `profiles` with row-level security and an avatar storage bucket.
6. Restart the app. The sign-in card appears; sign up, sign in, and your initials or avatar
   appear in the sidebar with a role badge and a **Sign out** button.

Without Supabase credentials the app runs in **local demo mode** (single synthetic user,
clearly labelled) so you can still evaluate the RAG system.

Tokens are verified on the server for every request: signature (JWKS ES256/RS256, or HS256
shared secret), `aud="authenticated"`, `iss="<project>/auth/v1"`, and `exp`.

---

## 7. Evaluation, in detail

The full write-up is in [`docs/EVALUATION.md`](docs/EVALUATION.md). The short version:

**The golden set** (`data/eval/golden_set.jsonl`) has 20 questions against the 3-document
demo corpus:

| Type | Count | Examples |
| --- | --- | --- |
| Single-fact extraction | 8 | revenue, EPS, refrigerant charge, alarm code |
| Multi-hop / cross-document | 3 | leverage 1.61x vs covenant 3.50x |
| Numeric precision | 3 | 4.2 K approach alarm, 45 N·m torque, 0.52 kW/ton |
| Table / procedure | 2 | oil change interval, auditor opinion |
| **Refusal controls** | 4 | Q3 2025 revenue (not disclosed), CEO pay, CX-1200, out-of-domain |

**Metrics** (Ragas): `faithfulness`, `answer_relevancy`, `context_recall`.
**Always computed:** `hit_rate`, `MRR`, `context_precision`, `numeric_fidelity`,
`abstention_accuracy`, latency.

**The gate** applies absolute floors *and* a baseline comparison:

```python
faithfulness      >= 0.70     context_recall >= 0.60
answer_relevancy  >= 0.55     abstention_accuracy >= 0.85
# plus: no metric may drop more than 0.05 below eval/baseline.json
```

Promote a good run to the baseline:

```bash
python scripts/evaluate.py --judge ragas --update-baseline
```

### Continuous evaluation on GitHub

`.github/workflows/rag-eval.yml` runs on every push to `main`, every PR, weekly, and on
demand. It:

1. installs CPU PyTorch and the project, caches Hugging Face models,
2. ingests the corpus,
3. runs the deterministic evaluation (works with no API key),
4. optionally runs Ragas if `RAG_OPENAI_API_KEY` is set as a repository secret,
5. uploads the report as an artefact,
6. **commits `eval/reports/**` back to `main`** with `[skip ci]`, so the metrics history lives
   in git next to the code.

A failing gate fails the workflow — and the red report is still committed, because hiding a
regression defeats the purpose.

---

## 8. Deploying the UI to Streamlit Community Cloud

1. Push to GitHub (the repo can stay private).
2. [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick the repo.
3. Main file path: **`app/streamlit_app.py`**. Python 3.12 is selected from `runtime.txt`.
4. **Advanced settings → Secrets**: paste the contents of `.streamlit/secrets.toml.example`
   with your Supabase values and either your Ollama URL (must be publicly reachable — use a
   tunnel) or an OpenAI-compatible key.
5. Deploy. First load takes 30-60 s while models download; that is exactly what the loading
   screen and the disclaimer are for.
6. Add the app URL to Supabase *Redirect URLs*, and set its public origin in Secrets:

   ```toml
   RAG_PUBLIC_URL = "https://<your-app>.streamlit.app"
   ```

   That value is the redirect target for Supabase confirmation emails. Leave it out and
   Supabase falls back to its *Site URL*; if that is still `localhost`, the confirmation link
   is unreachable for anyone but you and the account is never confirmed — the usual reason a
   Cloud deployment "cannot log in". The sign-in card states which target is in use, and
   **System → Configuration** shows `auth_redirect_url`.

> Streamlit Community Cloud cannot reach `localhost:11434` on your PC, so use the Groq block
> from `.streamlit/secrets.toml.example` (`RAG_LLM_PROVIDER=openai_compat` + `gsk_...` key).
> Keep Ollama for local work. Note that `storage/` is ephemeral on Cloud: either re-ingest on
> first boot or set `RAG_VECTOR_BACKEND=pinecone`.

Commands, ports, and troubleshooting: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

---

## 9. Configuration reference

Every setting is an environment variable (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAG_LLM_PROVIDER` | `ollama` | `ollama`, `openai_compat`, or `auto` (key-based) |
| `RAG_OLLAMA_MODEL` | `qwen2.5:3b-instruct` | local generator |
| `RAG_OPENAI_BASE_URL` / `_MODEL` / `_API_KEY` | Groq / `openai/gpt-oss-20b` / empty | hosted generator |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embeddings (changing it requires `--rebuild`) |
| `RAG_RERANK_MODEL` | `ms-marco-MiniLM-L-6-v2` | cross-encoder |
| `RAG_VECTOR_BACKEND` | `chroma` | `chroma` or `pinecone` |
| `RAG_TOP_K_DENSE` / `RAG_TOP_K_BM25` | 20 / 20 | candidate pools |
| `RAG_TOP_K_FINAL` | 5 | chunks passed to the model |
| `RAG_RERANK_MIN_SCORE` | 0.20 | retrieval floor (0-1) |
| `RAG_CHUNK_MAX_CHARS` | 1400 | hard chunk ceiling |
| `RAG_SEMANTIC_PERCENTILE` | 88 | break at the top 12% of sentence distances |
| `RAG_MAX_FILE_MB` | 10 | upload limit |
| `RAG_GROUNDING_MIN_OVERLAP` | 0.30 | minimum answer/evidence token overlap |
| `RAG_STRICT_NUMERIC` | `true` | every number must exist in the cited text |
| `RAG_RATE_LIMIT_PER_MIN` | 20 | per-user question budget |
| `SUPABASE_URL` / `SUPABASE_ANON_KEY` | empty | enables real auth |
| `RAG_ADMIN_EMAILS` | empty | comma-separated admins (re-index, delete) |
| `RAG_PUBLIC_URL` | empty | public origin of the deployment; redirect target for Supabase confirmation emails |

Using Pinecone:

```bash
pip install pinecone
RAG_VECTOR_BACKEND=pinecone
PINECONE_API_KEY=...
```

---

## 10. Security summary

Full threat model: [`docs/SECURITY.md`](docs/SECURITY.md).

- **Uploads**: extension allowlist + magic-byte sniffing + 10 MB cap + DOCX zip-bomb guard +
  content-addressed storage (no user-controlled paths).
- **Prompt injection**: every chunk is scanned; retrieved context is fenced in `<context>` with
  an explicit "this is data, not instructions" system rule.
- **Grounding**: citations are validated against the chunks actually retrieved; every number
  in the answer must appear in the cited text; otherwise → "I don't know."
- **Auth**: JWT verification on every request (signature, `aud`, `iss`, `exp`), 120-minute idle
  timeout, generic error messages (no account enumeration), avatars restricted to https image
  hosts.
- **Secrets**: env/secrets only, `.gitignore`d, redacted from logs; the anon key is the only
  client-side credential and RLS is what protects data.
- **Rate limiting**: per-user token bucket, enforced server-side.
- **Audit log**: JSONL in `storage/audit/`, questions hashed, PII redacted.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Ollama is not reachable` | not running | `ollama serve` |
| `Model 'x' is not pulled` | missing model | `ollama pull qwen2.5:3b-instruct` |
| Answers are always "I don't know." | retrieval floor too high, or model not following the prompt | lower `RAG_RERANK_MIN_SCORE`, check `Evidence & scores` |
| `Index is empty` | ingestion not run | `make ingest` |
| Evaluation `gate FAIL` | real regression | read `eval/reports/latest.md` |
| Streamlit: `Uncaught app exception` | missing secret | `python scripts/check_env.py` |
| Slow first answer | model download / cold start | expected, 30-60 s on free tier |
| `chromadb` install fails on 3.13 | old pip | `pip install -U pip` |

---

## 12. Honest limitations

- The demo corpus is **synthetic but realistic** (a fictional NYSE-listed manufacturer). Numbers
  are internally consistent (revenue = sum of segments, net debt = debt − cash, leverage =
  net debt / EBITDA) but they are not a real filing. Point it at real 10-Ks or manuals to
  evaluate on live data.
- Ragas scoring depends on the judge model. A 3B local judge is noisier than a hosted 70B one;
  the deterministic retrieval metrics are the stable signal in CI.
- The re-ranker and embedder run on CPU in Codespaces: expect ~200-600 ms per query. On an
  RTX 2060 both fit comfortably in VRAM.
- Auth protects the *session* and the UI; document-level ACLs (per-user corpora) are a natural
  next step using the `documents` table in `docs/SUPABASE_SETUP.md`.

---

## 13. Licence / attribution

MIT for this code. Models keep their own licences: BGE (MIT), ms-marco MiniLM (Apache-2.0),
Qwen2.5 (Apache-2.0), Ragas (Apache-2.0).
