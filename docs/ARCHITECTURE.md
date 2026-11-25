# Architecture

## 1. Request lifecycle

```
user question
   │
   ├─ clean_question()          NFC normalise, strip control chars, 1200-char cap
   ├─ rate limit                token bucket, 20/min/user (server-side)
   │
   ├─ follow-up rewrite         only when chat history exists (LLM, 1 short call)
   │
   ├─ RETRIEVAL
   │    ├─ embed(query)                 bge-small, query instruction prefix
   │    ├─ Chroma cosine top-20  ──┐
   │    ├─ BM25Okapi top-20      ──┤
   │    └─ RRF fuse ───────────────┴─▶ 25 candidates
   │         └─ cross-encoder re-rank ──▶ top-5, score ≥ 0.20
   │
   ├─ GENERATION
   │    context = numbered blocks: "[1] file (page 3) — heading\n<text>"
   │    system   = answer only from context, cite [n], copy numbers exactly,
   │               abstain sentence, injection rules
   │
   ├─ VERIFICATION (in this order, first failure wins)
   │    1. empty answer?                          → abstain
   │    2. refusal phrase detected?                → normalise to IDK, no citations
   │    3. no citations?                           → infer chunks with ≥50% overlap,
   │                                                 else abstain
   │    4. citation index out of range?            → abstain
   │    5. token coverage below 0.30?              → abstain (low_grounding)
   │    6. number not present in cited text?       → abstain (unsupported_number)
   │    otherwise → answer with citations
   │
   └─ answer + citation objects + retrieval trace + latency
```

The verification stage is what makes the failure mode *safe*: a 3B model that hallucinates a
number gets caught deterministically, without needing a bigger judge model.

## 2. Ingestion

```
file bytes
  ├─ validate_upload        extension → size → magic bytes → zip-bomb ratio → UTF-8
  ├─ extract                pypdf page-by-page | python-docx blocks+tables | bs4 | text
  ├─ normalise              CRLF, bullets, ligatures, collapse whitespace
  ├─ scan_for_injection     flags stored on the chunk (kind + snippet), never silently dropped
  └─ chunk_pages
       ├─ split_sections        markdown headings → structure-aware sections
       ├─ split_sentences       keeps table rows / numbered steps intact
       ├─ embed sentences       bge-small, batch 32
       ├─ cosine distance       d_i = 1 - cos(e_i, e_{i+1})
       ├─ break at percentile   threshold = P(88) of distances
       ├─ hard ceiling          ≤1400 chars (recursive split)
       └─ heading prefix        each chunk starts with its section heading
```

### Why semantic chunking beats fixed windows

A 1000-character window cuts a covenant clause in half and separates a table from its header.
Sentence-embedding distance finds the *topic boundary* instead, so a chunk like
"Section 7.1 Maximum Leverage … 3.50 to 1.00" stays whole even if it is 900 characters, and a
list of alarm codes stays grouped. The 1400-character ceiling guarantees the chunk still fits
comfortably inside a small model's context.

### Why RRF instead of score averaging

Dense cosine similarity lives in `[0,1]` (typically 0.55-0.9 for relevant text); BM25 scores are
unbounded (0-30) and corpus-dependent. Averaging requires per-corpus calibration that silently
breaks when documents change. RRF uses rank only:

```
score(d) = Σ_r 1 / (60 + rank_r(d))     over retrievers r
```

`k=60` damps the influence of exact rank position, which is the published default and works
without tuning. It cannot regress when a new document changes BM25's score distribution.

### Why the cross-encoder floor

`ms-marco-MiniLM-L-6-v2` outputs a logit per (query, passage). We map it through a sigmoid to
`[0,1]` so the threshold is interpretable and stable. At 0.20 the system retrieves when the
passage is weakly relevant and abstains when nothing is; per-corpus, inspect the `Retrieval
trace` panel and tune `RAG_RERANK_MIN_SCORE`.

## 3. Storage layout

| Artefact | Path | Contents |
| --- | --- | --- |
| Chunk text | `storage/chunks.jsonl` | source of truth for BM25 + citations |
| Vectors | `storage/chroma/` | Chroma persistent collection (cosine) |
| Keyword index | `storage/bm25/index.pkl` | pickled BM25 + corpus hash (staleness check) |
| Manifest | `storage/manifest.json` | doc ids, page counts, chunk counts, embed model |
| Uploads | `storage/uploads/<sha>.ext` | content-addressed originals |
| Audit | `storage/audit/audit-YYYY-MM-DD.jsonl` | redacted events |

`chunks.jsonl` is deliberately plain JSON lines: the whole index can be rebuilt from it, and a
reviewer can `grep` it during an incident.

## 4. Concurrency and caching

| Concern | Approach |
| --- | --- |
| Model load | `SentenceTransformer` loaded once per process, guarded by a `Lock` |
| Streamlit | `@st.cache_resource` for pipeline, auth service and rate limiter |
| Embedding batches | 64 chunks per `upsert`, 256 ids per Chroma call |
| BM25 rebuild | full rebuild on ingest only (10 MB corpus ≈ 1-2 s) |
| Health check | `pipeline.health()` — index + LLM, cached by Streamlit |

Streamlit runs one script per session; the cached pipeline is shared, so concurrent users reuse
the loaded models. Rate limiting is per user id, in-process.

## 5. Failure modes and what the system does

| Failure | Detection | Behaviour |
| --- | --- | --- |
| LLM server down | `LLMError` | "I don't know." + a redaction-safe error in the UI |
| LLM ignores citations | citation parse returns none | infer supporting chunks; if none, abstain |
| Hallucinated number | numeric grounding | abstain, reason `unsupported_number:…` |
| Paraphrased unsupported claim | token coverage | abstain, reason `low_grounding:0.18` |
| Retrieved junk | re-rank floor | abstain, reason `no_relevant_context` |
| Prompt injection in a document | scan at ingest + fencing | chunk still retrievable, instructions ignored, flag surfaced |
| Embedding model changed | manifest comparison | `check_env.py` warns; `--rebuild` required |
| Corrupt PDF | loader exception | skipped with a warning, ingestion continues |
| Duplicate upload | content hash `doc_id` | old vectors deleted, new ones written |

## 6. Extension points

- **Different embedding model**: set `RAG_EMBED_MODEL`, rerun `--rebuild`. The manifest records
  which model built the index.
- **Different vector store**: implement `upsert / search / delete_doc / count / reset` and
  register it in `vectorstore.get_store`.
- **Different generator**: implement `BaseLLM.complete` (+ optional `stream`) in `llm.py`.
- **Per-user documents**: add `owner_id` to chunk metadata, filter in `ChromaStore.search`
  (`where={"owner_id": ...}`) — the plumbing already accepts a `source` filter.
- **More metrics**: add a metric class in `ragkit/eval/metrics.py` and a floor in
  `report.GATE_FLOORS`.
