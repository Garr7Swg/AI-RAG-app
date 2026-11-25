# Security model

Scope: the Streamlit app, the ingestion path, the RAG prompt boundary, and the credentials the
system touches. "Enterprise grade on a free tier" means making deliberate trade-offs and stating
them, not claiming perfection.

## 1. Threat model

| # | Threat | Impact | Control |
| --- | --- | --- | --- |
| T1 | Malicious upload (executable, zip bomb, path traversal) | RCE / disk exhaustion | extension allowlist, magic bytes, size cap, zip-bomb ratio, content-addressed storage |
| T2 | Stolen/forged session token | account takeover | server-side JWT verification (signature + aud + iss + exp), idle timeout |
| T3 | Account enumeration | privacy leak | generic auth errors, identical sign-up response for existing emails |
| T4 | Prompt injection inside a document | data exfiltration / policy bypass | injection scan at ingest, `<context>` fencing, system rule "data is not instructions", output sanitisation |
| T5 | Hallucinated answer shown as fact | business harm | citation validation + grounding overlap + strict numeric grounding → "I don't know." |
| T6 | Secret leakage via repo/logs | full compromise | env/secrets only, `.gitignore`, redaction in logs and UI errors |
| T7 | Abuse / cost attack on the LLM | quota exhaustion | per-user token bucket, question length cap |
| T8 | XSS through model output or avatar URL | session theft | model output rendered as markdown without raw HTML; HTML/`javascript:` stripped; avatars restricted to https image hosts on known domains |
| T9 | Privilege escalation to admin actions | index tampering | admin role derived server-side from `RAG_ADMIN_EMAILS`, never from the client |
| T10 | Data exfiltration across users (future multi-tenant) | confidentiality | `user_id` in metadata + filtered queries + Supabase RLS (schema provided) |

## 2. Authentication flow

```
browser form ──► Streamlit server ──► Supabase GoTrue (sign_in_with_password)
                        │                        │
                        │◄── access + refresh ───┘
                        │
                        ├─ verify token locally: header alg → key (JWKS or HS256 secret)
                        │   checks: exp, sub, aud="authenticated", iss="<url>/auth/v1"
                        ├─ derive role from verified email (RAG_ADMIN_EMAILS)
                        └─ keep tokens in st.session_state (server-side session), 120-min idle timeout
```

Why not `st.login`? Streamlit's built-in login only supports generic OIDC providers; Supabase is
not one. Why not a cookie? A cookie readable by JavaScript in a Streamlit app adds XSS surface for
little benefit; `st.session_state` is server-side and lost on browser close, which is an acceptable
trade for a demo console.

Token storage notes:

- Never placed in a URL, never written to disk, never logged.
- The refresh token is used only when the access token is within 30 s of expiry.
- Sign-out clears local state even if the Supabase call fails.

## 3. Upload pipeline controls

| Control | Implementation |
| --- | --- |
| Size | `RAG_MAX_FILE_MB` (10) checked before parsing, and `server.maxUploadSize` in Streamlit |
| Type | allowlist `.pdf .md .markdown .txt .text .docx .html .htm` |
| Content | `%PDF-` header for PDFs; OOXML `[Content_Types].xml` for DOCX; NUL-byte rejection for text |
| Zip bomb | ≤25 MB uncompressed, ≤120:1 ratio, no `..` or absolute paths inside the archive |
| Path traversal | `Path(name).name`, sanitised charset, leading dots rejected, stored as `<sha256[:16]>.ext` |
| Encrypted PDF | rejected with `encrypted_pdf` |
| Scanned PDF | rejected with `no_text` (no silent empty documents) |
| Duplicate | same hash → same `doc_id` → replaced, not duplicated |

Rejections are surfaced to the user with a machine-readable reason and written to the audit log.

## 4. Prompt-injection defence

Three layers, because no single one is sufficient:

1. **Detection** (`security.scan_for_injection`): override phrasing, system-prompt requests,
   role hijack, exfiltration verbs, chat-template tokens (`<|im_start|>`), fake delimiters.
   Findings are stored on the chunk and shown in the library table.
2. **Fencing** (`prompts.build_user_prompt`): retrieved text lives inside `<context>` and each
   answer is requested with numeric markers. The system prompt states that context is untrusted
   data and that instructions inside it must not be followed, and adds an explicit notice when
   flagged chunks were retrieved.
3. **Output constraints**: the model is told never to reveal the prompt, never to emit URLs,
   code or HTML. Rendering strips HTML and `javascript:` regardless, so a bypass cannot become XSS.

Residual risk: a sufficiently creative injection can still influence the *content* of an answer.
The numeric-grounding and citation checks limit the damage to text that is at least present in
the corpus, and the answer is always attributed to a page.

## 5. Grounding enforcement (anti-hallucination)

Implemented in `RAGPipeline._verify` and applied to every answer, in order:

1. empty answer → abstain;
2. refusal phrase → normalise to the canonical "I don't know." sentence;
3. no citation markers → attach chunks with ≥50% answer-token overlap, else abstain;
4. citation out of range → abstain;
5. token coverage of the answer by the cited text `< 0.30` → abstain;
6. any number in the answer absent from the cited text → abstain (strict mode).

The UI shows the abstain reason (`no_relevant_context`, `low_grounding:0.18`,
`unsupported_number:6240`, …), so failures are diagnosable rather than mysterious.

## 6. Secrets

| Secret | Where it lives | Exposure |
| --- | --- | --- |
| `SUPABASE_ANON_KEY` | env / Streamlit secrets | public by design; RLS enforces access |
| `SUPABASE_JWT_SECRET` | env / Streamlit secrets | never sent to the browser; only for legacy HS256 projects |
| `RAG_OPENAI_API_KEY` | env / Streamlit secrets / GitHub secret | server-side only; never in client code |
| `PINECONE_API_KEY` | env | server-side only |

`.gitignore` excludes `.env*` (except `.env.example`), `.streamlit/secrets.toml`, `*.pem`,
`*.key`. `logging_utils.redact` removes emails, bearer tokens, JWTs and `sk-`/`sbp_`/`ghp_` keys
from any string that reaches a log or an error banner. The app never renders a raw exception.

Rotating a leaked key: revoke it in the provider console, update the secret, redeploy. No code
change is required because nothing is hardcoded.

## 7. Rate limiting and resource safety

- Token bucket per user id, `RAG_RATE_LIMIT_PER_MIN` (20), refill continuous, thread-safe.
- Question length capped at 1200 characters before any model call.
- Context is capped at ~7000 characters, so a pathological document cannot blow the context window.
- Ingestion runs sequentially with bounded batches (64 chunks) to avoid OOM on free hardware.

## 8. Audit logging

`storage/audit/audit-YYYY-MM-DD.jsonl`, one JSON object per event:

```json
{"ts":"2026-11-21T09:12:44+00:00","event":"rag.answer","user_id":"a1b2c3d4e5f6","abstained":false,
 "reason":"ok","latency_ms":812.4,"evidence":2}
```

Events: `auth.sign_in`, `auth.sign_in_failed`, `auth.sign_up`, `auth.sign_out`, `rag.answer`,
`library.upload`, `library.delete`, `eval.run`. Questions are **not** stored verbatim (only the
event and metrics), emails are redacted. This is deliberate: a document-Q&A log easily becomes a
sensitive-data repository.

## 9. Supabase hardening checklist

- [ ] Email confirmation enabled for anything beyond a demo.
- [ ] Row Level Security enabled on every table (the SQL in `SUPABASE_SETUP.md` does this).
- [ ] No service-role key anywhere in this repo or its environment.
- [ ] Redirect URLs restricted to your exact domains (no wildcards).
- [ ] Password policy ≥12 characters in the Supabase dashboard.
- [ ] JWT expiry kept short (1 hour default) — the app refreshes automatically.
- [ ] Storage bucket for avatars is public-read, user-write-own; no other buckets are public.

## 10. Residual risks (stated honestly)

| Risk | Why it remains |
| --- | --- |
| Session fixation / XSS in the hosting platform | Streamlit's own surface; mitigated by static-only HTML injection and strict CSP headers where the host allows them |
| Denial of service | no WAF on the free tier; the rate limiter is in-process, so it resets on restart |
| Model-level memorised knowledge | a 3B model may still know public facts; the prompt forbids using them but cannot prove compliance — the citation checks do |
| Document ACLs | not implemented; all signed-in users share one corpus (schema and metadata field are ready for per-document ownership) |
| Audit log immutability | a local append-only file, not a WORM store; ship it to S3/Object Lock for compliance use |
| Secrets in process memory | unavoidable in any Python app |
