# Runbook

## Commands

```bash
make install-dev        # dependencies (runtime + dev)
make seed               # (re)write the bundled demo corpus — already committed
make ingest             # build storage/chroma + storage/bm25 + manifest
make ingest ARGS="--rebuild"
make serve              # Streamlit on 0.0.0.0:8501
make demo               # ingest + serve
make test               # 52 offline unit tests
make eval-offline       # deterministic judge, no model server (CI gate)
make eval               # Ragas judge (needs Ollama or RAG_OPENAI_API_KEY)
make check              # tests + offline eval
python scripts/check_env.py
```

## Ports

| Port | Service |
| --- | --- |
| 8501 | Streamlit UI |
| 11434 | Ollama (local models) |

## Cold start expectations

| Phase | Time (Codespaces CPU) | Time (i7-9700K + RTX 2060) |
| --- | --- | --- |
| `pip install -r requirements.txt` | 3-6 min (first time) | 2-4 min |
| First ingest (model download 130 MB + 90 MB) | 2-4 min | 1-2 min |
| Later ingest (3 docs, ~30 chunks) | 3-8 s | 2-4 s |
| First question after restart | 20-60 s | 3-6 s |
| Subsequent questions | 3-15 s (hosted) | 1-4 s |

These are estimates for this hardware class, not benchmarks from a specific run.

## Incident playbook

### "I don't know." for everything

1. `python scripts/check_env.py` → is the index non-empty?
2. In the UI, expand **Retrieval trace**: if all scores are `< 0.20`, the re-ranker floor is too
   high for your corpus → set `RAG_RERANK_MIN_SCORE=0.10`.
3. If retrieval looks right but the answer still abstains, check the reason shown in the answer
   footer: `low_grounding` means the model paraphrased away from the source (use a stronger
   model or lower `RAG_GROUNDING_MIN_OVERLAP` to 0.20);
   `unsupported_number:…` means it invented or converted a number — that is the guardrail working.

### Ragas score dropped

```bash
git log --oneline eval/reports | head
git diff HEAD~1 eval/baseline.json
python scripts/evaluate.py --judge ragas --json | head -100
```

Look at the per-question rows: a single broken document or a stricter judge model shifts scores.
Never "fix" a regression by lowering a floor without a written reason — update the baseline
deliberately instead.

### Index corruption / wrong embedding model

```bash
make ingest ARGS="--rebuild"
```

`check_env.py` compares the configured embedding model with the manifest and warns on mismatch.
Changing `RAG_EMBED_MODEL` without a rebuild produces meaningless similarities — always rebuild.

### Ollama out of memory (6 GB GPU)

```bash
ollama ps                       # what is loaded
ollama stop qwen2.5:3b-instruct
# prefer a smaller quantisation
ollama pull qwen2.5:3b-instruct-q4_K_M
```

Only one model should be resident; the embedder and re-ranker run on CPU by default
(`sentence-transformers` uses CUDA if available, which is fine at ~130 MB).

### "I cannot sign in" on Streamlit Cloud

Work down this list; each item is a distinct failure with a different symptom.

| Symptom | Cause | Fix |
| --- | --- | --- |
| No sign-in card, sidebar says *Demo mode* | Secrets never reached the process | Keys must be named exactly `SUPABASE_URL` / `SUPABASE_ANON_KEY`; save, then **Reboot app** |
| *Please confirm your email address, then sign in.* | Confirmation link pointed at `localhost` | Set `RAG_PUBLIC_URL` to the app origin and add it to Supabase *Redirect URLs* + *Site URL* |
| *Supabase is throttling confirmation emails…* | Free-tier email rate limit | Wait an hour, or turn off *Confirm email* while testing |
| *Email or password is incorrect.* | Wrong password, unconfirmed user, or paused project | Supabase → Authentication → Users; check the project is not paused |
| *SUPABASE_ANON_KEY was rejected…* | Service-role key pasted instead of anon | Paste the anon/publishable key, reboot |
| Signs in, then bounces back to the form | Idle timeout or expired/refresh failure | `RAG_SESSION_TIMEOUT_MIN` (default 120); check Supabase JWT expiry |

`RAG_PUBLIC_URL` is optional locally (the Supabase *Site URL* is used) and required on Cloud.
The app never hardcodes a host: the sign-in card prints the redirect target it will use and
**System → Configuration** exposes it as `auth_redirect_url`.

### Streamlit Cloud app won't start

1. Check the app logs in the Streamlit dashboard.
2. `runtime.txt` must say `python-3.12` — several dependencies have no 3.13 wheels on the cloud image.
3. Secrets: keys must match the env var names exactly (`SUPABASE_URL`, not `supabase_url`).
4. `server.maxUploadSize = 10` in `.streamlit/config.toml` — not `max_upload_size`.

### Leaked credential

1. Revoke in Supabase / Groq / Pinecone.
2. Issue a new key, update `.env` and Streamlit secrets (and the GitHub repo secret).
3. `git log -p -- .env` should show nothing — if it does, rotate, then rewrite history.
4. Check `storage/audit/*.jsonl` for unexpected usage patterns (values are redacted by design).

## Backups

| Artefact | How to back up |
| --- | --- |
| Corpus + golden set | git (this repo) |
| Chroma index | rebuilt from `chunks.jsonl`; no backup needed |
| Evaluation history | committed under `eval/reports/` by CI |
| Chat transcripts | `st.session_state` only — downloaded by the user, or add the Supabase tables |

## Useful one-liners

```bash
# how many chunks per document?
python -c "import json;m=json.load(open('storage/manifest.json'));print({v['source']:v['chunks'] for v in m['documents'].values()})"

# inspect a chunk
head -1 storage/chunks.jsonl | python -m json.tool

# was anything flagged for prompt injection?
python -c "import json;print([json.loads(l)['source'] for l in open('storage/chunks.jsonl') if json.loads(l)['injection_flags']])"
```
