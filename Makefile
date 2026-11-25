.PHONY: help install install-dev bootstrap ingest seed eval eval-offline serve app test lint clean check

PY ?= python
export PYTHONPATH := src

help:
	@echo "install      Install runtime deps"
	@echo "install-dev  Install runtime + dev deps"
	@echo "seed         Write the bundled 3-document demo corpus (already committed)"
	@echo "ingest       Ingest data/corpus -> storage/ (chroma + bm25)"
	@echo "eval         Ragas evaluation (needs an LLM for the judge)"
	@echo "eval-offline Deterministic evaluation with no LLM (CI-safe)"
	@echo "serve        Run the Streamlit app on :8501"
	@echo "demo         ingest + serve"
	@echo "test         Offline unit tests"
	@echo "check        Compile-all + tests + offline eval gate"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

bootstrap:
	bash scripts/bootstrap.sh

seed:
	$(PY) scripts/seed_demo.py

ingest:
	$(PY) scripts/ingest.py

eval:
	$(PY) scripts/evaluate.py --judge ragas

eval-offline:
	$(PY) scripts/evaluate.py --judge heuristic

serve:
	$(PY) -m streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

demo: ingest serve

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src app scripts tests

clean:
	rm -rf storage/chroma storage/bm25 storage/uploads .pytest_cache .ruff_cache

check: test eval-offline
