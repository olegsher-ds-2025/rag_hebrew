# CLAUDE.md

Guidance for working in this repository. Verified against source — where this file and
`.github/copilot-instructions.md` disagree, this file is authoritative.

## Overview

Hebrew-focused **RAG pipeline** specialized for **Israeli planning & building documents**
(תכנון ובניה). End-to-end flow:

```
data/raw/ (PDF/image)
  → ingestion   PyMuPDF text extraction, Tesseract (heb) OCR fallback
  → processing  clean Hebrew text → sentence-aware chunking
  → embeddings  intfloat/multilingual-e5-large (query:/passage: prefixing)
  → index       FAISS IndexFlatIP over normalized embeddings — cosine (semantic) + Whoosh BM25 (keyword)
  → retrieval   RRF fusion of vector+keyword → synonym expansion → plot/number rerank
  → generation  streaming llama.cpp /completion → concise Hebrew answer
  → api         FastAPI + minimal RTL web UI (port 9000)
```

A downloader fetches plans from `mavat.iplan.gov.il` by plan name (שם תכנית). Target
deployment is an **NVIDIA Jetson Orin Nano**: the app runs in Docker while the LLM runs as a
`llama-server` on the host GPU (see [DEPLOY.md](DEPLOY.md)).

## Commands

```bash
# Install (also needs system packages: tesseract-ocr, tesseract-ocr-heb)
pip install -r requirements.txt
playwright install chromium              # install Chromium for mavat document scraping (optional)

python main.py                          # index one file (data/raw/45.pdf) + test query (non-destructive)
python scripts/build_index_all.py       # full rebuild from everything in data/raw/
python scripts/query_helper.py          # manual smoke test against a live index
uvicorn api.app:app --host 0.0.0.0 --port 9000   # run the API + web UI

# Tests (no torch / OCR / network needed — FakeEmbedder + mocks)
pip install -r requirements-dev.txt
pytest                                  # ~100 unit/integration/api tests
ruff check .                            # lint (config in ruff.toml)

# Docker (deployment)
docker compose up --build -d
docker compose exec app python scripts/build_index_all.py   # index inside the container
```

The `tests/` suite (unit/integration/api) runs in ~1s with no model download and is
enforced by CI (`.github/workflows/ci.yml`). `scripts/query_helper.py` remains a manual
smoke test against a real index.

## Architecture

| Module | Responsibility |
| --- | --- |
| [ingestion/pdf_loader.py](ingestion/pdf_loader.py) | PyMuPDF text extraction per page; renders to image + Tesseract OCR (`lang=heb`) when a page has no text |
| [ingestion/ocr_pipeline.py](ingestion/ocr_pipeline.py) | Standalone image OCR (`ocr_image`) for `.jpg/.png` inputs |
| [processing/cleaner.py](processing/cleaner.py) | Normalize horizontal whitespace but **preserve newlines** (so paragraph chunking works); regex keeps Hebrew `֐-׿`, `\w`, `/ - .` (IDs like `306/02/6`) and `" ' : ( ) %` (gershayim units like `מ"ר`, structured fields like `שם התכנית:`) |
| [processing/chunker.py](processing/chunker.py) | **Sentence-aware** chunking: split on page/paragraph/sentence boundaries, group to ~`CHUNK_SIZE` chars with `CHUNK_OVERLAP` carry-over |
| [embeddings/embedder.py](embeddings/embedder.py) | `SentenceTransformer` wrapper; **lazy** (model builds on first `encode()`/`warmup()`, so importing never pulls torch); applies E5 `query:`/`passage:` prefixes when `"e5"` is in the model name; **unit-normalizes** output (cosine via IP); `batch_size` from settings |
| [storage/vector_store.py](storage/vector_store.py) | FAISS `IndexFlatIP` over unit-normalized embeddings (cosine); persists `index.faiss` + `texts.json` + `meta.json` to `vector_store/` (atomic temp+`os.replace`). `load()` validates the `meta.json` version + `ntotal == len(texts)` and raises `StoreCorruptError` on stale/desynced stores (see rebuild note below) |
| [retrieval/hybrid_search.py](retrieval/hybrid_search.py) | Whoosh BM25 in `indexdir/`; imports synonyms from `processing/synonyms.py`, quotes multi-word synonyms as phrases, tries AND-group then falls back to OR-group; `clear()` recreates the index for full rebuilds |
| [processing/synonyms.py](processing/synonyms.py) | **Single source** for the Hebrew planning synonym map (`PLANNING_SYNONYMS`) + `expand_words()`, shared by keyword search and rerank |
| [rag/pipeline.py](rag/pipeline.py) | Orchestration: query synonym expansion, hybrid merge, `_rerank` (plot/number scoring), `build_index` / `index_new_files` / `index_texts` / `query` / `query_with_answer`. Injectable `embedder`/`store_dir`/`index_dir` (DI for tests); a `threading.Lock` serializes all index mutation |
| [rag/generator.py](rag/generator.py) | Streams from llama.cpp native `/completion`; Hebrew planning prompt template, strips `[filename]` prefixes. Returns `None` when llama-server is unreachable/errors (vs `""` for no context) so the API can surface "model unavailable" |
| [downloader/](downloader/) | `manager.py` reads `sites.csv` → `DOWNLOADER_REGISTRY` and instantiates a downloader **per request**; `mavat_downloader.py` queries ArcGIS REST + downloads PDFs (legacy-SSL adapter scoped to the two gov hosts, **cert verification stays on**; retries; allowlist-sanitized filenames); `base_downloader.py` ABC |
| [api/app.py](api/app.py) | FastAPI (`lifespan` builds the pipeline + `warmup()`s the model at startup so `/health` reflects readiness): `POST /query`, `POST /download` (busy-locked → 409, optional `API_TOKEN`), `GET /health`, static mounts `/static` (JS) and `/files` (raw docs), inline RTL HTML home page |
| [config.py](config.py) | pydantic-settings `Settings` + `settings` singleton — every tunable is a field, overridable via env var of the same name (case-insensitive) or `.env`. See `.env.example` |

`RAGPipeline.__init__` auto-loads a persisted `vector_store/index.faiss` if present (and
valid), so the API is queryable on startup without re-indexing.

> **Metric/format change:** the store is cosine (`IndexFlatIP`, normalized) with a
> `meta.json` version marker. Pre-existing L2 stores are **refused on load** — rebuild
> once with `python scripts/build_index_all.py` (or `docker compose exec app ...`).

## Key conventions (preserve these)

- **All tunables live in [config.py](config.py)** as fields on the pydantic-settings
  `Settings` object; consumers do `from config import settings` and read
  `settings.<field>`. Every field is env-overridable (e.g. `TOP_K`, `EMBEDDING_BATCH_SIZE`,
  `LLAMACPP_URL`, `API_TOKEN`, `MAVAT_INSECURE_SSL`) — read once at import. Add new
  tunables as fields, not scattered `os.getenv` calls.
- **`[filename]` provenance prefix**: every chunk is prefixed with `[<filename>] ` at index
  time (`build_index_all.py`, `index_new_files`). The API strips it to resolve a `/files/<name>`
  source URL; the generator strips it before building the prompt. Keep this pattern in any new
  ingestion path.
- **Hebrew preservation**: the cleaner regex explicitly keeps `֐-׿`. Edit it carefully
  so Hebrew characters are never dropped.
- **Hybrid fusion is RRF**: `query()` merges the vector and keyword ranked lists with
  `_reciprocal_rank_fusion` (score `Σ 1/(RRF_K + rank)`), so chunks both retrievers rank
  highly win; ties keep vector-before-keyword order. `_rerank` then runs as a second stage,
  boosting chunks where query numbers sit adjacent to plot terms (תא שטח / מגרש / חלקה) and
  rewarding explicit "size of plot N" patterns. `RRF_K` lives in `config.py`.
- **Synonyms live in ONE place**: `PLANNING_SYNONYMS` in
  [processing/synonyms.py](processing/synonyms.py); both keyword search and rerank import
  it. Add terms there only.
- **E5 prefixing** is keyed off `"e5" in model_name`. If you change `EMBEDDING_MODEL` to a
  non-E5 model, the prefixing turns off automatically — but a full re-index is required.
- **Incremental vs full**: `index_new_files(paths)` appends to existing FAISS + Whoosh stores
  and **skips already-indexed files** via `vector_store/manifest.json` (filename + size/mtime),
  so re-downloading a plan does not duplicate chunks; `build_index(chunks)` does a full rebuild
  (clears Whoosh first so the two stores stay in sync).
- **Scripts** under `scripts/` insert the project root onto `sys.path` so relative imports
  work — follow this for any new script.
- **Persisted stores & data** (`vector_store/`, `indexdir/`, `data/`) are Docker
  volume-mounted, gitignored, and must not be committed.
- **LLM endpoint** is llama.cpp's **native `/completion`** (NOT the OpenAI-compatible `/v1`).
  Configure via env: `LLAMACPP_URL`, `LLAMACPP_N_PREDICT`, `LLAMACPP_TEMPERATURE`
  ([rag/generator.py](rag/generator.py)).
- **Adding a download site**: add a row to `downloader/sites.csv` (`enabled=true`), create a
  `BaseDownloader` subclass, and register its `type` in `DOWNLOADER_REGISTRY`
  ([downloader/manager.py](downloader/manager.py)).
- **RTL UI**: the web UI sets `direction: rtl` for input and output. Preserve for Hebrew.

## Deployment

The app container talks to a `llama-server` running on the Jetson host (for GPU access) over
HTTP. `docker-compose.yml` maps `host.docker.internal:host-gateway` and sets
`LLAMACPP_URL=http://host.docker.internal:8080/completion`. The app is served on port **9000**.
Full instructions and troubleshooting are in [DEPLOY.md](DEPLOY.md).

## Notes

- Tests live in `tests/` (`pytest`), run without torch/OCR/network, and are enforced by
  `.github/workflows/ci.yml`. `scripts/query_helper.py` is a manual smoke test against a
  real index.
- Security-sensitive spots to preserve: `api/static/app.js` escapes all HTML before
  `innerHTML` and only trusts `/files/<basename>` sources; `api/app.py` `_safe_source`
  rejects non-basename/absent filenames; the mavat downloader keeps TLS verification on
  and allowlist-sanitizes remote filenames + the ArcGIS `where` input.
