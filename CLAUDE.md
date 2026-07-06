# CLAUDE.md

Guidance for working in this repository. Verified against source — where this file and
`.github/copilot-instructions.md` disagree, this file is correct (the copilot file has stale
claims about the embedding model and chunking strategy).

## Overview

Hebrew-focused **RAG pipeline** specialized for **Israeli planning & building documents**
(תכנון ובניה). End-to-end flow:

```
data/raw/ (PDF/image)
  → ingestion   PyMuPDF text extraction, Tesseract (heb) OCR fallback
  → processing  clean Hebrew text → sentence-aware chunking
  → embeddings  intfloat/multilingual-e5-large (query:/passage: prefixing)
  → index       FAISS IndexFlatIP over normalized embeddings — cosine (semantic) + Whoosh BM25 (keyword)
  → retrieval   hybrid merge → synonym expansion → plot/number rerank
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

python main.py                          # index one file (data/raw/45.pdf) + test query
python scripts/build_index_all.py       # full rebuild from everything in data/raw/
python scripts/query_helper.py          # manual smoke test (no automated tests exist)
uvicorn api.app:app --host 0.0.0.0 --port 9000   # run the API + web UI

# Docker (deployment)
docker compose up --build -d
docker compose exec app python scripts/build_index_all.py   # index inside the container
```

There are **no automated tests**. `scripts/query_helper.py` is the manual smoke test.

## Architecture

| Module | Responsibility |
| --- | --- |
| [ingestion/pdf_loader.py](ingestion/pdf_loader.py) | PyMuPDF text extraction per page; renders to image + Tesseract OCR (`lang=heb`) when a page has no text |
| [ingestion/ocr_pipeline.py](ingestion/ocr_pipeline.py) | Standalone image OCR (`ocr_image`) for `.jpg/.png` inputs |
| [processing/cleaner.py](processing/cleaner.py) | Normalize whitespace; regex keeps Hebrew `֐-׿`, `\w`, and `/ - .` so IDs like `306/02/6` survive |
| [processing/chunker.py](processing/chunker.py) | **Sentence-aware** chunking: split on page/paragraph/sentence boundaries, group to ~`CHUNK_SIZE` chars with `CHUNK_OVERLAP` carry-over |
| [embeddings/embedder.py](embeddings/embedder.py) | `SentenceTransformer` wrapper; applies E5 `query:`/`passage:` prefixes when `"e5"` is in the model name |
| [storage/vector_store.py](storage/vector_store.py) | FAISS `IndexFlatIP` over unit-normalized embeddings (cosine); persists `index.faiss` + `texts.json` + `meta.json` to `vector_store/` (atomic writes, validated on load) |
| [retrieval/hybrid_search.py](retrieval/hybrid_search.py) | Whoosh BM25 in `indexdir/`; Hebrew synonym expansion, tries AND-group then falls back to OR-group |
| [rag/pipeline.py](rag/pipeline.py) | Orchestration: query synonym expansion, hybrid merge, `_rerank` (plot/number scoring), `build_index` / `index_new_files` / `query` / `query_with_answer` |
| [rag/generator.py](rag/generator.py) | Streams from llama.cpp native `/completion`; Hebrew planning prompt template, strips `[filename]` prefixes from context |
| [downloader/](downloader/) | `manager.py` reads `sites.csv` → `DOWNLOADER_REGISTRY`; `mavat_downloader.py` queries ArcGIS REST + downloads PDFs (uses a legacy-SSL adapter for old gov servers); `base_downloader.py` ABC |
| [api/app.py](api/app.py) | FastAPI: `GET`/`POST /query`, `POST /download`, static mounts `/static` (JS) and `/files` (raw docs), inline RTL HTML home page |
| [config.py](config.py) | All tunable constants |

`RAGPipeline.__init__` auto-loads a persisted `vector_store/index.faiss` if present, so the API
is queryable on startup without re-indexing.

## Key conventions (preserve these)

- **All tunables live in [config.py](config.py)** (`EMBEDDING_MODEL`, `CHUNK_SIZE`,
  `CHUNK_OVERLAP`, `TOP_K`, `RAW_DIR`). The pipeline does `from config import *`.
- **`[filename]` provenance prefix**: every chunk is prefixed with `[<filename>] ` at index
  time (`build_index_all.py`, `index_new_files`). The API strips it to resolve a `/files/<name>`
  source URL; the generator strips it before building the prompt. Keep this pattern in any new
  ingestion path.
- **Hebrew preservation**: the cleaner regex explicitly keeps `֐-׿`. Edit it carefully
  so Hebrew characters are never dropped.
- **Hybrid dedup order matters**: `query()` merges with `dict.fromkeys(vector_results + keyword_results)`
  — vector (semantic) results come first. `_rerank` then boosts chunks where query numbers sit
  adjacent to plot terms (תא שטח / מגרש / חלקה) and rewards explicit "size of plot N" patterns.
- **Synonym maps exist in TWO places**: `_QUERY_EXPANSIONS` in [rag/pipeline.py](rag/pipeline.py)
  and `PLANNING_SYNONYMS` in [retrieval/hybrid_search.py](retrieval/hybrid_search.py). Keep both
  in sync when adding terms.
- **E5 prefixing** is keyed off `"e5" in model_name`. If you change `EMBEDDING_MODEL` to a
  non-E5 model, the prefixing turns off automatically — but a full re-index is required.
- **Incremental vs full**: `index_new_files(paths)` appends to existing FAISS + Whoosh stores;
  `build_index(chunks)` does a full rebuild.
- **Scripts** under `scripts/` insert the project root onto `sys.path` so relative imports
  work — follow this for any new script.
- **Persisted stores** (`vector_store/`, `indexdir/`) are Docker volume-mounted and must not be
  committed.
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

- No automated tests; use `scripts/query_helper.py` as a manual smoke test.
- [storage/metadata_store.py](storage/metadata_store.py) (SQLite `MetadataStore`) exists but is
  not wired into the active pipeline.
