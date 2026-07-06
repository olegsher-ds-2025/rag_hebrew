# Copilot Instructions

## Project Overview

Hebrew-focused RAG (Retrieval-Augmented Generation) pipeline. Ingests PDF and image files (with OCR fallback), cleans and chunks text, builds a hybrid FAISS + BM25 index, and exposes a FastAPI query endpoint with a minimal RTL web UI.

## Running the Project

```bash
# Install dependencies (requires tesseract-ocr and tesseract-ocr-heb system packages)
pip install -r requirements.txt

# Index a single file and run a test query
python main.py

# Batch-index all PDFs/images from data/raw/
python scripts/build_index_all.py

# Start the API server (serves on http://localhost:9000)
uvicorn api.app:app --host 0.0.0.0 --port 9000

# Docker (recommended for deployment)
docker-compose up --build -d
```

There are no automated tests. `scripts/query_helper.py` serves as a manual smoke test for a specific file.

## Architecture

```
data/raw/          ← source PDFs and images
    ↓
ingestion/         ← pdf_loader.py: PyMuPDF text extraction with Tesseract OCR fallback
    ↓                  ocr_pipeline.py: standalone image OCR entry point
processing/        ← cleaner.py: normalise whitespace, strip non-Hebrew/non-ASCII chars
    ↓                  chunker.py: fixed-size sliding window (char-level)
rag/pipeline.py    ← orchestrates embed → FAISS search + BM25 search → deduplicated results
    ↓                  build_index(): full rebuild | index_new_files(): incremental append
storage/           ← vector_store.py: FAISS IndexFlatIP (cosine, normalized embeddings), persists to vector_store/
retrieval/         ← hybrid_search.py: Whoosh BM25, index lives in indexdir/
downloader/        ← manager.py: reads sites.csv, dispatches to site-specific downloaders
    ↓                  mavat_downloader.py: ArcGIS query + PDF download for iplan.gov.il
    ↓                  sites.csv: registry of enabled download sources
api/app.py         ← FastAPI: GET+POST /query, POST /download, static file serving, RTL HTML UI
```

`RAGPipeline` auto-loads any persisted `vector_store/index.faiss` on startup, so the API is ready to query without re-indexing.

## Key Conventions

- **All tunable constants live in `config.py`** (`EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `RAW_DIR`), imported explicitly where needed.

- **Chunk provenance prefix**: `scripts/build_index_all.py` and `downloader/` both prepend `[filename]` to every chunk (e.g. `[45.pdf] chunk text...`). The API strips this prefix and resolves `source` URLs from it. Maintain this pattern when adding ingestion scripts.

- **Hebrew text preservation**: `processing/cleaner.py` keeps Unicode range `\u0590-\u05FF` explicitly alongside `\w`, `/`, `-`, `.`. Extend the regex carefully to avoid dropping Hebrew characters.

- **Hybrid deduplication**: `RAGPipeline.query()` merges vector and keyword results with `dict.fromkeys(...)` — order matters (vector results first). The combined list is returned directly as chunk strings to the API.

- **Embedding model**: `intfloat/multilingual-e5-large` (multilingual, 1024-dim), set in `config.py`. The embedder auto-prepends E5 task prefixes (`query: ` / `passage: `). The Dockerfile pre-bakes the model into the image layer.

- **Answer generation**: `rag/generator.py` calls a llama.cpp `llama-server` (`LLAMACPP_URL`, native `/completion` streaming endpoint). `RAGPipeline.query_with_answer()` reranks retrieved chunks (plot-number/size heuristics) before synthesis. See CLAUDE.md for the authoritative architecture description.

- **Persisted stores**: `vector_store/` holds `index.faiss` + `texts.json`; `indexdir/` holds the Whoosh BM25 index. Both are volume-mounted in Docker. Do not check these into git.

- **Adding a new download site**: add a row to `downloader/sites.csv` (set `enabled=true`), create `downloader/<type>_downloader.py` inheriting `BaseDownloader`, and register the type in `DOWNLOADER_REGISTRY` in `downloader/manager.py`.

- **Incremental vs full index**: use `rag.index_new_files(file_paths)` to append new files to the existing FAISS + Whoosh stores without touching existing data. Use `build_index(chunks)` only for a full rebuild.

- **Scripts need path fix**: scripts under `scripts/` do `sys.path.insert(0, project_root)` at the top so relative imports work. Follow this pattern for any new scripts.

- **RTL UI**: The web UI (`api/app.py`, `api/static/app.js`) sets `direction: rtl` for both input and output elements. Keep this for Hebrew content.
