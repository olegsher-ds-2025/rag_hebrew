# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Overview

Hebrew-focused RAG (Retrieval-Augmented Generation) pipeline for Israeli planning and
construction documents (MAVAT / iplan.gov.il). It ingests PDFs and images (OCR
fallback), cleans and chunks the text, builds a hybrid **FAISS (semantic) + Whoosh BM25
(keyword)** index, retrieves and re-ranks chunks, then synthesises a concise Hebrew
answer with a local **Ollama** LLM. A FastAPI app exposes the query/download endpoints
and a minimal right-to-left (RTL) web UI.

## Commands

There is **no automated test suite and no linter configured**; `scripts/query_helper.py`
is the manual smoke test.

```bash
# Install (needs system packages: tesseract-ocr, tesseract-ocr-heb)
pip install -r requirements.txt

# Quickstart: index a single file (data/raw/45.pdf) and run a test query
python main.py

# Batch-index all PDFs/images in data/raw/ into the persisted indices
python scripts/build_index_all.py

# Manual smoke test (chunk + query a specific file)
python scripts/query_helper.py

# Run the API (serves http://localhost:8000 with the RTL UI)
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Docker (recommended for deployment)
docker-compose up --build -d
```

## Architecture / data flow

```
data/raw/ (PDFs, images)
  → ingestion/pdf_loader.py     PyMuPDF text extraction, Tesseract OCR fallback
  → processing/cleaner.py       normalise whitespace, preserve Hebrew + ASCII
  → processing/chunker.py       sliding-window chunks (CHUNK_SIZE / CHUNK_OVERLAP)
  → rag/pipeline.py (RAGPipeline) — orchestrator:
        embeddings/embedder.py      SentenceTransformer encode (E5)
        storage/vector_store.py     FAISS IndexFlatL2 → vector_store/
        retrieval/hybrid_search.py  Whoosh BM25 → indexdir/
  → rag/generator.py            synthesise answer via Ollama
  → api/app.py                  FastAPI: GET/POST /query, POST /download, RTL HTML UI

downloader/  manager.py reads sites.csv, dispatches to mavat_downloader.py to fetch
             plans on demand; downloaded files are indexed incrementally.
```

`RAGPipeline` auto-loads any persisted `vector_store/index.faiss` on startup, so the API
is ready to query without re-indexing.

## Key conventions

- **All tunable constants live in `config.py`**: `EMBEDDING_MODEL`, `CHUNK_SIZE=800`,
  `CHUNK_OVERLAP=200`, `TOP_K=10`, `RAW_DIR`. The pipeline does `from config import *`.
- **Embedding model is `intfloat/multilingual-e5-large`.** `embeddings/embedder.py`
  detects `"e5"` in the name and auto-prepends E5 task prefixes (`query: ` for queries,
  `passage: ` for documents) — do not remove these.
- **Chunk provenance prefix**: every chunk is prefixed `[filename]` (e.g.
  `[45.pdf] ...`). The API and `rag/generator.py` strip it to build `source` links.
  Preserve this in any new ingestion path.
- **Hebrew preservation**: `processing/cleaner.py` explicitly keeps the Unicode range
  `֐-׿` alongside `\w`, `/`, `-`, `.`. Edit that regex carefully so Hebrew
  characters are not dropped.
- **Hybrid dedup & re-ranking**: `RAGPipeline.query()` merges vector results first, then
  keyword results, via `dict.fromkeys` (order matters). It applies planning-domain query
  expansion (פלוט / מגרש / חלקה / תא שטח synonyms) and boosts chunks mentioning
  plot-adjacent numbers and plot sizes.
- **Incremental vs full index**: `index_new_files(file_paths)` appends to the existing
  FAISS + Whoosh stores; `build_index(chunks)` does a full rebuild. The downloader uses
  the incremental path.
- **Persisted stores (gitignored, never commit)**: `vector_store/` holds `index.faiss` +
  `texts.json`; `indexdir/` holds the Whoosh BM25 index. Both are volume-mounted in
  Docker.
- **Scripts need a path fix**: scripts under `scripts/` do
  `sys.path.insert(0, project_root)` at the top so package imports resolve. Follow this
  pattern for new scripts.
- **Adding a download site**: add a row to `downloader/sites.csv` (set `enabled=true`),
  create `downloader/<type>_downloader.py` subclassing `BaseDownloader`, and register the
  type in `DOWNLOADER_REGISTRY` in `downloader/manager.py`.
- **Ollama LLM**: `rag/generator.py` reads `OLLAMA_URL` (default
  `http://10.0.0.20:11434/api/generate`, model `qwen2.5-coder:7b`). It uses a streaming
  request and a generous read timeout for slow/CPU inference.
- **RTL UI**: keep `direction: rtl` on input/output elements in `api/app.py` and
  `api/static/app.js` for Hebrew content.

## Further docs

- `README.md` — feature list and quickstart.
- `DEPLOY.md` — Docker deployment and troubleshooting.
- `.github/copilot-instructions.md` — additional notes. **Note:** its embedding-model
  line is outdated; `config.py` (multilingual-e5-large) is the source of truth.
