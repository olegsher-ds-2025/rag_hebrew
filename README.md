# Hebrew RAG

RAG pipeline for Israeli planning & building documents (Hebrew), with hybrid
FAISS + BM25 retrieval, llama.cpp answer synthesis, and a FastAPI web UI.

## Features
- PDF + OCR ingestion (Tesseract, Hebrew)
- Hebrew text cleaning and sentence-aware chunking
- Multilingual embeddings (`intfloat/multilingual-e5-large`)
- FAISS vector search + Whoosh BM25 keyword search (hybrid, re-ranked)
- LLM answer synthesis via llama.cpp (`llama-server`)
- Document downloader for mavat.iplan.gov.il (by plan name/number)
- FastAPI server with RTL Hebrew web UI (port 9000)

## Run

```bash
# Requires system packages: tesseract-ocr, tesseract-ocr-heb
pip install -r requirements.txt

# Index all PDFs/images from data/raw/
python scripts/build_index_all.py

# Start the API + web UI at http://localhost:9000
uvicorn api.app:app --host 0.0.0.0 --port 9000
```

For deployment on NVIDIA Jetson (Docker + host llama-server), see
[DEPLOY.md](DEPLOY.md). For codebase guidance, see [CLAUDE.md](CLAUDE.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
