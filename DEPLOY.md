Deployment (Docker)

Prerequisites
- Docker and docker-compose installed on your Ubuntu machine
- Internet access (for model download on first build)

Build and run with Docker

1. Build the image:

```
docker build -t rag_hebrew:latest .
```

2. Run the container (bind port 8000):

```
docker run -it --rm -p 8000:8000 \
  -v "$PWD/data":/app/data \
  -v "$PWD/vector_store":/app/vector_store \
  rag_hebrew:latest
```

This exposes the FastAPI app on port 8000.

Using docker-compose (recommended for dev):

```
docker-compose up --build -d
```

Notes and recommendations
- The Docker image pre-installs Tesseract (including Hebrew language) and pre-downloads the sentence-transformers model during build. The first build can take several minutes.
- Persisted `vector_store` directory is mounted as a volume so the FAISS index and texts survive container restarts.
- For production, consider using Gunicorn with multiple workers and an nginx reverse-proxy, and mounting `vector_store` on durable storage.
- If you use a cloud host (DigitalOcean, AWS, Render, Fly.io), deploy the built image there. For cloud providers that use containers, use the `docker build` output or push to a registry.

Troubleshooting
- If the image build fails installing some binary wheels (faiss-cpu etc.), try building on a matching platform (x86_64) and ensure you have enough memory. Using `faiss-cpu` wheel from PyPI should work for many Linux distributions.
- If Tesseract needs additional language packs, install them via apt (`tesseract-ocr-heb`) or configure accordingly.
