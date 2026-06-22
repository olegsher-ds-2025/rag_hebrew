# Deployment on NVIDIA Jetson Orin Nano

This branch runs the RAG app in a container on the Jetson, while the LLM is
served by **llama.cpp (`llama-server`) running directly on the Jetson host** so
it can use the GPU. The container reaches the server over HTTP.

```
┌────────────────────────── Jetson Orin Nano (host) ──────────────────────────┐
│                                                                              │
│   llama-server  ──/completion──►  (GPU inference)        port 8080           │
│        ▲                                                                     │
│        │ HTTP (host.docker.internal:8080)                                    │
│   ┌────┴───────────────── Docker container ──────────────────┐              │
│   │  FastAPI app + embeddings (CPU) + FAISS/BM25   port 8000 │              │
│   └──────────────────────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- JetPack with Docker and the `docker compose` plugin installed.
- llama.cpp already built on the Jetson (with CUDA), and a GGUF model file.

## 1. Start llama-server on the host

Run the server on the host (not in the container) so it has GPU access. Bind it
to all interfaces so the container can reach it:

```bash
# Adjust the path to your llama.cpp build and GGUF model.
./llama-server \
  -m /path/to/your-model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999          # offload all layers to the Jetson GPU
```

Verify it responds:

```bash
curl -s http://localhost:8080/health
```

## 2. Build and run the container

```bash
docker compose up --build -d
```

The first build downloads the embedding model (`intfloat/multilingual-e5-large`)
and installs Tesseract, so it can take several minutes. Subsequent starts are
fast.

The app is served at `http://<jetson-ip>:8000`.

## How the container reaches llama-server

`docker-compose.yml` maps `host.docker.internal` to the host gateway and sets:

```yaml
environment:
  - LLAMACPP_URL=http://host.docker.internal:8080/completion
```

If your llama-server listens on a different host/port, override `LLAMACPP_URL`
(and, if needed, `LLAMACPP_N_PREDICT` / `LLAMACPP_TEMPERATURE`) in
`docker-compose.yml`. The app uses llama.cpp's **native** `/completion`
streaming endpoint, not the OpenAI-compatible `/v1` one.

## Indexing documents

Persisted indices live in mounted volumes (`vector_store/`, `indexdir/`) so they
survive container restarts. Build them inside the container:

```bash
# Place PDFs/images under ./data/raw first, then:
docker compose exec app python scripts/build_index_all.py
```

## Troubleshooting

- **App returns empty answers / logs `[llama.cpp] Connection failed`:** confirm
  `llama-server` is running and was started with `--host 0.0.0.0`. From inside
  the container, test with
  `docker compose exec app python -c "import requests; print(requests.get('http://host.docker.internal:8080/health').text)"`.
- **`host.docker.internal` not resolving:** requires Docker 20.10+. The
  `extra_hosts: host-gateway` entry in `docker-compose.yml` provides it; as a
  fallback, set `LLAMACPP_URL` to the Jetson's LAN IP, e.g.
  `http://192.168.x.x:8080/completion`.
- **Embeddings are slow / OOM:** `multilingual-e5-large` runs on CPU in the
  container and is memory-hungry. On a 4 GB Orin Nano, ensure swap is enabled,
  or switch `EMBEDDING_MODEL` in `config.py` to a smaller multilingual model.
- **Build fails on torch/faiss wheels:** build natively on the Jetson (arm64);
  ensure enough RAM/swap during `pip install`.
