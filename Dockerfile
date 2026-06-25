# Multi-arch base: Docker pulls the linux/arm64 variant automatically on the
# Jetson, so this image builds natively on the Orin Nano. The LLM runs outside
# the container via llama-server on the host; only embeddings/OCR run in here.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-heb \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model to image cache. This runs BEFORE copying the app
# code so that source changes don't invalidate this layer and re-download the
# ~2 GB model on every rebuild.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

# Copy project
COPY . /app

EXPOSE 9000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "9000"]
