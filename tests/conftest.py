"""
Shared fixtures. Tests never load the real embedding model (torch) or OCR
stack — FakeEmbedder produces deterministic unit vectors, and pipeline tests
run against tmp_path-backed store/index dirs.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbedder:
    """Deterministic, normalized embeddings with a crude similarity property:
    identical texts map to identical vectors (hash-based)."""

    dim = 32

    def encode(self, texts, is_query=False):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode('utf-8')).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype('float32')
            out.append(v / np.linalg.norm(v))
        return np.array(out)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def pipeline(fake_embedder, tmp_path):
    from rag.pipeline import RAGPipeline
    return RAGPipeline(
        embedder=fake_embedder,
        store_dir=str(tmp_path / 'vector_store'),
        index_dir=str(tmp_path / 'indexdir'),
    )
