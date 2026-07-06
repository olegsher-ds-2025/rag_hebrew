"""
Embedder tests without torch: a fake `sentence_transformers` module is injected
into sys.modules, so the lazy `from sentence_transformers import ...` inside the
model property picks it up. Verifies lazy loading, batch-size forwarding, e5
prefixing, and normalization — none of which need the real model.
"""

import sys
import types

import pytest

from embeddings.embedder import Embedder


class FakeST:
    instances = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.encode_calls = []
        FakeST.instances.append(self)

    def encode(self, texts, show_progress_bar=None, normalize_embeddings=None, batch_size=None):
        self.encode_calls.append({
            "texts": list(texts),
            "normalize_embeddings": normalize_embeddings,
            "batch_size": batch_size,
        })
        return [[0.0, 1.0] for _ in texts]


@pytest.fixture
def fake_st(monkeypatch):
    FakeST.instances = []
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return FakeST


def test_construction_does_not_load_model(fake_st):
    Embedder("intfloat/multilingual-e5-large")
    assert fake_st.instances == []   # lazy: nothing built yet


def test_warmup_loads_once(fake_st):
    emb = Embedder("intfloat/multilingual-e5-large")
    emb.warmup()
    emb.warmup()
    assert len(fake_st.instances) == 1
    assert fake_st.instances[0].model_name == "intfloat/multilingual-e5-large"


def test_encode_triggers_load_and_forwards_batch_size(fake_st):
    emb = Embedder("intfloat/multilingual-e5-large", batch_size=8)
    assert fake_st.instances == []
    emb.encode(["טקסט"])
    assert len(fake_st.instances) == 1
    call = fake_st.instances[0].encode_calls[0]
    assert call["batch_size"] == 8
    assert call["normalize_embeddings"] is True


def test_e5_prefixes_applied(fake_st):
    emb = Embedder("intfloat/multilingual-e5-large")
    emb.encode(["שאלה"], is_query=True)
    emb.encode(["מסמך"], is_query=False)
    calls = fake_st.instances[0].encode_calls
    assert calls[0]["texts"] == ["query: שאלה"]
    assert calls[1]["texts"] == ["passage: מסמך"]


def test_non_e5_model_gets_no_prefix(fake_st):
    emb = Embedder("some/minilm-model")
    emb.encode(["טקסט"], is_query=True)
    assert fake_st.instances[0].encode_calls[0]["texts"] == ["טקסט"]
