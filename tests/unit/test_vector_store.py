import json

import pytest

from storage.vector_store import StoreCorruptError, VectorStore


def test_add_search_roundtrip(fake_embedder):
    vs = VectorStore()
    docs = ['מסמך ראשון', 'מסמך שני', 'מסמך שלישי']
    vs.add(fake_embedder.encode(docs), docs)
    hits = vs.search(fake_embedder.encode(['מסמך שני'])[0], k=1)
    assert hits == ['מסמך שני']


def test_dim_inferred_from_first_add(fake_embedder):
    vs = VectorStore()
    assert vs.index is None
    vs.add(fake_embedder.encode(['א']), ['א'])
    assert vs.index.d == fake_embedder.dim


def test_search_empty_store_returns_empty(fake_embedder):
    assert VectorStore().search(fake_embedder.encode(['ש'])[0]) == []


def test_save_load_roundtrip(fake_embedder, tmp_path):
    vs = VectorStore()
    docs = ['אחד', 'שניים']
    vs.add(fake_embedder.encode(docs), docs)
    vs.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert loaded.texts == docs
    assert loaded.index.ntotal == 2
    assert loaded.search(fake_embedder.encode(['אחד'])[0], k=1) == ['אחד']


def test_save_leaves_no_tmp_files(fake_embedder, tmp_path):
    vs = VectorStore()
    vs.add(fake_embedder.encode(['א']), ['א'])
    vs.save(tmp_path)
    assert not list(tmp_path.glob('*.tmp'))
    assert (tmp_path / 'meta.json').exists()


def test_load_refuses_desynced_store(fake_embedder, tmp_path):
    vs = VectorStore()
    docs = ['אחד', 'שניים', 'שלושה']
    vs.add(fake_embedder.encode(docs), docs)
    vs.save(tmp_path)
    # simulate a partial write: texts.json lost an entry
    (tmp_path / 'texts.json').write_text(json.dumps(docs[:2]), encoding='utf-8')
    with pytest.raises(StoreCorruptError, match='inconsistent'):
        VectorStore.load(tmp_path)


def test_load_refuses_legacy_store_without_meta(fake_embedder, tmp_path):
    # Pre-v2 stores (L2 metric, no meta.json) must be rebuilt, not mis-ranked.
    vs = VectorStore()
    vs.add(fake_embedder.encode(['א']), ['א'])
    vs.save(tmp_path)
    (tmp_path / 'meta.json').unlink()
    with pytest.raises(StoreCorruptError, match='Rebuild'):
        VectorStore.load(tmp_path)


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / 'nope')


def test_guarded_search_on_manually_desynced_instance(fake_embedder):
    # Even if an in-memory instance desyncs, search must not IndexError.
    vs = VectorStore()
    docs = ['אחד', 'שניים', 'שלושה']
    vs.add(fake_embedder.encode(docs), docs)
    vs.texts = vs.texts[:1]
    hits = vs.search(fake_embedder.encode(['שלושה'])[0], k=3)
    assert all(h == 'אחד' for h in hits)
