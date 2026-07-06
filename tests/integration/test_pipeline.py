
import rag.pipeline as pipeline_mod
from rag.pipeline import RAGPipeline


def test_build_and_query_roundtrip(pipeline):
    chunks = ['[x.pdf] מגרש 22 בשטח 500 מר', '[x.pdf] פרטים על הבניין']
    pipeline.build_index(chunks)
    results = pipeline.query('מגרש 22')
    assert any('מגרש 22' in c for c in results)


def test_rebuild_twice_keeps_stores_in_sync(pipeline):
    # Regression: rebuilds used to recreate FAISS but APPEND to Whoosh,
    # duplicating keyword docs and desyncing the two stores.
    chunks = ['[x.pdf] קטע ראשון', '[x.pdf] קטע שני']
    pipeline.build_index(chunks)
    pipeline.build_index(chunks)
    assert pipeline.keyword_search.doc_count() == 2
    assert pipeline.vector_store.index.ntotal == 2


def test_index_texts_appends(pipeline):
    pipeline.build_index(['[x.pdf] בסיס'])
    added = pipeline.index_texts(['[504] תוכנית 504-0100552 סטטוס: מאושרת'])
    assert added == 1
    assert pipeline.vector_store.index.ntotal == 2
    assert pipeline.keyword_search.doc_count() == 2


def test_persisted_store_reloads(pipeline, fake_embedder, tmp_path):
    pipeline.build_index(['[x.pdf] תוכן קבוע'])
    reloaded = RAGPipeline(
        embedder=fake_embedder,
        store_dir=str(pipeline.store_dir),
        index_dir=pipeline.keyword_search.index_dir,
    )
    assert reloaded.vector_store is not None
    assert reloaded.vector_store.texts == ['[x.pdf] תוכן קבוע']


def test_index_new_files_manifest_skips_unchanged(pipeline, tmp_path, monkeypatch):
    doc = tmp_path / 'plan.pdf'
    doc.write_bytes(b'%PDF fake')

    monkeypatch.setattr(
        RAGPipeline, '_extract_file_chunks',
        lambda self, fp: [f'[{fp.name}] תוכן המסמך'],
    )

    assert pipeline.index_new_files([doc]) == 1
    # Regression: re-downloading the same plan used to re-append its chunks.
    assert pipeline.index_new_files([doc]) == 0
    assert pipeline.vector_store.index.ntotal == 1

    # a changed file is re-indexed
    doc.write_bytes(b'%PDF fake but longer than before')
    assert pipeline.index_new_files([doc]) == 1


def test_index_new_files_survives_bad_file(pipeline, tmp_path, monkeypatch):
    good = tmp_path / 'good.pdf'
    good.write_bytes(b'%PDF')
    bad = tmp_path / 'missing.pdf'  # never created → stat() fails

    monkeypatch.setattr(
        RAGPipeline, '_extract_file_chunks',
        lambda self, fp: [f'[{fp.name}] תוכן'],
    )
    assert pipeline.index_new_files([bad, good]) == 1


def test_query_with_answer_llm_unavailable(pipeline, monkeypatch):
    pipeline.build_index(['[x.pdf] מגרש 22 בשטח 500 מר'])
    monkeypatch.setattr(pipeline_mod, 'generate_answer', lambda q, c: None)
    out = pipeline.query_with_answer('מה שטח מגרש 22')
    assert 'אינו זמין' in out['answer']
    assert out['chunks']


def test_query_with_answer_no_data_suppressed(pipeline, monkeypatch):
    pipeline.build_index(['[x.pdf] מגרש 22'])
    monkeypatch.setattr(pipeline_mod, 'generate_answer', lambda q, c: 'אין מידע בתכניות')
    out = pipeline.query_with_answer('שאלה על מגרש 22')
    assert out['answer'] == 'אין תשובה'


def test_query_with_answer_plan_number_filter(pipeline, monkeypatch):
    pipeline.build_index([
        '[a.pdf] תוכנית 504-0100552 שטח 500',
        '[b.pdf] תוכנית 606-0200311 שטח 900',
    ])
    monkeypatch.setattr(pipeline_mod, 'generate_answer', lambda q, c: 'תשובה')
    out = pipeline.query_with_answer('מה שטח תוכנית 504-0100552?')
    assert out['chunks']
    assert all('504-0100552' in c for c in out['chunks'])


def test_empty_index_query_returns_empty(pipeline):
    assert pipeline.query('שאלה כלשהי') == []
