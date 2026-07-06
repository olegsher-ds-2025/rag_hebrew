import pytest
from fastapi.testclient import TestClient

import api.app as appmod


class FakeKeywordSearch:
    @staticmethod
    def doc_count():
        return 3


class FakeRag:
    def __init__(self, chunks=None, answer='תשובה לדוגמה'):
        self._chunks = chunks or []
        self._answer = answer
        self.vector_store = None
        self.keyword_search = FakeKeywordSearch()

    def query_with_answer(self, q):
        return {'answer': self._answer, 'chunks': self._chunks}

    def index_texts(self, chunks):
        return len(chunks)

    def index_new_files(self, files):
        return 0


class FakeDownloadManager:
    def download(self, plan_name):
        return [], [f'חיפוש: {plan_name}'], ['[504] מטא-נתונים']


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, 'raw_dir', tmp_path)
    appmod.app.state.rag = FakeRag()
    appmod.app.state.download_manager = FakeDownloadManager()
    with TestClient(appmod.app) as c:
        yield c
    appmod.app.state.rag = None
    appmod.app.state.download_manager = None


def test_health_ok(client):
    body = client.get('/health').json()
    assert body['status'] == 'ok'
    assert body['keyword_docs'] == 3


def test_query_shape(client):
    appmod.app.state.rag = FakeRag(chunks=['[doc.pdf] תוכן הקטע'])
    r = client.post('/query', json={'q': 'שאלה'})
    assert r.status_code == 200
    body = r.json()
    assert body['answer'] == 'תשובה לדוגמה'
    assert body['results'][0]['text'] == 'תוכן הקטע'


def test_query_source_resolved_for_existing_file(client, tmp_path):
    (tmp_path / 'doc.pdf').write_bytes(b'%PDF')
    appmod.app.state.rag = FakeRag(chunks=['[doc.pdf] תוכן'])
    body = client.post('/query', json={'q': 'ש'}).json()
    assert body['results'][0]['source'] == '/files/doc.pdf'


@pytest.mark.parametrize('prefix', [
    'missing.pdf',            # not on disk
    '../secret.pdf',          # traversal-shaped
    'x" onmouseover="a.pdf',  # XSS-shaped (not on disk)
    '..',
])
def test_query_source_rejected_for_bad_prefixes(client, prefix):
    appmod.app.state.rag = FakeRag(chunks=[f'[{prefix}] תוכן'])
    body = client.post('/query', json={'q': 'ש'}).json()
    assert body['results'][0]['source'] is None


def test_query_get_removed(client):
    assert client.get('/query?q=x').status_code == 405


def test_query_size_limit(client):
    assert client.post('/query', json={'q': 'x' * 2001}).status_code == 422
    assert client.post('/query', json={'q': ''}).status_code == 422


def test_download_indexes_metadata(client):
    body = client.post('/download', json={'plan_name': 'נוף הפארק'}).json()
    assert body['indexed_metadata'] == 1
    assert body['downloaded'] == []
    assert any('נוף הפארק' in line for line in body['log'])


def test_download_busy_returns_409(client):
    assert appmod._download_lock.acquire(blocking=False)
    try:
        assert client.post('/download', json={'plan_name': 'x'}).status_code == 409
    finally:
        appmod._download_lock.release()


def test_download_api_token(client, monkeypatch):
    monkeypatch.setattr(appmod, 'API_TOKEN', 'secret')
    assert client.post('/download', json={'plan_name': 'x'}).status_code == 401
    ok = client.post('/download', json={'plan_name': 'x'}, headers={'X-API-Token': 'secret'})
    assert ok.status_code == 200


def test_home_serves_html(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'RAG Query' in r.text
