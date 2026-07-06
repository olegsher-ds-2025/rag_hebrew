import json

import requests

import rag.generator as gen


class FakeResp:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self, chunk_size=None, decode_unicode=False):
        return iter(self._lines)


def sse(content, stop=False):
    payload = json.dumps({"content": content, "stop": stop}, ensure_ascii=False)
    return f'data: {payload}'.encode()


# ---------------------------------------------------------------- prompt

def test_build_prompt_strips_prefix_and_includes_question():
    p = gen._build_prompt('מה השטח?', ['[45.pdf] שטח המגרש 500 מ"ר'])
    assert '[45.pdf]' not in p
    assert 'שטח המגרש 500' in p
    assert 'מה השטח?' in p


def test_build_prompt_caps_context_per_chunk():
    huge = 'א' * 10000
    p = gen._build_prompt('שאלה', [f'[a.pdf] {huge}'] * gen.TOP_N_CHUNKS)
    # context body must respect the overall cap (allow template overhead)
    assert len(p) < gen.MAX_CONTEXT_CHARS + 1500


def test_build_prompt_uses_top_n_chunks_only():
    chunks = [f'[f.pdf] קטע{i}' for i in range(10)]
    p = gen._build_prompt('שאלה', chunks)
    assert 'קטע0' in p and f'קטע{gen.TOP_N_CHUNKS - 1}' in p
    assert f'קטע{gen.TOP_N_CHUNKS}' not in p


# ---------------------------------------------------------------- streaming

def test_generate_answer_parses_sse_stream(monkeypatch):
    resp = FakeResp([sse('שטח '), b'', sse('500 מר', stop=True)])
    monkeypatch.setattr(gen.requests, 'post', lambda *a, **k: resp)
    assert gen.generate_answer('מה השטח?', ['[a.pdf] קטע']) == 'שטח 500 מר'


def test_generate_answer_handles_plain_json_lines(monkeypatch):
    # Older llama-server builds stream bare JSON lines without the SSE prefix
    resp = FakeResp([b'{"content": "abc", "stop": true}'])
    monkeypatch.setattr(gen.requests, 'post', lambda *a, **k: resp)
    assert gen.generate_answer('q', ['[a.pdf] x']) == 'abc'


def test_generate_answer_skips_done_and_garbage(monkeypatch):
    resp = FakeResp([b'not json', sse('ok', stop=False), b'data: [DONE]'])
    monkeypatch.setattr(gen.requests, 'post', lambda *a, **k: resp)
    assert gen.generate_answer('q', ['[a.pdf] x']) == 'ok'


def test_no_chunks_returns_empty_not_none():
    assert gen.generate_answer('שאלה', []) == ''


# ---------------------------------------------------------------- errors → None

def test_connection_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError('refused')
    monkeypatch.setattr(gen.requests, 'post', boom)
    assert gen.generate_answer('q', ['[a.pdf] x']) is None


def test_timeout_returns_none(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.Timeout('slow')
    monkeypatch.setattr(gen.requests, 'post', boom)
    assert gen.generate_answer('q', ['[a.pdf] x']) is None


def test_http_error_returns_none(monkeypatch):
    class ErrResp(FakeResp):
        status_code = 500

        def raise_for_status(self):
            raise requests.exceptions.HTTPError('500')
    monkeypatch.setattr(gen.requests, 'post', lambda *a, **k: ErrResp([]))
    assert gen.generate_answer('q', ['[a.pdf] x']) is None


# ---------------------------------------------------------------- answer filters

def test_is_no_answer_response():
    assert gen._is_no_answer_response('')
    assert gen._is_no_answer_response('אין מידע בתכניות')
    assert gen._is_no_answer_response('הנתון לא נמצא במסמכים')
    assert not gen._is_no_answer_response('שטח המגרש 500 מ"ר')


def test_fix_plan_name_replaces_bare_number():
    # signature: (answer, question, chunks)
    chunks = ['[504] תוכנית 504-0100552\nשם התכנית: נוף הפארק - יובלים גנים']
    out = gen._fix_plan_name('תוכנית 504-0100552', 'מה שם התכנית?', chunks)
    assert out == 'נוף הפארק - יובלים גנים'
