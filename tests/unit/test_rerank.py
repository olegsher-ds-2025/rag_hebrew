from rag.pipeline import _contains_number, _rerank


def test_contains_number_exact():
    assert _contains_number('22', 'מגרש 22 גדול')
    assert _contains_number('22', 'גודל מגרש 22')


def test_contains_number_rejects_substrings():
    # Regression: '22' used to match inside 220 / 1220 / 5.22
    assert not _contains_number('22', 'מגרש 220')
    assert not _contains_number('22', 'חלקה 1220')
    assert not _contains_number('5', 'שטח 500 מר')


def test_plot_number_chunk_ranks_first():
    chunks = [
        '[a.pdf] מגרש 220 בשטח 100 מר',
        '[b.pdf] מגרש 22 בשטח 500 מר',
        '[c.pdf] טקסט כללי ללא מספרים',
    ]
    reranked = _rerank('שטח מגרש 22', chunks)
    assert reranked[0] == '[b.pdf] מגרש 22 בשטח 500 מר'


def test_size_statement_beats_plain_mention():
    chunks = [
        '[a.pdf] ראו מגרש 22 בתשריט',
        '[b.pdf] גודל מגרש 22 הוא 500 מר',
    ]
    reranked = _rerank('גודל מגרש 22', chunks)
    assert reranked[0].startswith('[b.pdf]')


def test_stable_order_on_ties():
    chunks = ['[a.pdf] אחד', '[b.pdf] שתיים', '[c.pdf] שלוש']
    assert _rerank('שאלה ללא חפיפה', chunks) == chunks
