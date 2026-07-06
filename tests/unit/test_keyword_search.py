from retrieval.hybrid_search import KeywordSearch, _as_query_term


def make_ks(tmp_path):
    return KeywordSearch(index_dir=str(tmp_path / 'kw'))


def test_add_and_search_hebrew(tmp_path):
    ks = make_ks(tmp_path)
    ks.add_docs(['מגרש 22 בשטח 500 מר', 'בניין מגורים ברחוב הרצל'])
    hits = ks.search('מגרש 22')
    assert hits and 'מגרש 22' in hits[0]


def test_synonym_expansion_recall(tmp_path):
    ks = make_ks(tmp_path)
    ks.add_docs(['תא שטח 22 בגודל 500 מר'])
    # 'פלוט' itself never appears in the doc — only its synonym 'תא שטח'
    hits = ks.search('פלוט 22')
    assert hits, 'synonym expansion should recall the תא שטח chunk'


def test_multiword_synonyms_quoted():
    assert _as_query_term('תא שטח') == '"תא שטח"'
    assert _as_query_term('מגרש') == 'מגרש'


def test_and_falls_back_to_or(tmp_path):
    ks = make_ks(tmp_path)
    ks.add_docs(['מסמך על מגרשים בלבד'])
    # AND over both words fails (second word absent) → OR fallback still hits
    hits = ks.search('מגרשים לאקיים')
    assert hits


def test_clear_empties_index(tmp_path):
    ks = make_ks(tmp_path)
    ks.add_docs(['אחד', 'שניים'])
    assert ks.doc_count() == 2
    ks.clear()
    assert ks.doc_count() == 0
    assert ks.search('אחד') == []


def test_clear_then_add_no_duplicates(tmp_path):
    # Regression: full rebuilds used to append to the existing index,
    # duplicating every doc on each rebuild.
    ks = make_ks(tmp_path)
    docs = ['מסמך ראשון', 'מסמך שני']
    ks.add_docs(docs)
    ks.clear()
    ks.add_docs(docs)
    assert ks.doc_count() == 2


def test_reopens_persisted_index(tmp_path):
    ks = make_ks(tmp_path)
    ks.add_docs(['מסמך קבוע'])
    ks2 = make_ks(tmp_path)  # same dir → reopen
    assert ks2.doc_count() == 1
