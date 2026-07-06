from processing.synonyms import PLANNING_SYNONYMS, expand_words


def test_known_word_expands():
    extras = expand_words('שטח המגרש')
    assert 'גודל' in extras


def test_unknown_words_no_expansion():
    assert expand_words('מילה לאיהקיימת') == []


def test_expansion_covers_plot_terms_both_ways():
    # The planning-domain core: פלוט/מגרש/חלקה/תא שטח must be mutual synonyms
    assert 'מגרש' in PLANNING_SYNONYMS['פלוט']
    assert 'פלוט' in PLANNING_SYNONYMS['מגרש']
    assert 'תא שטח' in PLANNING_SYNONYMS['חלקה']


def test_expand_question_appends():
    from rag.pipeline import _expand_question
    out = _expand_question('שטח פלוט 22')
    assert out.startswith('שטח פלוט 22')
    assert 'מגרש' in out and 'גודל' in out


def test_expand_question_no_synonyms_unchanged():
    from rag.pipeline import _expand_question
    assert _expand_question('שאלה כללית') == 'שאלה כללית'
