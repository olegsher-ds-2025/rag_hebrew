from processing.cleaner import clean_text


def test_preserves_hebrew():
    assert clean_text('שלום עולם') == 'שלום עולם'


def test_preserves_gershayim_units():
    # Regression: ASCII quotes in Hebrew abbreviations must survive cleaning —
    # they carry the units the pipeline exists to answer about.
    assert 'מ"ר' in clean_text('שטח המגרש 500 מ"ר')
    assert "ג'" in clean_text("רחוב ג'")


def test_preserves_identifiers():
    assert '306/02/6' in clean_text('תכנית 306/02/6 אושרה')
    assert '504-0100552' in clean_text('מספר 504-0100552')


def test_preserves_structured_fields_and_percent():
    out = clean_text('שם התכנית: נוף הפארק (שלב ב) 25%')
    assert 'שם התכנית:' in out and '(שלב ב)' in out and '25%' in out


def test_strips_junk_symbols():
    out = clean_text('טקסט @#$ עם ~ רעש')
    assert '@' not in out and '#' not in out and '~' not in out
    assert 'טקסט' in out and 'רעש' in out


def test_preserves_paragraph_breaks():
    # Regression: newlines used to be collapsed to spaces, which disabled the
    # chunker's paragraph-boundary splitting entirely.
    out = clean_text('פסקה ראשונה\n\nפסקה שניה')
    assert '\n\n' in out


def test_collapses_horizontal_whitespace():
    assert clean_text('מילה    \t מילה') == 'מילה מילה'


def test_caps_blank_line_runs():
    assert '\n\n\n' not in clean_text('א\n\n\n\n\nב')
