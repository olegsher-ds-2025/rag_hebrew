import pytest

from processing.chunker import chunk_text


def test_empty_text():
    assert chunk_text('') == []
    assert chunk_text('   ') == []


def test_short_text_single_chunk():
    assert chunk_text('משפט קצר.', chunk_size=100, overlap=10) == ['משפט קצר.']


def test_sentences_grouped_within_chunk_size():
    text = 'משפט ראשון. משפט שני. משפט שלישי.'
    chunks = chunk_text(text, chunk_size=25, overlap=5)
    assert len(chunks) >= 2
    # nothing lost
    joined = ' '.join(chunks)
    for word in ['ראשון', 'שני', 'שלישי']:
        assert word in joined


def test_oversized_sentence_char_split():
    long_sentence = 'א' * 500  # no punctuation at all
    chunks = chunk_text(long_sentence, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) >= 500  # full coverage (overlap allowed)


def test_paragraph_boundaries_respected():
    text = 'פסקה ראשונה כאן\n\nפסקה שניה כאן'
    chunks = chunk_text(text, chunk_size=20, overlap=0)
    assert any('ראשונה' in c for c in chunks)
    assert any('שניה' in c for c in chunks)


def test_page_markers_are_boundaries():
    text = 'עמוד ראשון --- PAGE 1 --- עמוד שני'
    chunks = chunk_text(text, chunk_size=15, overlap=0)
    assert not any('PAGE' in c for c in chunks)


@pytest.mark.parametrize('chunk_size,overlap', [(100, 100), (100, 150), (0, 0), (-5, 0), (100, -1)])
def test_invalid_params_raise(chunk_size, overlap):
    # Regression: overlap >= chunk_size used to crash (step 0) or silently
    # drop oversized sentences (negative step).
    with pytest.raises(ValueError):
        chunk_text('טקסט כלשהו', chunk_size=chunk_size, overlap=overlap)


def test_overlap_carry_over():
    text = '. '.join(['משפט מספר ' + str(i) for i in range(20)]) + '.'
    chunks = chunk_text(text, chunk_size=60, overlap=30)
    assert len(chunks) >= 2
    # consecutive chunks share some content (the overlap tail)
    tail_words = set(chunks[0].split()[-3:])
    assert tail_words & set(chunks[1].split())
