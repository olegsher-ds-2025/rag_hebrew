from rag.pipeline import _reciprocal_rank_fusion


def test_agreement_wins():
    # B is rank-2 in both lists; A and D are rank-1 in only one list each.
    # RRF: B = 1/62 + 1/62 = 0.03226 beats A = D = 1/61 = 0.01639.
    vector = ['A', 'B', 'C']
    keyword = ['D', 'B', 'E']
    fused = _reciprocal_rank_fusion([vector, keyword], k=60)
    assert fused[0] == 'B'


def test_output_is_deduplicated():
    fused = _reciprocal_rank_fusion([['A', 'B'], ['B', 'A']], k=60)
    assert sorted(fused) == ['A', 'B']
    assert len(fused) == 2


def test_empty_lists_return_empty():
    assert _reciprocal_rank_fusion([[], []]) == []
    assert _reciprocal_rank_fusion([]) == []


def test_one_empty_list_preserves_other_order():
    assert _reciprocal_rank_fusion([['A', 'B', 'C'], []]) == ['A', 'B', 'C']
    assert _reciprocal_rank_fusion([[], ['A', 'B', 'C']]) == ['A', 'B', 'C']


def test_single_list_preserves_order():
    assert _reciprocal_rank_fusion([['X', 'Y', 'Z']]) == ['X', 'Y', 'Z']


def test_low_ranked_only_item_still_appears():
    # 'E' is last in the keyword list only — must still be included.
    fused = _reciprocal_rank_fusion([['A', 'B'], ['C', 'D', 'E']])
    assert 'E' in fused


def test_ties_keep_first_seen_order():
    # No overlap → every item scores by its rank; equal ranks across the two
    # lists tie, and the earlier (first) list must win the tie.
    fused = _reciprocal_rank_fusion([['A', 'B'], ['C', 'D']], k=60)
    # rank-1: A (list0) and C (list1) tie → A first. rank-2: B then D.
    assert fused == ['A', 'C', 'B', 'D']


def test_query_fuses_vector_and_keyword(pipeline):
    # 'shared' is retrievable by BOTH vector and keyword; a vector-only chunk
    # should not outrank it once RRF rewards the agreement.
    pipeline.build_index([
        '[a.pdf] מגרש 22 בשטח 500 מר',       # matches 'מגרש 22' both ways
        '[b.pdf] מגרש 22 טקסט נוסף',          # also matches
        '[c.pdf] נתונים כלליים אחרים',        # unrelated
    ])
    results = pipeline.query('מגרש 22')
    assert results
    assert any('מגרש 22' in r for r in results[:2])
