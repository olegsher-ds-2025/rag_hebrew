"""
Hebrew planning-domain synonyms, shared by keyword-search expansion
(retrieval/hybrid_search.py) and rerank query expansion (rag/pipeline.py).

Single source of truth — the two consumers previously kept separate,
diverging copies of this table.
"""

PLANNING_SYNONYMS: dict[str, list[str]] = {
    'פלוט':    ['מגרש', 'חלקה', 'תא שטח'],
    'פלות':    ['מגרשים', 'חלקות', 'תאי שטח', 'תא שטח'],
    'מגרש':    ['פלוט', 'חלקה', 'תא שטח'],
    'מגרשים':  ['פלות', 'חלקות', 'תאי שטח'],
    'חלקה':    ['פלוט', 'מגרש', 'תא שטח'],
    'חלקות':   ['פלות', 'מגרשים', 'תאי שטח'],
    'שטח':     ['גודל', 'מידות', 'מר', 'דונם'],
    'גודל':    ['שטח', 'מידות'],
    'בנין':    ['מבנה', 'בניין'],
    'קומות':   ['מפלסים', 'קומה'],
    'יחידות':  ['דירות', 'יחידות דיור'],
    'גובה':    ['גובה מבנה'],
}


def expand_words(question: str) -> list[str]:
    """Synonym terms for each word of the question (flat list, may be empty)."""
    extras: list[str] = []
    for w in question.split():
        extras.extend(PLANNING_SYNONYMS.get(w, []))
    return extras
