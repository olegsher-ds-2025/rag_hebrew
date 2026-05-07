from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser, OrGroup, AndGroup
import os

# Common Hebrew planning term synonyms for query expansion
PLANNING_SYNONYMS: dict[str, list[str]] = {
    'פלוט':    ['מגרש', 'חלקה', 'תא שטח'],
    'פלות':    ['מגרשים', 'חלקות', 'תאי שטח'],
    'מגרש':    ['פלוט', 'חלקה', 'תא שטח'],
    'מגרשים':  ['פלות', 'חלקות', 'תאי שטח'],
    'חלקה':    ['פלוט', 'מגרש', 'תא שטח'],
    'חלקות':   ['פלות', 'מגרשים', 'תאי שטח'],
    'שטח':     ['גודל', 'מר', 'דונם'],
    'גודל':    ['שטח', 'מידות'],
    'בנין':    ['מבנה', 'בניין'],
    'קומות':   ['מפלסים', 'קומה'],
    'יחידות':  ['דירות', 'יחידות דיור'],
    'גובה':    ['גובה מבנה'],
}


class KeywordSearch:
    def __init__(self):
        schema = Schema(content=TEXT(stored=True))
        os.makedirs("indexdir", exist_ok=True)
        try:
            if os.listdir("indexdir"):
                self.ix = open_dir("indexdir")
            else:
                self.ix = create_in("indexdir", schema)
        except Exception:
            self.ix = create_in("indexdir", schema)

    def add_docs(self, texts: list[str]) -> None:
        writer = self.ix.writer()
        for t in texts:
            writer.add_document(content=t)
        writer.commit()

    def _expand(self, query_str: str) -> str:
        """Expand query terms with Hebrew planning synonyms."""
        words = query_str.split()
        expanded = []
        for w in words:
            syns = PLANNING_SYNONYMS.get(w)
            if syns:
                expanded.append('(' + ' OR '.join([w] + syns) + ')')
            else:
                expanded.append(w)
        return ' '.join(expanded)

    def search(self, query_str: str, top_k: int = 10) -> list[str]:
        expanded = self._expand(query_str)
        with self.ix.searcher() as searcher:
            # Try AND-group first for higher precision
            try:
                q = QueryParser("content", schema=self.ix.schema, group=AndGroup).parse(expanded)
                results = searcher.search(q, limit=top_k)
                if len(results) > 0:
                    return [r['content'] for r in results]
            except Exception:
                pass
            # Fall back to OR-group for recall
            try:
                q = QueryParser("content", schema=self.ix.schema, group=OrGroup).parse(expanded)
                results = searcher.search(q, limit=top_k)
                return [r['content'] for r in results]
            except Exception:
                return []
