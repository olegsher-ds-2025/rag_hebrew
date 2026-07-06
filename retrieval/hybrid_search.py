import os

from whoosh.fields import TEXT, Schema
from whoosh.index import create_in, open_dir
from whoosh.qparser import AndGroup, OrGroup, QueryParser

from processing.synonyms import PLANNING_SYNONYMS


def _as_query_term(term: str) -> str:
    """Quote multi-word synonyms so Whoosh parses them as phrases, not two loose tokens."""
    return f'"{term}"' if ' ' in term else term


class KeywordSearch:
    def __init__(self, index_dir: str = "indexdir"):
        self.index_dir = index_dir
        self.schema = Schema(content=TEXT(stored=True))
        os.makedirs(index_dir, exist_ok=True)
        try:
            if os.listdir(index_dir):
                self.ix = open_dir(index_dir)
            else:
                self.ix = create_in(index_dir, self.schema)
        except Exception:
            self.ix = create_in(index_dir, self.schema)

    def add_docs(self, texts: list[str]) -> None:
        writer = self.ix.writer()
        for t in texts:
            writer.add_document(content=t)
        writer.commit()

    def clear(self) -> None:
        """Drop all indexed documents (used by full rebuilds to stay in sync with FAISS)."""
        self.ix = create_in(self.index_dir, self.schema)

    def doc_count(self) -> int:
        return self.ix.doc_count()

    def _expand(self, query_str: str) -> str:
        """Expand query terms with Hebrew planning synonyms."""
        words = query_str.split()
        expanded = []
        for w in words:
            syns = PLANNING_SYNONYMS.get(w)
            if syns:
                terms = [_as_query_term(t) for t in [w] + syns]
                expanded.append('(' + ' OR '.join(terms) + ')')
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
