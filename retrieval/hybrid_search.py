from whoosh.index import create_in
from whoosh.fields import Schema, TEXT
from whoosh.qparser import QueryParser
import os

class KeywordSearch:
    def __init__(self):
        schema = Schema(content=TEXT(stored=True))
        os.makedirs("indexdir", exist_ok=True)
        self.ix = create_in("indexdir", schema)

    def add_docs(self, texts):
        writer = self.ix.writer()
        for t in texts:
            writer.add_document(content=t)
        writer.commit()

    def search(self, query):
        with self.ix.searcher() as searcher:
            qp = QueryParser("content", schema=self.ix.schema)
            q = qp.parse(query)
            results = searcher.search(q)
            return [r['content'] for r in results]