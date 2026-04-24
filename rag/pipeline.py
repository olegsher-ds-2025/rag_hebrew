from embeddings.embedder import Embedder
from storage.vector_store import VectorStore
from retrieval.hybrid_search import KeywordSearch
from config import *
from pathlib import Path
import os


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder(EMBEDDING_MODEL)
        self.vector_store = None
        self.keyword_search = KeywordSearch()

        # try to load persisted vector store if present
        vs_dir = Path('vector_store')
        try:
            if vs_dir.exists() and (vs_dir / 'index.faiss').exists():
                self.vector_store = VectorStore.load(vs_dir)
        except Exception:
            # ignore load errors and start without vector store
            self.vector_store = None

    def build_index(self, chunks):
        embeddings = self.embedder.encode(chunks)

        self.vector_store = VectorStore()
        self.vector_store.add(embeddings, chunks)
        # persist vector store
        self.vector_store.save('vector_store')

        self.keyword_search.add_docs(chunks)

    def query(self, question):
        q_emb = self.embedder.encode([question])
        vector_results = []
        if self.vector_store is not None:
            vector_results = self.vector_store.search(q_emb, TOP_K)
        keyword_results = self.keyword_search.search(question)

        return list(dict.fromkeys(vector_results + keyword_results))