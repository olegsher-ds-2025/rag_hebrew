from embeddings.embedder import Embedder
from storage.vector_store import VectorStore
from retrieval.hybrid_search import KeywordSearch
from config import *

class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder(EMBEDDING_MODEL)
        self.vector_store = None
        self.keyword_search = KeywordSearch()

    def build_index(self, chunks):
        embeddings = self.embedder.encode(chunks)
        dim = len(embeddings[0])

        self.vector_store = VectorStore(dim)
        self.vector_store.add(embeddings, chunks)

        self.keyword_search.add_docs(chunks)

    def query(self, question):
        q_emb = self.embedder.encode([question])
        vector_results = self.vector_store.search(q_emb, TOP_K)
        keyword_results = self.keyword_search.search(question)

        return list(set(vector_results + keyword_results))