from embeddings.embedder import Embedder
from storage.vector_store import VectorStore
from retrieval.hybrid_search import KeywordSearch
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.ocr_pipeline import ocr_image
from processing.cleaner import clean_text
from processing.chunker import chunk_text
from config import *
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


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

    def index_new_files(self, file_paths: list) -> int:
        """
        Incrementally index new files without rebuilding the full index.

        Extracts text from each file, cleans and chunks it, then appends
        the new chunks to the existing FAISS vector store and Whoosh keyword
        index.  Both stores are persisted after the update.

        Args:
            file_paths: list of Path or str pointing to PDF/image files.

        Returns:
            Number of new chunks added.
        """
        all_chunks = []
        for fp in file_paths:
            fp = Path(fp)
            try:
                if fp.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf(str(fp))
                else:
                    text = ocr_image(str(fp))
                cleaned = clean_text(text)
                chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                prefixed = [f"[{fp.name}] " + c for c in chunks]
                all_chunks.extend(prefixed)
                logger.info("index_new_files: %s -> %d chunks", fp.name, len(chunks))
            except Exception as exc:
                logger.error("index_new_files: failed to process %s: %s", fp, exc)

        if not all_chunks:
            return 0

        # Append to FAISS store (create new store if none exists yet)
        embeddings = self.embedder.encode(all_chunks)
        if self.vector_store is None:
            self.vector_store = VectorStore()
        self.vector_store.add(embeddings, all_chunks)
        self.vector_store.save('vector_store')

        # Append to Whoosh keyword index
        self.keyword_search.add_docs(all_chunks)

        return len(all_chunks)

    def query(self, question):
        q_emb = self.embedder.encode([question])
        vector_results = []
        if self.vector_store is not None:
            vector_results = self.vector_store.search(q_emb, TOP_K)
        keyword_results = self.keyword_search.search(question)

        return list(dict.fromkeys(vector_results + keyword_results))