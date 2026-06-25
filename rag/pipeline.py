from embeddings.embedder import Embedder
from storage.vector_store import VectorStore
from retrieval.hybrid_search import KeywordSearch
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.ocr_pipeline import ocr_image
from processing.cleaner import clean_text
from processing.chunker import chunk_text
from rag.generator import generate_answer
from config import *
from pathlib import Path
import logging
import re
import os

logger = logging.getLogger(__name__)

# Hebrew planning synonyms injected into query before keyword search
_QUERY_EXPANSIONS: dict[str, list[str]] = {
    'פלות':   ['מגרשים', 'חלקות', 'תאי שטח', 'תא שטח'],
    'פלוט':   ['מגרש', 'חלקה', 'תא שטח'],
    'מגרשים': ['פלות', 'תאי שטח', 'חלקות'],
    'מגרש':   ['פלוט', 'תא שטח', 'חלקה'],
    'חלקות':  ['פלות', 'מגרשים', 'תאי שטח'],
    'חלקה':   ['פלוט', 'מגרש', 'תא שטח'],
    'שטח':    ['גודל', 'מידות', 'גודל מגרש'],
    'גודל':   ['שטח'],
}


def _expand_question(question: str) -> str:
    """Append synonym terms to the query for better keyword recall."""
    words = question.split()
    extras: list[str] = []
    for w in words:
        syns = _QUERY_EXPANSIONS.get(w)
        if syns:
            extras.extend(syns)
    return question + (' ' + ' '.join(extras) if extras else '')


_PLOT_TERMS = {'תא', 'שטח', 'תאי', 'מגרש', 'מגרשים', 'חלקה', 'חלקות', 'פלוט', 'פלות'}

# Matches: <plot-term> <number>  or  <number> <plot-term>
_PLOT_NUMBER_RE = re.compile(
    r'(?:תא\s*שטח|מגרש|חלקה|תאי\s*שטח|מגרשים|חלקות)\s*(\d+)'
    r'|(\d+)\s*(?:תא\s*שטח|מגרש|חלקה)'
)
# Matches: גודל <plot-term> <number>  — the most direct "size of plot N" statement
_PLOT_SIZE_RE = re.compile(
    r'גודל\s*(?:תא\s*שטח|מגרש|חלקה|מגרשים|תאי\s*שטח)\s*(\d+)'
    r'|(?:תא\s*שטח|מגרש|חלקה)\s*(\d+)[^\d]{1,20}(\d{4,5})\s*מר'
)


def _rerank(question: str, chunks: list[str]) -> list[str]:
    """
    Re-rank retrieved chunks by exact overlap with query tokens.

    Scoring (additive):
    - +5  per query number found anywhere in chunk
    - +8  bonus when a query number appears adjacent to a plot keyword —
          capped at ONE bonus per distinct number (avoids over-rewarding chunks
          that mention the same plot in multiple sentences)
    - +12 bonus when a "גודל <plot-term> N" or "<plot-term> N ... AREA מר"
          pattern is found (direct size statement — highest signal)
    - +1  per query word found in chunk
    """
    tokens   = re.findall(r'\d+|[\u0590-\u05FF]+', question)
    numbers  = {t for t in tokens if t.isdigit()}
    words    = {t for t in tokens if not t.isdigit()}

    def score(chunk: str) -> int:
        s = 0

        # +50 boost for metadata chunks (סטטוס וזכויות בנייה) that contain a query number.
        # Metadata is a direct answer for plan-number lookups.
        is_metadata = "סטטוס:" in chunk or "שם התכנית:" in chunk
        if is_metadata and any(n in chunk for n in numbers):
            s += 50

        s += sum(5 for n in numbers if n in chunk)
        s += sum(1 for w in words if w in chunk)

        # +8 for plot-adjacent number match (capped per number)
        seen = set()
        for m in _PLOT_NUMBER_RE.finditer(chunk):
            num = m.group(1) or m.group(2)
            if num in numbers and num not in seen:
                s += 8
                seen.add(num)

        # +12 for explicit size statement involving a query number
        for m in _PLOT_SIZE_RE.finditer(chunk):
            num = m.group(1) or m.group(2)
            if num in numbers:
                s += 12

        return s

    indexed = sorted(enumerate(chunks), key=lambda iv: score(iv[1]), reverse=True)
    return [c for _, c in indexed]


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

        return self._persist_chunks(all_chunks)

    def _persist_chunks(self, chunks: list[str]) -> int:
        """Append prepared chunks to the FAISS + Whoosh stores and persist."""
        if not chunks:
            return 0

        # Append to FAISS store (create new store if none exists yet)
        embeddings = self.embedder.encode(chunks)
        if self.vector_store is None:
            self.vector_store = VectorStore()
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save('vector_store')

        # Append to Whoosh keyword index
        self.keyword_search.add_docs(chunks)

        return len(chunks)

    def index_texts(self, chunks: list[str]) -> int:
        """
        Index already-prepared text chunks directly, skipping file extraction.

        Used for plan metadata pulled from the ArcGIS service (status, area,
        approval date, etc.). Chunks should already carry their [prefix].

        Returns:
            Number of chunks added.
        """
        return self._persist_chunks(chunks)

    def query(self, question: str) -> list[str]:
        # Embed with query prefix (E5 convention) for vector search
        q_emb = self.embedder.encode([question], is_query=True)
        vector_results = []
        if self.vector_store is not None:
            vector_results = self.vector_store.search(q_emb, TOP_K)

        # Expand question with synonyms for keyword search
        expanded = _expand_question(question)
        keyword_results = self.keyword_search.search(expanded, top_k=TOP_K)

        # Merge: vector results first (semantic relevance), then keyword hits
        return list(dict.fromkeys(vector_results + keyword_results))

    def query_with_answer(self, question: str) -> dict:
        """
        Retrieve relevant chunks and synthesise a concise Hebrew answer via LLM.

        Returns:
            {
                "answer": str,          # LLM-generated Hebrew answer, or "אין תשובה" if no relevant data
                "chunks": list[str],    # raw retrieved chunks with [filename] prefixes
            }
        """
        from rag.generator import _is_no_answer_response

        chunks = self.query(question)
        # Rerank using synonym-expanded question so "שטח" also matches "גודל"
        # and "פלות 22" also matches chunks with "תא שטח 22" / "מגרש 22"
        expanded = _expand_question(question)
        reranked = _rerank(expanded, chunks)

        # If question asks about a specific plan number, filter to only that plan.
        # This avoids showing unrelated plans in the sources.
        plan_nums = re.findall(r'\b(\d{3}-\d{7})\b', question)
        if plan_nums:
            plan_num = plan_nums[0]  # use the first mentioned plan
            reranked = [c for c in reranked if plan_num in c]

        answer = generate_answer(question, reranked)

        # If the LLM explicitly said it has no relevant data, suppress the answer
        # and return "אין תשובה" so the UI does not show speculative content.
        if _is_no_answer_response(answer):
            answer = "אין תשובה"
        else:
            # Safety nets: fix obvious extraction mistakes
            from rag.generator import _fix_area_number, _fix_plan_name
            answer = _fix_area_number(answer, question, reranked)
            answer = _fix_plan_name(answer, question, reranked)

        return {"answer": answer, "chunks": reranked}
