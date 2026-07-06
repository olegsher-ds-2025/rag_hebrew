import json
import logging
import re
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, TOP_K
from processing.chunker import chunk_text
from processing.cleaner import clean_text
from processing.synonyms import expand_words
from rag.generator import generate_answer
from retrieval.hybrid_search import KeywordSearch
from storage.vector_store import StoreCorruptError, VectorStore

logger = logging.getLogger(__name__)


def _expand_question(question: str) -> str:
    """Append synonym terms to the query for better keyword/rerank recall."""
    extras = expand_words(question)
    return question + (' ' + ' '.join(extras) if extras else '')


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


def _contains_number(number: str, chunk: str) -> bool:
    """True if `number` appears in `chunk` as a standalone number (22 ≠ 220/1220)."""
    return re.search(rf'(?<!\d){re.escape(number)}(?!\d)', chunk) is not None


def _rerank(question: str, chunks: list[str]) -> list[str]:
    """
    Re-rank retrieved chunks by exact overlap with query tokens.

    Scoring (additive):
    - +5  per query number found anywhere in chunk (as a standalone number)
    - +8  bonus when a query number appears adjacent to a plot keyword —
          capped at ONE bonus per distinct number (avoids over-rewarding chunks
          that mention the same plot in multiple sentences)
    - +12 bonus when a "גודל <plot-term> N" or "<plot-term> N ... AREA מר"
          pattern is found (direct size statement — highest signal)
    - +1  per query word found in chunk
    """
    tokens   = re.findall(r'\d+|[֐-׿]+', question)
    numbers  = {t for t in tokens if t.isdigit()}
    words    = {t for t in tokens if not t.isdigit()}

    def score(chunk: str) -> int:
        s = sum(5 for n in numbers if _contains_number(n, chunk))
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


def _file_signature(fp: Path) -> dict:
    st = fp.stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


class RAGPipeline:
    """
    Orchestrates embedding, FAISS vector search, Whoosh keyword search and LLM
    answer synthesis.

    Args:
        embedder:  object with .encode(texts, is_query=False) — injected in
                   tests; defaults to the SentenceTransformer-backed Embedder
                   (imported lazily so importing this module never loads torch).
        store_dir: directory for the persisted FAISS store + manifest.
        index_dir: directory for the persisted Whoosh keyword index.
    """

    def __init__(self, embedder=None, store_dir: str = 'vector_store', index_dir: str = 'indexdir'):
        if embedder is None:
            from embeddings.embedder import Embedder  # lazy: pulls in torch
            embedder = Embedder(EMBEDDING_MODEL)
        self.embedder = embedder
        self.store_dir = Path(store_dir)
        self.vector_store = None
        self.keyword_search = KeywordSearch(index_dir)

        # try to load persisted vector store if present
        try:
            if (self.store_dir / 'index.faiss').exists():
                self.vector_store = VectorStore.load(self.store_dir)
        except StoreCorruptError as exc:
            logger.warning("Vector store not loaded: %s", exc)
            self.vector_store = None
        except Exception:
            logger.exception("Vector store load failed; starting without it")
            self.vector_store = None

    # ------------------------------------------------------------------
    # Indexed-files manifest (dedup for incremental indexing)
    # ------------------------------------------------------------------

    @property
    def _manifest_path(self) -> Path:
        return self.store_dir / 'manifest.json'

    def _load_manifest(self) -> dict:
        try:
            with open(self._manifest_path, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with open(self._manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build_index(self, chunks, source_files: list | None = None):
        """
        Full rebuild: replaces BOTH the FAISS store and the Whoosh keyword
        index (previously Whoosh was appended to, so every rebuild duplicated
        keyword docs and desynced the two stores).
        """
        embeddings = self.embedder.encode(chunks)

        self.vector_store = VectorStore()
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save(self.store_dir)

        self.keyword_search.clear()
        self.keyword_search.add_docs(chunks)

        manifest = {}
        for fp in source_files or []:
            fp = Path(fp)
            manifest[fp.name] = _file_signature(fp)
        self._save_manifest(manifest)

    def _extract_file_chunks(self, fp: Path) -> list[str]:
        """Extract, clean and chunk one PDF/image file; chunks carry the [filename] prefix."""
        # Lazy imports: pymupdf/cv2/pytesseract are only needed when actually
        # ingesting files, and this keeps API startup and tests lightweight.
        if fp.suffix.lower() == '.pdf':
            from ingestion.pdf_loader import extract_text_from_pdf
            text = extract_text_from_pdf(str(fp))
        else:
            from ingestion.ocr_pipeline import ocr_image
            text = ocr_image(str(fp))
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        return [f"[{fp.name}] " + c for c in chunks]

    def index_new_files(self, file_paths: list) -> int:
        """
        Incrementally index new files without rebuilding the full index.

        Files already recorded in the manifest with an unchanged size/mtime are
        skipped, so re-downloading the same plan does not duplicate its chunks.

        Returns:
            Number of new chunks added.
        """
        manifest = self._load_manifest()
        all_chunks = []
        processed: list[Path] = []
        for fp in file_paths:
            fp = Path(fp)
            try:
                sig = _file_signature(fp)
                if manifest.get(fp.name) == sig:
                    logger.info("index_new_files: %s unchanged — skipping", fp.name)
                    continue
                prefixed = self._extract_file_chunks(fp)
                all_chunks.extend(prefixed)
                processed.append(fp)
                logger.info("index_new_files: %s -> %d chunks", fp.name, len(prefixed))
            except Exception as exc:
                logger.error("index_new_files: failed to process %s: %s", fp, exc)

        added = self._persist_chunks(all_chunks)
        if processed:
            for fp in processed:
                manifest[fp.name] = _file_signature(fp)
            self._save_manifest(manifest)
        return added

    def _persist_chunks(self, chunks: list[str]) -> int:
        """Append prepared chunks to the FAISS + Whoosh stores and persist."""
        if not chunks:
            return 0

        # Append to FAISS store (create new store if none exists yet)
        embeddings = self.embedder.encode(chunks)
        if self.vector_store is None:
            self.vector_store = VectorStore()
        self.vector_store.add(embeddings, chunks)
        self.vector_store.save(self.store_dir)

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

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> list[str]:
        # Embed with query prefix (E5 convention) for vector search
        q_emb = self.embedder.encode([question], is_query=True)
        vector_results = []
        if self.vector_store is not None:
            vector_results = self.vector_store.search(q_emb, TOP_K)

        # KeywordSearch applies synonym expansion itself (per-word OR groups);
        # passing a pre-expanded question would double-expand and pollute the
        # high-precision AND query.
        keyword_results = self.keyword_search.search(question, top_k=TOP_K)

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
