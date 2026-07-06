import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from downloader.manager import DownloadManager
from rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Only one /download may run at a time: it drives network fetches plus heavy
# embedding/indexing work, and a second concurrent run would fight over the
# same stores. Non-blocking acquire → 409 for the loser.
_download_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the heavy pieces at startup (not import time): the embedding model
    # takes a while on the Jetson, and /health reports readiness meanwhile.
    # Tests inject fakes by setting app.state.rag before startup.
    if getattr(app.state, "rag", None) is None:
        logger.info("Loading RAG pipeline (embedding model)...")
        app.state.rag = RAGPipeline()
        app.state.rag.warmup()   # load the model now so /health reflects readiness
        n = len(app.state.rag.vector_store.texts) if app.state.rag.vector_store else 0
        logger.info("RAG pipeline ready (%d indexed chunks)", n)
    if getattr(app.state, "download_manager", None) is None:
        app.state.download_manager = DownloadManager()
    yield


app = FastAPI(lifespan=lifespan)

# serve static JS from api/static
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# serve raw documents under /files
raw_dir = Path(__file__).resolve().parents[1] / 'data' / 'raw'
raw_dir.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=raw_dir), name="files")


def _get_rag() -> RAGPipeline:
    rag = getattr(app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="pipeline is still loading")
    return rag


def _safe_source(fname: str) -> str | None:
    """
    Map a [filename] chunk prefix to a /files/ URL, but only for names that are
    plain basenames of files actually present in data/raw. The prefix
    originates from remote-supplied document names — never trust it as a path.
    """
    if not fname or Path(fname).name != fname or fname in ('.', '..'):
        return None
    if not (raw_dir / fname).is_file():
        return None
    return f"/files/{fname}"


def _format_query_response(q: str) -> dict:
    logger.info("[server] query q=%r", q)
    result = _get_rag().query_with_answer(q)
    formatted = []
    for r in result["chunks"]:
        source = None
        text = r
        if isinstance(r, str) and r.startswith('['):
            end = r.find(']')
            if end > 1:
                fname = r[1:end]
                rest = r[end + 1:].lstrip()
                text = rest or r
                source = _safe_source(fname)
        formatted.append({"text": text, "source": source})
    return {"answer": result["answer"], "results": formatted}


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=2000)


class DownloadRequest(BaseModel):
    plan_name: str = Field(min_length=1, max_length=200)


# Optional shared-secret protection for the write/network-triggering endpoint.
# Set API_TOKEN in the environment to require an X-API-Token header on /download.
API_TOKEN = settings.api_token


def require_api_token(x_api_token: str = Header(default="")):
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Token")


@app.get("/health")
def health():
    rag = getattr(app.state, "rag", None)
    if rag is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return {
        "status": "ok",
        "indexed_chunks": len(rag.vector_store.texts) if rag.vector_store else 0,
        "keyword_docs": rag.keyword_search.doc_count(),
    }


@app.post("/download", dependencies=[Depends(require_api_token)])
def download_and_index(req: DownloadRequest):
    logger.info("[server] POST /download plan_name=%r", req.plan_name)
    rag = _get_rag()
    if not _download_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="הורדה אחרת כבר רצה — נסה שוב בעוד רגע")
    try:
        downloaded, log, metadata = app.state.download_manager.download(req.plan_name)

        # Index the plan's basic info (status, area, dates) — works even when the
        # PDFs are reCAPTCHA-gated and could not be downloaded.
        meta_indexed = rag.index_texts(metadata)
        if meta_indexed:
            log.append(f"אונדקס מידע בסיסי ({meta_indexed} קטעים) מדף התכנית")

        indexed = rag.index_new_files(downloaded) if downloaded else 0
        names = [p.name for p in downloaded]
        if downloaded:
            log.append(f"אונדקסו {indexed} קטעים מ-{len(names)} קבצים")
    finally:
        _download_lock.release()

    logger.info("[server] download: %d file(s), %d doc chunks, %d metadata chunks indexed",
                len(names), indexed, meta_indexed)
    return {
        "downloaded": names,
        "indexed_chunks": indexed,
        "indexed_metadata": meta_indexed,
        "log": log,
    }


@app.post("/query")
def query_post(req: QueryRequest):
    return _format_query_response(req.q)


@app.get("/", response_class=HTMLResponse)
def home():
        # Cache-bust app.js by its mtime so browsers always pick up JS changes
        # (e.g. the source-chunk line-break formatting) instead of a stale copy.
        try:
            app_js_ver = int((static_dir / "app.js").stat().st_mtime)
        except OSError:
            app_js_ver = 0
        return _HOME_HTML.replace(
            "/static/app.js", f"/static/app.js?v={app_js_ver}"
        )


_HOME_HTML = """<!doctype html>
<html>
    <head>
        <meta charset="utf-8" />
        <title>RAG Query</title>
        <style>
            body { font-family: Arial, Helvetica, sans-serif; margin: 20px; }
            textarea { width: 100%; max-width: 800px; }
            button { margin-top: 8px; padding: 8px 12px; }
            /* response box RTL formatting for Hebrew */
            #response { direction: rtl; text-align: right; }
            /* question box also set to RTL for Hebrew input */
            #question { direction: rtl; text-align: right; }
            .section { border: 1px solid #ccc; border-radius: 6px; padding: 16px; max-width: 820px; margin-bottom: 24px; }
            .section h2 { margin-top: 0; }
            input[type=text] { padding: 6px 10px; font-size: 1em; width: 120px; }
            #download-status { direction: rtl; text-align: right; margin-top: 10px; color: #333; }
            #answer-box { display:none; background:#eef6ff; border:1px solid #aad0f0; border-radius:6px; padding:14px 18px; margin-top:14px; direction:rtl; text-align:right; font-size:1.05em; line-height:1.8; white-space:pre-wrap; }
            #answer-box strong { color:#1a5fa8; }
            details { margin-top:10px; }
            details summary { cursor:pointer; color:#555; font-size:0.9em; }
        </style>
    </head>
    <body>
        <h1>RAG Query</h1>

        <div class="section">
            <h2>הורדת מסמכים לפי שם תכנית</h2>
            <label>שם תכנית: <input type="text" id="plan-name" placeholder="לדוגמה: נוף הפארק - יובלים גנים" dir="rtl" style="width:320px;" /></label>
            &nbsp;
            <button id="download-btn">הורד ואנדקס</button>
            <div id="download-status" style="margin-top:10px;padding:10px;background:#f5f5f5;border-radius:4px;min-height:36px;display:none;direction:rtl;text-align:right;font-family:monospace;white-space:pre-wrap;line-height:1.7;"></div>
        </div>

        <div class="section">
            <h2>שאילתה</h2>
            <label for="question">שאלה</label>
            <br />
            <textarea id="question" rows="3" placeholder="הכנס שאלה כאן..." dir="rtl" style="width:100%;max-width:800px;"></textarea>
            <br />
            <button id="send">שלח</button>

            <div id="answer-box"></div>

            <details id="sources-details" style="display:none;">
                <summary>מקורות (<span id="sources-count">0</span>)</summary>
                <div id="results" style="margin-top:8px;"></div>
            </details>
        </div>

        <script src="/static/app.js"></script>
    </body>
</html>"""