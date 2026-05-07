from pathlib import Path
import sys
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.pipeline import RAGPipeline
from downloader.manager import DownloadManager

logging.basicConfig(level=logging.INFO)

app = FastAPI()
rag = RAGPipeline()
download_manager = DownloadManager()

# serve static JS from api/static
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# serve raw documents under /files
raw_dir = Path(__file__).resolve().parents[1] / 'data' / 'raw'
app.mount("/files", StaticFiles(directory=raw_dir), name="files")


def _format_query_response(q: str) -> dict:
    print(f"[server] query q={q!r}")
    result = rag.query_with_answer(q)
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
                if (Path('data') / 'raw' / fname).exists():
                    source = f"/files/{fname}"
        formatted.append({"text": text, "source": source})
    return {"answer": result["answer"], "results": formatted}


@app.get("/query")
def query(q: str):
    return _format_query_response(q)


class QueryRequest(BaseModel):
    q: str


class DownloadRequest(BaseModel):
    plan_name: str


@app.post("/download")
def download_and_index(req: DownloadRequest):
    print(f"[server] POST /download plan_name={req.plan_name!r}")
    downloaded, log = download_manager.download(req.plan_name)
    if not downloaded:
        return {"downloaded": [], "indexed_chunks": 0, "log": log}
    indexed = rag.index_new_files(downloaded)
    names = [p.name for p in downloaded]
    log.append(f"אונדקסו {indexed} קטעים מ-{len(names)} קבצים")
    print(f"[server] download: {len(names)} file(s) downloaded, {indexed} chunks indexed")
    return {"downloaded": names, "indexed_chunks": indexed, "log": log}


@app.post("/query")
def query_post(req: QueryRequest):
    return _format_query_response(req.q)


@app.get("/", response_class=HTMLResponse)
def home():
        return """<!doctype html>
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