from pathlib import Path
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

# serve static JS from api/static
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
# serve raw documents under /files
raw_dir = Path(__file__).resolve().parents[1] / 'data' / 'raw'
app.mount("/files", StaticFiles(directory=raw_dir), name="files")


@app.get("/query")
def query(q: str):
        print(f"[server] GET /query q={q!r} vector_store_set={rag.vector_store is not None} indexdir_exists={Path('indexdir').exists()}")
        results = rag.query(q)
        formatted = []
        for r in results:
            source = None
            text = r
            # extract leading [filename] prefix if present
            if isinstance(r, str) and r.startswith('['):
                end = r.find(']')
                if end > 1:
                    fname = r[1:end]
                    # remainder after closing bracket
                    rest = r[end+1:].lstrip()
                    text = rest or r
                    file_path = Path('data') / 'raw' / fname
                    if file_path.exists():
                        source = f"/files/{fname}"
            formatted.append({"text": text, "source": source})
        return {"results": formatted}


class QueryRequest(BaseModel):
    q: str


@app.post("/query")
def query_post(req: QueryRequest):
    print(f"[server] POST /query q={req.q!r} vector_store_set={rag.vector_store is not None} indexdir_exists={Path('indexdir').exists()}")
    results = rag.query(req.q)
    formatted = []
    for r in results:
        source = None
        text = r
        if isinstance(r, str) and r.startswith('['):
            end = r.find(']')
            if end > 1:
                fname = r[1:end]
                rest = r[end+1:].lstrip()
                text = rest or r
                file_path = Path('data') / 'raw' / fname
                if file_path.exists():
                    source = f"/files/{fname}"
        formatted.append({"text": text, "source": source})
    return {"results": formatted}


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
        </style>
    </head>
    <body>
        <h1>RAG Query</h1>
        <label for="question">Question</label>
        <br />
        <textarea id="question" rows="4" placeholder="Enter your question here..."></textarea>
        <br />
        <button id="send">Send</button>

        <h2>Response</h2>
        <textarea id="response" rows="4" readonly placeholder="Response will appear here..." dir="rtl"></textarea>
        <div id="results" style="margin-top:12px;"></div>

        <script src="/static/app.js"></script>
    </body>
</html>"""