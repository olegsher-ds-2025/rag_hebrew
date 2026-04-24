from fastapi import FastAPI
from rag.pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

@app.get("/query")
def query(q: str):
    results = rag.query(q)
    return {"results": results}