#!/usr/bin/env python3
import sys
from pathlib import Path

# ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from ingestion.pdf_loader import extract_text_from_pdf
from processing.chunker import chunk_text
from processing.cleaner import clean_text
from rag.pipeline import RAGPipeline


def build_and_query(file_name, queries):
    file_path = Path(settings.raw_dir) / file_name
    print(f"Loading: {file_path}")

    text = extract_text_from_pdf(str(file_path))
    print(f"Raw extracted length: {len(text)} characters")

    cleaned = clean_text(text)
    print(f"Cleaned length: {len(cleaned)} characters")

    chunks = chunk_text(cleaned, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    print(f"Created {len(chunks)} chunks (first 2 shown):\n", chunks[:2])

    rag = RAGPipeline()
    print("Building vector+keyword index...")
    rag.build_index(chunks)
    print("Index built.")

    for q in queries:
        print('\n=== Query:', q)
        res = rag.query(q)
        print('Results count:', len(res))
        for i, r in enumerate(res[:10], 1):
            print(f"{i}. {r[:200]}")


if __name__ == '__main__':
    samples = [
        "מה מספר תוכנית",
        "מה כתוב במסמך?",
        "מספר תוכנית",
        "306/02/6",
    ]
    build_and_query("45.pdf", samples)
