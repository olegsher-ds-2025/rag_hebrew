#!/usr/bin/env python3
from pathlib import Path
import sys

# make project importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.ocr_pipeline import ocr_image
from processing.cleaner import clean_text
from processing.chunker import chunk_text
from rag.pipeline import RAGPipeline
from config import RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def build_index_all():
    raw = Path(RAW_DIR)
    if not raw.exists():
        print(f"Raw dir not found: {raw}")
        return

    extensions = ['.pdf', '.PDF', '.jpg', '.JPG', '.png', '.PNG']
    files = [p for p in raw.iterdir() if p.suffix in extensions]
    print(f"Found {len(files)} files in {raw}")

    all_chunks = []
    for f in files:
        print(f"Processing: {f.name}")
        try:
            if f.suffix.lower() == '.pdf':
                text = extract_text_from_pdf(str(f))
            else:
                # image file
                text = ocr_image(str(f))

            cleaned = clean_text(text)
            chunks = chunk_text(cleaned, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            # prefix chunks with filename to preserve provenance
            prefixed = [f"[{f.name}] " + c for c in chunks]
            all_chunks.extend(prefixed)
            print(f"  -> extracted {len(chunks)} chunks")
        except Exception as e:
            print(f"  Error processing {f.name}: {e}")

    print(f"Total chunks collected: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks to index. Exiting.")
        return

    rag = RAGPipeline()
    print("Building index from all chunks (this may take a while)...")
    rag.build_index(all_chunks)
    print("Index built and persisted (vector_store + keyword index). Done.")


if __name__ == '__main__':
    build_index_all()
