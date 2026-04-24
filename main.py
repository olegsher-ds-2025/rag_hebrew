from ingestion.pdf_loader import extract_text_from_pdf
from processing.cleaner import clean_text
from processing.chunker import chunk_text
from rag.pipeline import RAGPipeline

file_path = "data/raw/45.pdf"

text = extract_text_from_pdf(file_path)
text = clean_text(text)
chunks = chunk_text(text)

rag = RAGPipeline()
rag.build_index(chunks)

print(rag.query("מה כתוב במסמך?"))