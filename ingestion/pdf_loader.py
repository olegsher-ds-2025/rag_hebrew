import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page_num, page in enumerate(doc):
        text += f"\n--- PAGE {page_num} ---\n"
        text += page.get_text()

    return text