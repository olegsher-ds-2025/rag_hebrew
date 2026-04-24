import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
import pytesseract
import cv2
import numpy as np


def extract_text_from_pdf(file_path, ocr_lang='heb'):
    doc = fitz.open(file_path)
    text = ""

    for page_num, page in enumerate(doc):
        text += f"\n--- PAGE {page_num} ---\n"

        # try to extract text directly
        page_text = page.get_text().strip()
        if page_text:
            text += page_text
            continue

        # fallback: render page to image and run OCR
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(BytesIO(img_bytes)).convert('RGB')

        # convert to OpenCV image (BGR) for pytesseract compatibility
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ocr_text = pytesseract.image_to_string(cv_img, lang=ocr_lang)
        text += ocr_text

    return text