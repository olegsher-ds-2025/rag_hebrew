from pathlib import Path

import cv2
import pytesseract


def preprocess_image(image_path):
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Unreadable/unsupported image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def ocr_image(image_path, lang="heb"):
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text
