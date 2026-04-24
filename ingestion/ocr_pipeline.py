import pytesseract
from PIL import Image
import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def ocr_image(image_path, lang="heb"):
    img = preprocess_image(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text