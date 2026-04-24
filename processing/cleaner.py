import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u0590-\u05FF]', '', text)  # keep Hebrew
    return text.strip()