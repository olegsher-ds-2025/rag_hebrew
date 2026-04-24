import re


def clean_text(text):
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # keep Hebrew, letters/digits/underscore, whitespace, and common identifier punctuation
    # (slash, hyphen, dot) so IDs like 306/02/6 are preserved.
    text = re.sub(r"[^\w\s\u0590-\u05FF/\-\.]", '', text)
    return text.strip()