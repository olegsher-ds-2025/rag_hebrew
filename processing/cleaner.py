import re


def clean_text(text):
    # Normalize horizontal whitespace but PRESERVE newlines — the chunker
    # relies on paragraph breaks (\n{2,}) to find semantic boundaries.
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r' ?\n ?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Keep Hebrew, letters/digits/underscore, whitespace, and punctuation that
    # carries meaning in planning documents:
    #   / - .   identifiers like 306/02/6
    #   " '     Hebrew abbreviations/units written with ASCII quotes: מ"ר, ש"ח
    #           (the dedicated geresh/gershayim ׳ ״ are inside ֐-׿)
    #   : ( ) % structured fields ("שם התכנית: ..."), clauses, percentages
    text = re.sub(r"[^\w\s֐-׿/\-\.\"':()%]", '', text)
    return text.strip()
