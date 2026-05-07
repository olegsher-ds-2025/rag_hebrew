"""
LLM answer generator using a local Ollama model.
Synthesises retrieved chunks into a concise Hebrew answer.
"""

import requests
import logging
import os

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "qwen2.5-coder:7b"
# (connect_timeout, read_timeout) — generous read timeout for remote/CPU inference
OLLAMA_TIMEOUT = (10, 300)
MAX_CONTEXT_CHARS = 2000    # keep context short for speed
TOP_N_CHUNKS = 5

PROMPT_TEMPLATE = """אתה עוזר מומחה לתכנון ובניה בישראל.
בהתבסס על קטעי המסמכים הבאים בלבד, ענה על השאלה בעברית בצורה תמציתית ומדויקת.

הנחיות חשובות:
- כאשר שואלים על "שטח" של תא שטח / מגרש, הכוונה היא לגודל המגרש הפיזי (גודל מגרש במ"ר) ולא לזכויות הבנייה (שטח עיקרי/שטח מרפסות).
- כלול את הנתון המספרי הספציפי ואת יחידת המידה.
- אם המידע אינו במסמכים, ציין זאת במפורש.
- ענה במשפט אחד עד שניים בלבד.

מסמכים:
{context}

שאלה: {question}
תשובה:"""


def _strip_prefix(chunk: str) -> str:
    """Remove [filename] prefix from chunk."""
    if chunk.startswith('['):
        end = chunk.find(']')
        if end > 0:
            return chunk[end + 1:].strip()
    return chunk


def generate_answer(question: str, chunks: list[str]) -> str:
    """
    Stream a response from Ollama to avoid read-timeout on slower models.
    Returns the generated answer, or empty string on failure.
    """
    if not chunks:
        return ""

    # Build compact context from top chunks
    parts = []
    total = 0
    for c in chunks[:TOP_N_CHUNKS]:
        text = _strip_prefix(c)
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        parts.append(text[:remaining])
        total += len(text[:remaining])

    context = "\n---\n".join(parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()

        answer_parts = []
        import json as _json
        for line in resp.iter_lines(chunk_size=None):
            if not line:
                continue
            try:
                token = _json.loads(line)
            except Exception:
                continue
            answer_parts.append(token.get("response", ""))
            if token.get("done"):
                break

        return "".join(answer_parts).strip()

    except Exception as exc:
        logger.warning("LLM generation failed: %s", exc)
        return ""
