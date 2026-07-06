"""
LLM answer generator using a llama.cpp server (llama-server).
Synthesises retrieved chunks into a concise Hebrew answer.

The app runs in a container and talks to a `llama-server` HTTP daemon running
on the Jetson host. Point LLAMACPP_URL at the host's /completion endpoint, e.g.
    LLAMACPP_URL=http://host.docker.internal:8080/completion
"""

import json as _json
import logging
import os

import requests

logger = logging.getLogger(__name__)

# llama-server native completion endpoint (NOT the OpenAI-compatible /v1 one).
LLAMACPP_URL = os.getenv("LLAMACPP_URL", "http://host.docker.internal:8080/completion")
# Max tokens to generate. Answers are 1-2 sentences, so this stays small.
LLAMACPP_N_PREDICT = int(os.getenv("LLAMACPP_N_PREDICT", "256"))
LLAMACPP_TEMPERATURE = float(os.getenv("LLAMACPP_TEMPERATURE", "0.2"))
# (connect_timeout, read_timeout) — generous read timeout for Jetson inference.
LLAMACPP_TIMEOUT = (10, 300)
MAX_CONTEXT_CHARS = 2000    # keep context short for speed
TOP_N_CHUNKS = 5

PROMPT_TEMPLATE = """אתה עוזר מומחה לתכנון ובניה בישראל.
בהתבסס על קטעי המסמכים הבאים בלבד, ענה על השאלה בעברית בצורה תמציתית ומדויקת.

הנחיות חשובות:
- כאשר מופיעות שדות מובנות (שם התכנית:, סטטוס:, סה״כ שטח בדונם:, וכו׳), העתק בדיוק את הערך הנתון בשדה.
- כאשר שואלים על "שטח" של תא שטח / מגרש, הכוונה היא לגודל המגרש הפיזי (גודל מגרש במ"ר) ולא לזכויות הבנייה (שטח עיקרי/שטח מרפסות).
- כלול את הנתון המספרי הספציפי בדיוק כפי שהוא מופיע, ואת יחידת המידה.
- אם המידע אינו במסמכים, ענה בדיוק: "אין מידע בתכניות"
- ענה במשפט אחד עד שניים בלבד.
- אל תנחש, אל תשנה מספרים, אל תתן תשובה ספקולטיבית.

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


def _is_no_answer_response(answer: str) -> bool:
    """Check if the LLM explicitly said it has no relevant data."""
    if not answer:
        return True
    # LLM should say one of these when docs don't contain the answer
    no_data_phrases = [
        "אין מידע",
        "אינו מופיע",
        "לא נמצא",
        "לא קיים",
        "לא צוין",
        "לא מופיע",
    ]
    lower = answer.lower()
    return any(phrase in lower for phrase in no_data_phrases)


def _fix_area_number(answer: str, question: str, chunks: list[str]) -> str:
    """
    If the question asks about area (שטח) and the answer contains a suspicious area number,
    try to extract the correct value from the metadata field 'סה״כ שטח בדונם:'.

    This is a safety net for LLM mistakes like confusing 0.975 with 9.75.
    """
    import re

    if "שטח" not in question:
        return answer

    # Look for "סה״כ שטח בדונם:" field in chunks
    for chunk in chunks:
        m = re.search(r'סה"כ שטח בדונם:\s*([\d.]+)', chunk)
        if m:
            correct_area = m.group(1)
            # Replace any suspiciously large area (> 5 dunams for single plans) with the correct value
            # This catches cases like "9.75" that should be "0.975"
            if re.search(r'[5-9]\.\d+|[1-9]\d+', answer) and '0.' in correct_area:
                return f"סה״כ שטח בדונם: {correct_area}"

    return answer


def _fix_plan_name(answer: str, question: str, chunks: list[str]) -> str:
    """
    If the question asks about plan name (שם התכנית) and the LLM returned only
    the plan number, extract the actual name from the metadata field 'שם התכנית:'.
    """
    import re

    if "שם" not in question or "תכנית" not in question:
        return answer

    # If answer is just a plan number, try to extract the real name
    if re.match(r'^תוכנית\s*\d{3}-\d{7}$', answer.strip()):
        # Look for "שם התכנית:" field in chunks
        for chunk in chunks:
            m = re.search(r'שם התכנית:\s*(.+?)(?:\n|$)', chunk)
            if m:
                plan_name = m.group(1).strip()
                if plan_name and plan_name != answer:
                    return plan_name

    return answer


def _build_prompt(question: str, chunks: list[str]) -> str:
    """Assemble a compact prompt from the top retrieved chunks, distributing context fairly."""
    # Allocate equal context per chunk so one chunk can't starve others
    per_chunk = MAX_CONTEXT_CHARS // TOP_N_CHUNKS
    parts = []
    for c in chunks[:TOP_N_CHUNKS]:
        text = _strip_prefix(c)
        if text:
            parts.append(text[:per_chunk])

    context = "\n---\n".join(parts)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, chunks: list[str]) -> str | None:
    """
    Stream a response from llama-server to avoid read-timeout on slower models.

    Returns:
        The generated answer ("" when there was nothing to ask about), or
        None when the LLM itself was unreachable/failed — callers use this to
        show a clear "model unavailable" message instead of a silent blank.
    """
    if not chunks:
        return ""

    prompt = _build_prompt(question, chunks)

    logger.info("[llama.cpp] Connecting to %s", LLAMACPP_URL)
    logger.info("[llama.cpp] Prompt length: %d chars", len(prompt))

    payload = {
        "prompt": prompt,
        "n_predict": LLAMACPP_N_PREDICT,
        "temperature": LLAMACPP_TEMPERATURE,
        "stream": True,
        # Stop once the model starts a new turn / question.
        "stop": ["\nשאלה:", "\nמסמכים:"],
    }

    try:
        resp = requests.post(
            LLAMACPP_URL,
            json=payload,
            stream=True,
            timeout=LLAMACPP_TIMEOUT,
        )
        logger.info("[llama.cpp] HTTP status: %s", resp.status_code)
        resp.raise_for_status()

        answer_parts = []
        token_count = 0
        # Decode the stream as UTF-8 ourselves. llama-server sends UTF-8, but its
        # Content-Type is text/event-stream with no charset, and requests then
        # defaults text/* to ISO-8859-1 — which mangles Hebrew. Iterate raw bytes
        # (SSE lines end on \n, a safe boundary for multibyte UTF-8) and decode.
        for raw in resp.iter_lines(chunk_size=None, decode_unicode=False):
            if not raw:
                continue
            raw = raw.decode("utf-8", errors="replace")
            # llama-server streams Server-Sent Events: "data: {json}".
            line = raw[len("data:"):].strip() if raw.startswith("data:") else raw.strip()
            if not line or line == "[DONE]":
                continue
            try:
                token = _json.loads(line)
            except Exception:
                continue
            answer_parts.append(token.get("content", ""))
            token_count += 1
            if token.get("stop"):
                logger.info("[llama.cpp] Stream done after %d tokens", token_count)
                break

        answer = "".join(answer_parts).strip()
        logger.info("[llama.cpp] Answer length: %d chars", len(answer))
        return answer

    except requests.exceptions.ConnectionError as exc:
        logger.error("[llama.cpp] Connection failed (is llama-server running at %s?): %s", LLAMACPP_URL, exc)
        return None
    except requests.exceptions.Timeout as exc:
        logger.error("[llama.cpp] Request timed out (connect=%ss, read=%ss): %s", LLAMACPP_TIMEOUT[0], LLAMACPP_TIMEOUT[1], exc)
        return None
    except requests.exceptions.HTTPError as exc:
        logger.error("[llama.cpp] HTTP error %s: %s", resp.status_code, exc)
        return None
    except Exception as exc:
        logger.error("[llama.cpp] Unexpected error: %s", exc)
        return None
