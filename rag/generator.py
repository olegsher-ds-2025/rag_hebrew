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


def _build_prompt(question: str, chunks: list[str]) -> str:
    """Assemble a compact prompt from the top retrieved chunks."""
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
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, chunks: list[str]) -> str:
    """
    Stream a response from llama-server to avoid read-timeout on slower models.
    Returns the generated answer, or empty string on failure.
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
        return ""
    except requests.exceptions.Timeout as exc:
        logger.error("[llama.cpp] Request timed out (connect=%ss, read=%ss): %s", LLAMACPP_TIMEOUT[0], LLAMACPP_TIMEOUT[1], exc)
        return ""
    except requests.exceptions.HTTPError as exc:
        logger.error("[llama.cpp] HTTP error %s: %s", resp.status_code, exc)
        return ""
    except Exception as exc:
        logger.error("[llama.cpp] Unexpected error: %s", exc)
        return ""
