import re


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments on paragraph and page breaks."""
    # Split on page markers and paragraph breaks first
    segs = re.split(r'\s*---\s*PAGE\s*\d+\s*---\s*|\n{2,}', text)
    sentences = []
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        # Further split on sentence-ending punctuation followed by whitespace
        subs = re.split(r'(?<=[.!?])\s+', seg)
        sentences.extend(s.strip() for s in subs if s.strip())
    return sentences


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """
    Sentence-aware chunking: split on sentence/paragraph boundaries, then
    group sentences into chunks of ~chunk_size chars with overlap carry-over.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= overlap < chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size, got overlap={overlap}, chunk_size={chunk_size}"
        )

    sentences = _split_sentences(text)
    if not sentences:
        return [text[:chunk_size]] if text.strip() else []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_len + sent_len + 1 > chunk_size and current_parts:
            # Emit current chunk
            chunks.append(' '.join(current_parts))
            # Carry over the tail as overlap
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_parts):
                if overlap_len + len(part) + 1 <= overlap:
                    overlap_parts.insert(0, part)
                    overlap_len += len(part) + 1
                else:
                    break
            current_parts = overlap_parts
            current_len = overlap_len

        if sent_len > chunk_size:
            # Single sentence exceeds chunk_size — split by chars
            if current_parts:
                chunks.append(' '.join(current_parts))
                current_parts = []
                current_len = 0
            for i in range(0, sent_len, chunk_size - overlap):
                chunks.append(sent[i:i + chunk_size])
        else:
            current_parts.append(sent)
            current_len += sent_len + 1

    if current_parts:
        chunks.append(' '.join(current_parts))

    return chunks if chunks else [text[:chunk_size]]
