import re

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    return _recursive_split(text, chunk_size, overlap, _SEPARATORS)


def _recursive_split(
    text: str, chunk_size: int, overlap: int, separators: list[str]
) -> list[str]:
    sep = separators[0]
    remaining = separators[1:]

    parts = [p for p in text.split(sep) if p.strip()]
    chunks = []
    buffer: list[str] = []
    buffer_wc = 0

    for part in parts:
        part_wc = len(part.split())

        if part_wc > chunk_size and remaining:
            if buffer:
                chunks.append(sep.join(buffer))
                buffer, buffer_wc = _tail_overlap(buffer, overlap)
            chunks.extend(_recursive_split(part, chunk_size, overlap, remaining))

        elif buffer_wc + part_wc > chunk_size:
            if buffer:
                chunks.append(sep.join(buffer))
                buffer, buffer_wc = _tail_overlap(buffer, overlap)
            buffer.append(part)
            buffer_wc += part_wc

        else:
            buffer.append(part)
            buffer_wc += part_wc

    if buffer:
        chunks.append(sep.join(buffer))

    return chunks


def _tail_overlap(buffer: list[str], overlap: int) -> tuple[list[str], int]:
    tail: list[str] = []
    tail_wc = 0
    for part in reversed(buffer):
        wc = len(part.split())
        if tail_wc + wc <= overlap:
            tail.insert(0, part)
            tail_wc += wc
        else:
            break
    return tail, tail_wc


def format_context(chunks: list[str]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {i+1}]:\n{chunk}\n\n"
    return context.strip()