import re


def clean_text(text: str) -> str:
    """Remove extra whitespace and special characters from text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# Separators tried in order from coarsest to finest granularity
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Recursively split text using a hierarchy of separators.

    Tries paragraph breaks first, then newlines, then sentence-ending
    punctuation, and so on down to individual characters — stopping as
    soon as the resulting pieces fit within chunk_size characters.
    Neighbouring chunks share `overlap` characters of context.
    """
    return _recursive_split(text, chunk_size, overlap, _SEPARATORS)


def _recursive_split(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: list[str],
) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    # Split on the current separator
    if separator:
        splits = text.split(separator)
    else:
        # Character-level fallback
        splits = list(text)

    chunks: list[str] = []
    current = ""

    for piece in splits:
        candidate = (current + separator + piece) if current else piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Flush current buffer
            if current.strip():
                if len(current) > chunk_size and remaining_separators:
                    # Current piece still too large — recurse with finer separator
                    chunks.extend(_recursive_split(current, chunk_size, overlap, remaining_separators))
                else:
                    chunks.append(current)

            # Start a new buffer, prepending overlap from the last chunk
            if chunks and overlap > 0:
                overlap_text = chunks[-1][-overlap:]
                current = overlap_text + separator + piece if overlap_text else piece
            else:
                current = piece

    # Flush whatever remains
    if current.strip():
        if len(current) > chunk_size and remaining_separators:
            chunks.extend(_recursive_split(current, chunk_size, overlap, remaining_separators))
        else:
            chunks.append(current)

    return chunks


def format_context(chunks: list[str]) -> str:
    """Format retrieved chunks into a single context block for the LLM."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {i+1}]:\n{chunk}\n\n"
    return context.strip()