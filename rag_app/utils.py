import re


def clean_text(text: str) -> str:
    """Remove extra whitespace and special characters from text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    Overlap ensures context is not lost at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def format_context(chunks: list[str]) -> str:
    """Format retrieved chunks into a single context block for the LLM."""
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {i+1}]:\n{chunk}\n\n"
    return context.strip()