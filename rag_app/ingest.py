import os
import json
import logging
import numpy as np
import pdfplumber
from rag_app.query import load_embedding_model
from rag_app.utils import clean_text, chunk_text
from rag_app.config import CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                except Exception as e:
                    logger.warning("Failed to extract text from page %d of '%s': %s", page_num, pdf_path, e)
    except Exception as e:
        logger.error("Cannot open PDF '%s': %s", pdf_path, e)
        raise ValueError(f"Malformed or unreadable PDF '{os.path.basename(pdf_path)}': {e}") from e

    if not full_text.strip():
        raise ValueError(f"No extractable text found in '{os.path.basename(pdf_path)}'. It may be scanned or image-only.")

    return full_text


def build_index(data_dir: str) -> None:
    model = load_embedding_model()
    os.makedirs(INDEX_DIR, exist_ok=True)

    all_chunks = []
    all_metadata = []

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        return

    failed = []
    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)
        try:
            raw_text = extract_text_from_pdf(pdf_path)
        except ValueError as e:
            logger.error("Skipping '%s': %s", filename, e)
            failed.append(filename)
            continue

        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({"source": filename, "chunk_id": i})

    if failed:
        logger.warning("Skipped %d file(s) due to errors: %s", len(failed), failed)

    if not all_chunks:
        logger.error("No chunks to index — all PDFs failed or were empty.")
        return

    embeddings = model.encode(all_chunks, show_progress_bar=True)

    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)

    with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
        json.dump(all_metadata, f)

    with open(os.path.join(INDEX_DIR, "index_config.json"), "w") as f:
        json.dump({"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}, f)


if __name__ == "__main__":
    from rag_app.config import DATA_DIR
    build_index(DATA_DIR)