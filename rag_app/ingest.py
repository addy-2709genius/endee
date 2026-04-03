import os
import json
import numpy as np
import pdfplumber
from rag_app.query import load_embedding_model
from rag_app.utils import clean_text, chunk_text
from rag_app.config import CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR


def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


def build_index(data_dir: str) -> None:
    model = load_embedding_model()
    os.makedirs(INDEX_DIR, exist_ok=True)

    all_chunks = []
    all_metadata = []

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        return

    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)
        raw_text = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({"source": filename, "chunk_id": i})

    embeddings = model.encode(all_chunks, show_progress_bar=True)

    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)

    with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
        json.dump(all_metadata, f)


if __name__ == "__main__":
    from rag_app.config import DATA_DIR
    build_index(DATA_DIR)