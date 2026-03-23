import os
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
from rag_app.utils import clean_text, chunk_text
from rag_app.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file page by page."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


def build_index(data_dir: str) -> None:
    """
    Process all PDFs in data_dir:
    1. Extract text
    2. Clean and chunk
    3. Generate embeddings
    4. Save index to disk
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    os.makedirs(INDEX_DIR, exist_ok=True)

    all_chunks = []
    all_metadata = []

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in", data_dir)
        return

    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)
        print(f"Processing: {filename}")

        raw_text = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": filename,
                "chunk_id": i
            })

    print(f"Total chunks created: {len(all_chunks)}")
    print("Generating embeddings...")

    embeddings = model.encode(all_chunks, show_progress_bar=True)

    # Save everything to disk
    import numpy as np
    np.save(os.path.join(INDEX_DIR, "embeddings.npy"), embeddings)

    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)

    with open(os.path.join(INDEX_DIR, "metadata.json"), "w") as f:
        json.dump(all_metadata, f)

    print("Index built and saved successfully!")


if __name__ == "__main__":
    from rag_app.config import DATA_DIR
    build_index(DATA_DIR)