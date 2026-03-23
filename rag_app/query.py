import os
import json
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from rag_app.utils import format_context
from rag_app.config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    MAX_TOKENS,
    TOP_K,
    INDEX_DIR
)


def load_index():
    """Load embeddings, chunks and metadata from disk."""
    embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "r") as f:
        chunks = json.load(f)

    with open(os.path.join(INDEX_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return embeddings, chunks, metadata


def cosine_similarity(query_vec: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all chunk embeddings."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(embeddings_norm, query_norm)


def retrieve_chunks(query: str, top_k: int = TOP_K) -> tuple[list[str], list[dict]]:
    """
    Embed the query and retrieve top-k most similar chunks
    using cosine similarity — no external vector DB call needed here,
    Endee handles the vector storage layer.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings, chunks, metadata = load_index()

    query_embedding = model.encode([query])[0]
    scores = cosine_similarity(query_embedding, embeddings)

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]
    top_metadata = [metadata[i] for i in top_indices]

    return top_chunks, top_metadata


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """Send retrieved context + question to Groq LLM and get answer."""
    client = Groq(api_key=GROQ_API_KEY)
    context = format_context(context_chunks)

    prompt = f"""You are a helpful AI assistant. Answer the user's question 
based strictly on the provided context. If the answer is not in the context, 
say "I could not find relevant information in the document."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def ask(query: str) -> dict:
    """Full RAG pipeline: retrieve + generate."""
    chunks, metadata = retrieve_chunks(query)
    answer = generate_answer(query, chunks)

    return {
        "answer": answer,
        "sources": list(set([m["source"] for m in metadata]))
    }