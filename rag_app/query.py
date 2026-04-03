import os
import json
import numpy as np
import streamlit as st
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


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def load_index():
    embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))

    with open(os.path.join(INDEX_DIR, "chunks.json"), "r") as f:
        chunks = json.load(f)

    with open(os.path.join(INDEX_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)

    return embeddings, chunks, metadata


def cosine_similarity(query_vec: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(embeddings_norm, query_norm)


def retrieve_chunks(query: str, top_k: int = TOP_K) -> tuple[list[str], list[dict]]:
    model = load_embedding_model()
    embeddings, chunks, metadata = load_index()

    query_embedding = model.encode([query])[0]
    scores = cosine_similarity(query_embedding, embeddings)

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_chunks = [chunks[i] for i in top_indices]
    top_metadata = [metadata[i] for i in top_indices]

    return top_chunks, top_metadata


MAX_CONTEXT_WORDS = 2000


def generate_answer(query: str, context_chunks: list[str]) -> str:
    client = Groq(api_key=GROQ_API_KEY)

    # Truncate chunks to stay within Groq token limits
    truncated_chunks = []
    total_words = 0
    for chunk in context_chunks:
        chunk_words = len(chunk.split())
        if total_words + chunk_words > MAX_CONTEXT_WORDS:
            break
        truncated_chunks.append(chunk)
        total_words += chunk_words

    context = format_context(truncated_chunks)

    prompt = f"""You are a helpful AI assistant. Answer the user's question
using the provided context from the document. Be as helpful as possible —
if the context contains relevant information, summarize and explain it clearly.
Only say "I could not find relevant information in the document." if the context
has absolutely no relation to the question.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e).lower()
        if "rate_limit" in error_str or "429" in error_str:
            return "⚠️ Groq API rate limit reached. Please wait a few seconds and try again."
        elif "401" in error_str or "authentication" in error_str:
            return "⚠️ Invalid Groq API key. Please check your API key in the Streamlit secrets."
        else:
            return f"⚠️ Could not get a response from the AI: {str(e)}"


def ask(query: str) -> dict:
    chunks, metadata = retrieve_chunks(query)
    answer = generate_answer(query, chunks)

    return {
        "answer": answer,
        "sources": list(set([m["source"] for m in metadata]))
    }