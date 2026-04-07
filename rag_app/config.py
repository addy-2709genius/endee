import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

LLM_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 512

TOP_K = 5

DATA_DIR = "data/documents"
INDEX_DIR = "data/index"