import os

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:14b"
EMBEDDING_MODEL = "bge-m3"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")

# Quiz defaults
DEFAULT_QUIZ_COUNT = 5
DEFAULT_DIFFICULTY = "orta"
DEFAULT_QUESTION_TYPE = "multiple_choice"

# Contextual Retrieval
CONTEXTUAL_LLM_THRESHOLD = 50  # Bu chunk sayısının altında LLM ile bağlam üretilir, üstünde deterministic

# Limits
MAX_FILE_SIZE_MB = 50
