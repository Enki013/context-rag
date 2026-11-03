import os

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
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
DEFAULT_DIFFICULTY = "medium"
DEFAULT_QUESTION_TYPE = "multiple_choice"

# Grounding doğrulaması: her soru ikinci bir LLM çağrısıyla metne karşı kontrol edilir.
# Kapatmak için QUIZ_VERIFY_ANSWERS=0 (daha hızlı, ama yanlış cevaplı sorular geçebilir).
VERIFY_ANSWERS = os.environ.get("QUIZ_VERIFY_ANSWERS", "1") != "0"
# Doğrulayıcı model. Üreticiden farklı bir aile seçildi: evals/verifier_eval.json üzerinde
# gemma4:31b (thinking kapalı) hatalı soruların 22/25'ini, qwen2.5:14b 18/25'ini yakaladı.
# Model kurulu değilse doğrulama başarısız olur ve sorular atılır: `ollama pull gemma4:31b`
# veya QUIZ_VERIFY_MODEL=qwen2.5:14b ile üreticiyi kullanın.
VERIFY_MODEL = os.environ.get("QUIZ_VERIFY_MODEL", "gemma4:31b")
# "0" (varsayılan): thinking kapalı, ~10 kat hızlı ve thinking desteklemeyen modellerde de çalışır.
# "1": thinking açık (yalnızca destekleyen modellerde). "auto": modelin kendi varsayılanı.
VERIFY_THINK = {"0": False, "1": True}.get(os.environ.get("QUIZ_VERIFY_THINK", "0"))

# Tekrar eleme (bge-m3 benzerliği). İki ölçütten biri tutarsa tekrar sayılır:
#  - soru + doğru cevap >= 0.85: aynı soru, aynı cevap (havuzdaki gerçek tekrar 0.89, farklı sorular <= 0.79)
#  - yalnızca soru >= 0.95: aynı soru, farklı/çelişen cevap (gözlenen örnek 0.958)
# 1.0 ilgili ölçütü kapatır. Yanlışlıkla bir soru atmak, tekrar/çelişki göstermekten ucuzdur.
DEDUP_SIMILARITY_THRESHOLD = float(os.environ.get("QUIZ_DEDUP_THRESHOLD", "0.85"))
DEDUP_QUESTION_THRESHOLD = float(os.environ.get("QUIZ_DEDUP_QUESTION_THRESHOLD", "0.95"))

# Contextual Retrieval
CONTEXTUAL_LLM_THRESHOLD = 50  # Bu chunk sayısının altında LLM ile bağlam üretilir, üstünde deterministic

# Limits
MAX_FILE_SIZE_MB = 50

# Admin uçları (/books/import, /quiz/populate, cevaplı /quiz/serve) için X-Admin-Key değeri.
# Tanımlı değilse bu uçlar kapalıdır.
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY")

# Dataset & Books (Project Gutenberg)
HF_DATASET_NAME = "Project Gutenberg"
DATASET_CACHE_DIR = os.path.join(BASE_DIR, "data", "dataset_cache")
CATALOG_DB_PATH = os.path.join(DATASET_CACHE_DIR, "catalog.db")
QUIZ_ENGINE_DB_PATH = os.path.join(BASE_DIR, "data", "quiz_engine.db")
BOOKS_DIR = os.path.join(BASE_DIR, "data", "books")
DATASET_SEARCH_THRESHOLD = 0.0    # SQLite FTS arama kullandığı için threshold gerekmez
DATASET_MAX_RESULTS = 20          # Quiz'e beslenecek chunk sayısı
