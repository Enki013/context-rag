"""
api/main.py
-----------
Context RAG Mobile & Web Backend API.
FastAPI tabanlı, Swagger UI ve CORS destekli ana uygulama dosyası.

Çalıştırma:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Kök dizini PYTHONPATH'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import admin, books, health, jobs, quiz, quizzes
from src.jobs import start_worker, stop_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Uygulama açılınca üretim worker'ını başlatır, kapanınca durdurur (JOB_WORKER_ENABLED=0 kapatır)."""
    enabled = os.environ.get("JOB_WORKER_ENABLED", "1") != "0"
    if enabled:
        start_worker()
    yield
    if enabled:
        stop_worker()


app = FastAPI(
    lifespan=lifespan,
    title="Context RAG - Mobile & Headless Quiz Engine API",
    description=(
        "Mobil ve web uygulamaları için RAG tabanlı kitap arama, EPUB/PDF işleme, "
        "bölüm odaklı kalıcı soru bankası (0ms Cache Hit) ve hile korumalı değerlendirme mikroservisi.\n\n"
        "- 📚 **Project Gutenberg Kataloğu:** 79,000+ kitapta anlık yerel FTS5 arama\n"
        "- 📑 **EPUB & PDF İçerik İçe Aktarma:** Gömülü TOC ve sayfalarla otomatik bölümleme (`POST /api/v1/books/import`)\n"
        "- ⚡ **Headless Soru Bankası & Önbellek:** İhtiyaç anında LLM üretimi veya <1ms anlık havuz dağıtımı (`GET /api/v1/quiz/serve`)\n"
        "- 🛡️ **Hile Korumalı Değerlendirme:** Doğru cevapları istemciye sızdırmadan soru ID bazlı sunucu tarafı puanlama (`POST /api/v1/quiz/evaluate-answers`)\n"
        "- 🚫 **Anti-Spoiler Güvencesi:** Quizler kullanıcının okuduğu bölümün sınırlarına kilitlidir\n"
        "- 🤖 **Ollama Entegrasyonu:** qwen2.5:14b ve bge-m3 yerel modelleri"
    ),
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mobil ve web istemciler için CORS yapılandırması
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route'ları kaydet
app.include_router(health.router, prefix="/api/v1")
app.include_router(books.router, prefix="/api/v1")
app.include_router(quiz.router, prefix="/api/v1")       # Eski endpoint'ler (geriye dönük uyumluluk)
app.include_router(quizzes.router, prefix="/api/v1")    # Yeni RESTful endpoint
app.include_router(admin.router, prefix="/api/v1")      # Admin: metrikler
app.include_router(jobs.router, prefix="/api/v1")       # Asenkron üretim işleri


@app.get("/", include_in_schema=False)
def root():
    """Kök adrese gelindiğinde doğrudan Swagger dokümantasyonuna yönlendirir."""
    return RedirectResponse(url="/docs")


@app.get("/api/v1", tags=["General"])
def api_info():
    """API sürüm ve temel bilgilerini döndürür."""
    return {
        "name": "Context RAG Quiz API",
        "version": "1.0.0",
        "documentation": "/docs",
        "redoc": "/redoc",
        "status": "online",
    }
