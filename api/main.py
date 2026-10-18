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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Kök dizini PYTHONPATH'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import books, health, quiz, quizzes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="Context RAG - Quiz Generation API",
    description=(
        "Mobil ve web uygulamaları için RAG tabanlı kitap ve PDF quiz üretim API'si.\n\n"
        "- 📚 **Project Gutenberg Kataloğu:** 79,000+ kitapta anlık yerel arama ve quiz üretimi\n"
        "- 📄 **PDF Yükleme:** Contextual Retrieval ile PDF'lerden bağlamsal quiz üretimi\n"
        "- 🎯 **Otomatik Değerlendirme:** Kullanıcı cevaplarının puanlanması ve analizi\n"
        "- 🤖 **Ollama Entegrasyonu:** qwen2.5:14b ve bge-m3 yerel modelleri"
    ),
    version="1.0.0",
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
