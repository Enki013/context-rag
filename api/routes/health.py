"""
api/routes/health.py
--------------------
Sistem durumu, Ollama bağlantısı ve katalog metriklerini kontrol eden endpoint.
"""

import logging
import os
import sqlite3
import urllib.request
from fastapi import APIRouter
from config import CATALOG_DB_PATH, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL
from api.schemas import HealthResponse
from src.dataset_search import get_cache_status

router = APIRouter(prefix="/health", tags=["Health & Status"])
logger = logging.getLogger(__name__)


@router.get("", response_model=HealthResponse, summary="Sistem ve model sağlık kontrolü")
def check_health():
    """
    Mobil uygulamanın backend ve Ollama servislerine erişip erişemediğini kontrol eder.
    """
    ollama_ok = False
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/version")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                ollama_ok = True
    except Exception:
        ollama_ok = False

    cat_status = get_cache_status()

    return HealthResponse(
        status="ok" if ollama_ok and cat_status["exists"] else "degraded",
        ollama_connected=ollama_ok,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        catalog_installed=cat_status["exists"],
        total_books=cat_status["num_books"],
    )
