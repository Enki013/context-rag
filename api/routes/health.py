"""
api/routes/health.py
--------------------
Sistem durumu, Ollama bağlantısı ve katalog metriklerini kontrol eden endpoint.
"""

import json
import logging
import urllib.request
from typing import Optional

from fastapi import APIRouter

import config
from config import EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL
from api.schemas import HealthResponse
from src.dataset_search import get_cache_status

router = APIRouter(prefix="/health", tags=["Health & Status"])
logger = logging.getLogger(__name__)


def _installed_models() -> Optional[set[str]]:
    """Ollama'da kurulu modellerin adlarını döner; Ollama'ya ulaşılamazsa None."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            models = json.load(resp).get("models", [])
    except Exception:
        return None
    names = set()
    for m in models:
        name = m.get("name") or m.get("model") or ""
        names.add(name)
        if name.endswith(":latest"):
            names.add(name[: -len(":latest")])
    return names


@router.get("", response_model=HealthResponse, summary="Sistem ve model sağlık kontrolü")
def check_health():
    """
    Mobil uygulamanın backend ve Ollama servislerine erişip erişemediğini kontrol eder.
    Doğrulama açıkken doğrulayıcı model kurulu değilse durum "degraded" olur
    (bu durumda üretilen sorular doğrulanamaz ve atılır).
    """
    installed = _installed_models()
    ollama_ok = installed is not None

    verify_installed = None
    if ollama_ok and config.VERIFY_ANSWERS:
        verify_installed = config.VERIFY_MODEL in installed
        if not verify_installed:
            logger.warning(
                f"Doğrulayıcı model kurulu değil: {config.VERIFY_MODEL}. "
                f"`ollama pull {config.VERIFY_MODEL}` çalıştırın veya QUIZ_VERIFY_MODEL ayarlayın."
            )

    cat_status = get_cache_status()
    healthy = ollama_ok and cat_status["exists"] and verify_installed is not False

    return HealthResponse(
        status="ok" if healthy else "degraded",
        ollama_connected=ollama_ok,
        llm_model=LLM_MODEL,
        embedding_model=EMBEDDING_MODEL,
        catalog_installed=cat_status["exists"],
        total_books=cat_status["num_books"],
        verify_answers=config.VERIFY_ANSWERS,
        verify_model=config.VERIFY_MODEL if config.VERIFY_ANSWERS else None,
        verify_model_installed=verify_installed,
    )
