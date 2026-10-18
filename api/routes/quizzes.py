"""
api/routes/quizzes.py
----------------------
RESTful quiz kaynağı endpoint'i.

POST /api/v1/quizzes  →  Quiz oluştur (kaynak türü: book)

Eski /api/v1/quiz/* endpoint'leri geriye dönük uyumluluk için korunmuştur.
"""

import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from config import CATALOG_DB_PATH
from api.schemas import (
    CreateQuizRequest,
    QuizQuestion,
    QuizResourceResponse,
)
from src.dataset_search import get_chunks_for_book
from src.quiz_generator import generate_quiz

router = APIRouter(prefix="/quizzes", tags=["Quizzes"])
logger = logging.getLogger(__name__)

# API'den gelen İngilizce difficulty değerlerini
# quiz_generator / prompts katmanının beklediği Türkçe karşılıklara map eder.
_DIFFICULTY_MAP: dict[str, str] = {
    "easy": "kolay",
    "medium": "orta",
    "hard": "zor",
}


def _resolve_book_metadata(book_id: str) -> tuple[str, str]:
    """
    Katalog veritabanından kitap başlığı ve yazarını döndürür.
    Katalog yoksa veya kitap bulunamazsa varsayılan değerler kullanılır.
    """
    title = f"Kitap #{book_id}"
    author = ""

    if os.path.isfile(CATALOG_DB_PATH):
        try:
            with sqlite3.connect(CATALOG_DB_PATH) as conn:
                row = conn.cursor().execute(
                    "SELECT title, author FROM books WHERE book_id = ?", (book_id,)
                ).fetchone()
                if row:
                    title = row[0] or title
                    author = row[1] or ""
        except Exception as exc:
            logger.warning(f"Katalogdan kitap metadata okunamadı (book_id={book_id}): {exc}")

    return title, author


def _make_quiz_id() -> str:
    """Kısa, insan-okunabilir quiz kimliği üretir. Örnek: quiz_3f7a1b2c"""
    return "quiz_" + uuid.uuid4().hex[:8]


@router.post(
    "",
    response_model=QuizResourceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Quiz oluştur",
    description=(
        "Belirtilen kaynaktan (kitap) çoktan seçmeli quiz üretir.\n\n"
        "- **source.type**: Şimdilik yalnızca `book` desteklenmektedir.\n"
        "- **source.book_id**: Project Gutenberg kitap ID'si. Başlık/yazar otomatik çözülür.\n"
        "- **source.chapter_index**: Opsiyonel. Belirli bir bölümden soru üretmek için 1-tabanlı bölüm numarası.\n"
        "- **question_count**: 1–10 arası soru sayısı (varsayılan: 5).\n"
        "- **difficulty**: `easy` | `medium` | `hard` (varsayılan: `medium`).\n\n"
        "Yanıt `201 Created` ile döner ve oluşturulan quiz'i `id` alanıyla birlikte içerir."
    ),
)
def create_quiz(req: CreateQuizRequest) -> QuizResourceResponse:
    """
    POST /api/v1/quizzes

    Kitap kaynağından RESTful quiz kaynağı oluşturur.
    - title / author katalogdan otomatik çözülür.
    - difficulty İngilizce API değerinden Türkçe prompt değerine dönüştürülür.
    - Yanıt 201 Created, body'de tam quiz objesi.
    """
    book_id = req.source.book_id
    chapter_index = req.source.chapter_index
    api_difficulty = req.difficulty  # "easy" | "medium" | "hard"
    internal_difficulty = _DIFFICULTY_MAP[api_difficulty]  # "kolay" | "orta" | "zor"

    # Kitap metadata'sını katalogdan çöz
    title, author = _resolve_book_metadata(book_id)

    # Chunk'ları getir
    try:
        chunks = get_chunks_for_book(
            book_id,
            title,
            author,
            max_chunks=20,
            chapter_index=chapter_index,
        )
    except ValueError as exc:
        logger.warning(f"Kitap parametre hatası (book_id={book_id}): {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"Kitap chunk'ları getirilemedi (book_id={book_id}): {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kitap metni işlenemedi: {exc}",
        )

    # Quiz üret
    try:
        quiz_chunks = [
            {
                "text": c["text"],
                "source_pages": c.get("source_pages", []),
                "source_chapter": c.get("source_chapter"),
                "progress_percent": c.get("progress_percent"),
            }
            for c in chunks
        ]
        raw_questions = generate_quiz(
            quiz_chunks,
            num_questions=req.question_count,
            difficulty=internal_difficulty,
        )
    except RuntimeError as exc:
        logger.error(f"Quiz üretimi başarısız (book_id={book_id}): {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz üretimi sırasında hata oluştu: {exc}",
        )
    except Exception as exc:
        logger.error(f"Quiz üretimi beklenmedik hata (book_id={book_id}): {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz üretimi sırasında beklenmedik hata: {exc}",
        )

    questions = [
        QuizQuestion(
            question=q["question"],
            options=q["options"],
            correct_answer=q["correct_answer"],
            explanation=q["explanation"],
            source_pages=q.get("source_pages", []),
            source_chapter=q.get("source_chapter"),
            progress_percent=q.get("progress_percent"),
        )
        for q in raw_questions
    ]

    return QuizResourceResponse(
        id=_make_quiz_id(),
        source_type="book",
        book_id=book_id,
        chapter_index=chapter_index,
        title=title,
        author=author or None,
        question_count=len(questions),
        difficulty=api_difficulty,
        status="completed",
        questions=questions,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
