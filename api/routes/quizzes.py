"""
api/routes/quizzes.py
----------------------
RESTful quiz kaynağı. Üretim uzun sürdüğü için asenkron çalışır:

POST /api/v1/quizzes       →  202 Accepted: quiz işi kuyruğa alınır, `id` ve `poll_url` döner
GET  /api/v1/quizzes/{id}  →  202 (queued/running), 200 completed (sorularla) veya 200 failed

Eski /api/v1/quiz/* endpoint'leri geriye dönük uyumluluk için korunmuştur.
"""

import logging
import os
import sqlite3

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from config import CATALOG_DB_PATH
from api.routes.jobs import RETRY_AFTER_SECONDS
from api.schemas import (
    CreateQuizRequest,
    QuizAcceptedResponse,
    QuizQuestion,
    QuizResourceResponse,
)
from src.dataset_search import get_book_chapters
from src.jobs import enqueue, get_job, queue_position

router = APIRouter(prefix="/quizzes", tags=["Quizzes"])
logger = logging.getLogger(__name__)


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


def _quiz_id(job_id: str) -> str:
    return f"quiz_{job_id}"


def _job_id(quiz_id: str) -> str:
    return quiz_id[len("quiz_"):] if quiz_id.startswith("quiz_") else quiz_id


def _pending_body(job: dict) -> dict:
    return QuizAcceptedResponse(
        id=_quiz_id(job["id"]),
        status=job["status"],
        poll_url=f"/api/v1/quizzes/{_quiz_id(job['id'])}",
        queue_position=queue_position(job["id"]) if job["status"] == "queued" else None,
        retry_after_seconds=RETRY_AFTER_SECONDS,
    ).model_dump()


@router.post(
    "",
    response_model=QuizAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Quiz oluştur (asenkron)",
    description=(
        "Belirtilen kaynaktan (kitap) çoktan seçmeli quiz üretimini kuyruğa alır ve hemen `202 Accepted` döner.\n\n"
        "- **source.type**: Şimdilik yalnızca `book` desteklenmektedir.\n"
        "- **source.book_id**: Project Gutenberg kitap ID'si. Başlık/yazar otomatik çözülür.\n"
        "- **source.chapter_index**: Opsiyonel. Belirli bir bölümden soru üretmek için 1-tabanlı bölüm numarası.\n"
        "- **question_count**: 1–10 arası soru sayısı (varsayılan: 5).\n"
        "- **difficulty**: `easy` | `medium` | `hard` (varsayılan: `medium`).\n\n"
        "Sonuç için yanıttaki `poll_url` (`GET /api/v1/quizzes/{id}`) yoklanır."
    ),
)
def create_quiz(req: CreateQuizRequest):
    book_id = req.source.book_id
    chapter_index = req.source.chapter_index
    title, author = _resolve_book_metadata(book_id)

    # Geçersiz bölüm numarası iş kuyruğa girmeden, hemen bildirilir
    if chapter_index is not None:
        try:
            total = get_book_chapters(book_id)["total_chapters"]
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Kitap metnine ulaşılamadı (ID: {book_id})")
        if chapter_index > total:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Geçersiz bölüm numarası: {chapter_index}. Bu kitapta toplam {total} bölüm bulunuyor.",
            )

    job = enqueue("quiz", {
        "book_id": book_id,
        "chapter_index": chapter_index,
        "title": title,
        "author": author,
        "question_count": req.question_count,
        "difficulty": req.difficulty.value,
    })
    return JSONResponse(status_code=202, headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                        content=_pending_body(job))


@router.get(
    "/{quiz_id}",
    response_model=QuizResourceResponse,
    summary="Quiz durumunu veya sonucunu getir",
    responses={202: {"model": QuizAcceptedResponse, "description": "Quiz hâlâ hazırlanıyor"}},
)
def get_quiz(quiz_id: str):
    """
    - `202`: iş kuyrukta veya çalışıyor (`status`: queued | running)
    - `200` + `status: completed`: quiz soruları ve cevaplarıyla
    - `200` + `status: failed`: `error` alanında sebep
    """
    job = get_job(_job_id(quiz_id))
    if not job or job["kind"] != "quiz":
        raise HTTPException(status_code=404, detail=f"Quiz bulunamadı: {quiz_id}")

    if job["status"] in ("queued", "running"):
        return JSONResponse(status_code=202, headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                            content=_pending_body(job))
    if job["status"] == "failed":
        return JSONResponse(status_code=200, content={**_pending_body(job), "error": job.get("error")})

    params = job["params"]
    questions = [QuizQuestion(**q) for q in (job.get("result") or {}).get("questions", [])]
    return QuizResourceResponse(
        id=_quiz_id(job["id"]),
        source_type="book",
        book_id=params["book_id"],
        chapter_index=params.get("chapter_index"),
        title=params.get("title") or f"Kitap #{params['book_id']}",
        author=params.get("author") or None,
        question_count=len(questions),
        difficulty=params.get("difficulty", "medium"),
        status="completed",
        questions=questions,
        created_at=str(job.get("finished_at") or job.get("created_at")),
    )
