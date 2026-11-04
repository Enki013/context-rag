"""
api/routes/quiz.py
------------------
Kitap ve PDF üzerinden quiz üretimi ve cevap değerlendirme endpoint'leri.
"""

import logging
import os
import shutil
import sqlite3
from typing import Annotated, List, Optional
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from config import CATALOG_DB_PATH, MAX_FILE_SIZE_MB, UPLOAD_DIR
from api.schemas import (
    BankAnswerItem,
    BankEvaluationRequest,
    BankEvaluationResponse,
    BookQuizRequest,
    EngineServeQuizResponse,
    EngineServedQuestion,
    JobResponse,
    PreparingResponse,
    QuestionEvaluationResult,
    QuizEvaluationRequest,
    QuizEvaluationResponse,
    QuizQuestion,
    QuizResponse,
)
from api.routes.jobs import RETRY_AFTER_SECONDS, job_payload
from api.security import is_admin, require_admin
from src.chunker import chunk_text
from src.difficulty import Difficulty, DifficultyParam
from src.dataset_search import get_chunks_for_book
from src.pdf_processor import extract_text_from_bytes, get_full_text
from src.epub_processor import extract_text_from_epub_bytes
from src.jobs import enqueue
from src.question_bank import evaluate_answers_against_bank, get_pool_quiz
from src.metrics import record_generation_run
from src.quiz_generator import generate_quiz

router = APIRouter(prefix="/quiz", tags=["Quiz Operations"])
logger = logging.getLogger(__name__)


@router.post("/generate-from-book", response_model=QuizResponse, summary="Katalogdaki bir kitaptan quiz üret")
def generate_quiz_from_book(req: BookQuizRequest):
    """
    Gutenberg ID'si verilen kitaptan RAG pipeline'ı ile çoktan seçmeli quiz üretir.
    Kitabı tanımlamak için yalnızca `book_id` gönderilmesi yeterlidir (title ve author otomatik bulunur).
    Kitap metni önceden diske indirilmişse anında okunur, değilse otomatik olarak önbelleğe alınır.
    İsteğe bağlı olarak tek bir bölüm (chapter_index) hedeflenebilir.
    """
    book_id = req.book_id
    title = req.title
    author = req.author or ""

    # Title veya author boş ise SQLite katalog veritabanından otomatik çek
    if not title:
        if os.path.isfile(CATALOG_DB_PATH):
            try:
                with sqlite3.connect(CATALOG_DB_PATH) as conn:
                    row = conn.cursor().execute(
                        "SELECT title, author FROM books WHERE book_id = ?", (book_id,)
                    ).fetchone()
                    if row:
                        title = row[0]
                        if not author and row[1]:
                            author = row[1]
            except Exception as e:
                logger.warning(f"Katalogdan kitap metadata okunamadı: {e}")
        if not title:
            title = f"Kitap #{book_id}"

    try:
        chunks = get_chunks_for_book(
            book_id,
            title,
            author,
            max_chunks=20,
            chapter_index=req.chapter_index,
        )
    except ValueError as ve:
        logger.warning(f"Kitap parametre hatası ({book_id}): {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Kitap chunk'ları getirilemedi ({book_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Kitap metni işlenemedi: {e}")

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
        stats: dict = {}
        try:
            raw_quiz = generate_quiz(
                quiz_chunks,
                num_questions=req.num_questions,
                difficulty=req.difficulty.value,
                question_type=req.question_type,
                stats=stats,
            )
        finally:
            record_generation_run(stats, "book", book_id, req.chapter_index)
    except Exception as e:
        logger.error(f"Quiz üretilemedi: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quiz üretimi sırasında hata oluştu: {e}")

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
        for q in raw_quiz
    ]

    return QuizResponse(
        source_type="book",
        identifier=book_id,
        title=title,
        author=author,
        total_questions=len(questions),
        questions=questions,
    )


@router.post("/generate-from-pdf", response_model=QuizResponse, summary="Yüklenen bir PDF'ten quiz üret")
def generate_quiz_from_pdf(
    file: UploadFile = File(..., description="Quiz üretilecek PDF dosyası"),
    num_questions: int = Form(5, ge=1, le=10, description="Soru sayısı (1-10)"),
    difficulty: Annotated[DifficultyParam, Form(description="easy | medium | hard")] = Difficulty.MEDIUM,
    question_type: str = Form("multiple_choice", description="Soru türü"),
):
    """
    Mobil cihazdan yüklenen bir PDF dosyasını okur, chunk'lara böler ve quiz üretir.

    Not: Senkron `def` olarak tanımlı; böylece uzun süren PDF işleme ve LLM çağrıları
    FastAPI thread havuzunda çalışır ve event loop'u (diğer istekleri) bloklamaz.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Yalnızca .pdf uzantılı dosyalar kabul edilir.")

    contents = file.file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Dosya boyutu çok büyük ({file_size_mb:.1f} MB). İzin verilen maksimum: {MAX_FILE_SIZE_MB} MB",
        )

    # 1. Metin çıkarma
    pages = extract_text_from_bytes(contents, file.filename)
    full_text = get_full_text(pages)
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="PDF dosyasından metin çıkarılamadı (taranmış görsel olabilir).")

    # 2. Chunking
    chunks = chunk_text(full_text, pages=pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="Metin parçalanamadı.")

    # 3. Quiz üretimi
    # (Önceden burada özet + chunk başına LLM bağlamı + embedding üretilip diske yazılıyordu,
    #  ama hiçbir zaman sorgulanmıyordu. Retrieval eklenene kadar bu iş yapılmıyor.)
    stats: dict = {}
    try:
        raw_quiz = generate_quiz(
            chunks,
            num_questions=num_questions,
            difficulty=difficulty.value,
            question_type=question_type,
            stats=stats,
        )
    finally:
        record_generation_run(stats, "pdf", file.filename)

    questions = [
        QuizQuestion(
            question=q["question"],
            options=q["options"],
            correct_answer=q["correct_answer"],
            explanation=q["explanation"],
            source_pages=q.get("source_pages", []),
        )
        for q in raw_quiz
    ]

    return QuizResponse(
        source_type="pdf",
        identifier=file.filename,
        title=file.filename,
        author=None,
        total_questions=len(questions),
        questions=questions,
    )


@router.post("/generate-from-epub", response_model=QuizResponse, summary="Yüklenen bir EPUB'tan quiz üret")
def generate_quiz_from_epub(
    file: UploadFile = File(..., description="Quiz üretilecek EPUB dosyası"),
    num_questions: int = Form(5, ge=1, le=10, description="Soru sayısı (1-10)"),
    difficulty: Annotated[DifficultyParam, Form(description="easy | medium | hard")] = Difficulty.MEDIUM,
    question_type: str = Form("multiple_choice", description="Soru türü"),
):
    """
    Mobil veya web istemcisinden yüklenen bir EPUB dosyasını okur,
    gömülü içindekiler (TOC) ve sayfaları çıkarır, chunk'lara böler ve quiz üretir.
    """
    if not file.filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="Yalnızca .epub uzantılı dosyalar kabul edilir.")

    contents = file.file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Dosya boyutu çok büyük ({file_size_mb:.1f} MB). İzin verilen maksimum: {MAX_FILE_SIZE_MB} MB",
        )

    # 1. Metin ve metadata çıkarma
    try:
        epub_data = extract_text_from_epub_bytes(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"EPUB dosyası okunamadı: {e}")

    full_text = epub_data.get("full_text", "")
    pages = epub_data.get("pages", [])
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="EPUB dosyasından metin çıkarılamadı.")

    # 2. Chunking
    chunks = chunk_text(full_text, pages=pages)
    if not chunks:
        raise HTTPException(status_code=400, detail="Metin parçalanamadı.")

    # 3. Quiz üretimi
    # (Önceden burada özet + chunk başına LLM bağlamı + embedding üretilip diske yazılıyordu,
    #  ama hiçbir zaman sorgulanmıyordu. Retrieval eklenene kadar bu iş yapılmıyor.)
    stats: dict = {}
    try:
        raw_quiz = generate_quiz(
            chunks,
            num_questions=num_questions,
            difficulty=difficulty.value,
            question_type=question_type,
            stats=stats,
        )
    finally:
        record_generation_run(stats, "epub", file.filename)

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
        for q in raw_quiz
    ]

    return QuizResponse(
        source_type="epub",
        identifier=file.filename,
        title=epub_data.get("title") or file.filename,
        author=epub_data.get("author"),
        total_questions=len(questions),
        questions=questions,
    )



@router.post("/evaluate", response_model=QuizEvaluationResponse, summary="Kullanıcı cevaplarını değerlendir")
def evaluate_quiz(req: QuizEvaluationRequest):
    """
    Kullanıcının seçtiği şıkları orijinal quiz ile karşılaştırır.
    Doğru/yanlış adetlerini, başarı oranını ve her soru için açıklayıcı geri bildirimi döner.
    Mobil uygulamanın yerel değerlendirme mantığı yazmasına gerek kalmaz.
    """
    answer_map = {ans.question_index: ans.selected_option.strip() for ans in req.user_answers}
    correct_count = 0
    details = []

    for idx, q in enumerate(req.quiz):
        user_sel = answer_map.get(idx, "")
        correct_letter = q.correct_answer.strip().upper()

        # Doğru şıkkın metnini bul ("A) ..." gibi)
        correct_option_text = next(
            (opt for opt in q.options if opt.startswith(f"{correct_letter})")),
            correct_letter,
        )

        # Kullanıcı harfi veya şıkkın tamamını göndermiş olabilir
        is_correct = False
        if user_sel:
            if user_sel.upper() == correct_letter or user_sel.startswith(f"{correct_letter})"):
                is_correct = True
                correct_count += 1

        details.append(
            QuestionEvaluationResult(
                question_index=idx,
                question=q.question,
                user_selected=user_sel or "Cevaplanmadı",
                correct_answer=correct_letter,
                correct_option=correct_option_text,
                is_correct=is_correct,
                explanation=q.explanation,
            )
        )

    total = len(req.quiz)
    pct = (correct_count / total * 100.0) if total > 0 else 0.0

    if pct >= 80:
        badge = "Harika!"
    elif pct >= 50:
        badge = "İyi"
    else:
        badge = "Daha Çok Çalışmalısın"

    return QuizEvaluationResponse(
        total_questions=total,
        correct_count=correct_count,
        score_percentage=round(pct, 1),
        result_badge=badge,
        details=details,
    )


# ─── Headless Question Bank & Mobile Quiz Serving ──────────────────────────


POOL_TARGET = 10  # havuz boşsa arka planda bu kadar soru hazırlanır


def _populate_job(book_id: str, chapter_index: int, difficulty: str, target_count: int) -> dict:
    return enqueue(
        "populate",
        {"book_id": book_id, "chapter_index": chapter_index, "difficulty": difficulty, "target_count": target_count},
        dedup_key=f"populate:{book_id}:{chapter_index}:{difficulty}",
    )


@router.get(
    "/serve",
    response_model=EngineServeQuizResponse,
    summary="Mobil uygulama için bölüm quizi getir (Soru Bankası)",
    responses={202: {"model": PreparingResponse, "description": "Havuz hazırlanıyor; poll_url'i yoklayın"}},
)
def serve_quiz_endpoint(
    book_id: str = Query(..., description="Kitap ID (örn: book_kurk_mantolu_madonna veya 1661)"),
    chapter_index: int = Query(1, ge=1, description="1-tabanlı bölüm numarası"),
    count: int = Query(5, ge=1, le=20, description="Döndürülecek soru sayısı (1-20)"),
    difficulty: Annotated[DifficultyParam, Query(description="easy | medium | hard")] = Difficulty.MEDIUM,
    hide_answers: bool = Query(True, description="Doğru şık ve açıklamayı gizle (hile koruması). `false` yalnızca X-Admin-Key ile."),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
):
    """
    Mobil istemciler için soru bankası üzerinden quiz dağıtır. LLM çağrısı yapmaz.
    1. Havuzda yeterli doğrulanmış soru varsa hemen döner (`200`, `cache_hit: true`).
    2. Yoksa havuzu dolduracak bir arka plan işi kuyruğa alınır ve `202` döner
       (`job_id`, `poll_url`, `retry_after_seconds`). İstemci `poll_url`'i yoklar, iş bitince
       aynı isteği tekrarlar. Aynı bölüm için tekrar gelen istekler aynı işe bağlanır.
    3. `hide_answers=true` (varsayılan) ile doğru şık mobil kullanıcıya sızdırılmaz.
       Cevapları görmek (`hide_answers=false`) yalnızca geçerli `X-Admin-Key` ile mümkündür.
    """
    if not hide_answers and not is_admin(x_admin_key):
        raise HTTPException(status_code=403, detail="Cevaplı quiz yalnızca admin anahtarıyla alınabilir.")
    try:
        data = get_pool_quiz(
            book_id=book_id,
            chapter_index=chapter_index,
            count=count,
            difficulty=difficulty.value,
            hide_answers=hide_answers,
        )
        if data is None:
            job = _populate_job(book_id, chapter_index, difficulty.value, max(count, POOL_TARGET))
            return JSONResponse(
                status_code=202,
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
                content=PreparingResponse(
                    job_id=job["id"],
                    poll_url=f"/api/v1/jobs/{job['id']}",
                    retry_after_seconds=RETRY_AFTER_SECONDS,
                    message="Bu bölümün soruları hazırlanıyor. Biraz sonra tekrar deneyin.",
                ).model_dump(),
            )
        return EngineServeQuizResponse(
            book_id=data["book_id"],
            title=data["title"],
            author=data.get("author"),
            chapter_index=data["chapter_index"],
            chapter_title=data["chapter_title"],
            difficulty=data["difficulty"],
            cache_hit=data["cache_hit"],
            total_pool_size=data["total_pool_size"],
            question_count=data["question_count"],
            questions=[
                EngineServedQuestion(
                    id=q["id"],
                    question=q["question"],
                    options=q["options"],
                    source_pages=q.get("source_pages", []),
                    source_chapter=q.get("source_chapter"),
                    progress_percent=q.get("progress_percent"),
                    correct_answer=q.get("correct_answer"),
                    explanation=q.get("explanation"),
                )
                for q in data["questions"]
            ],
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Quiz servis hatası ({book_id}, ch={chapter_index}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Quiz servis edilirken hata oluştu: {e}")


@router.post("/evaluate-answers", response_model=BankEvaluationResponse, summary="Soru ID'leri üzerinden hile korumalı cevap değerlendirme")
def evaluate_bank_answers_endpoint(req: BankEvaluationRequest):
    """
    Mobil kullanıcının cevaplarını soru ID'leri üzerinden veritabanında doğrular.
    Mobil istemcinin doğru şıkkı bilmesine veya göndermesine gerek kalmaz.
    """
    try:
        answers_payload = [
            {"question_id": a.question_id, "selected_option": a.selected_option}
            for a in req.answers
        ]
        result = evaluate_answers_against_bank(answers_payload)
        return BankEvaluationResponse(
            total_questions=result["total_questions"],
            correct_count=result["correct_count"],
            score_percentage=result["score_percentage"],
            result_badge=result["result_badge"],
            details=result["details"],
        )
    except Exception as e:
        logger.error(f"Cevap değerlendirme hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Değerlendirme sırasında hata oluştu: {e}")


@router.post(
    "/populate",
    status_code=202,
    response_model=JobResponse,
    summary="Bölüm için soru havuzunu önceden doldur (Arka Plan / Admin)",
    dependencies=[Depends(require_admin)],
)
def populate_pool_endpoint(
    book_id: str = Query(..., description="Kitap ID"),
    chapter_index: int = Query(..., ge=1, description="Bölüm numarası"),
    target_count: int = Query(10, ge=1, le=30, description="Havuzda olması hedeflenen toplam soru sayısı"),
    difficulty: Annotated[DifficultyParam, Query(description="easy | medium | hard")] = Difficulty.MEDIUM,
):
    """
    Bölüm havuzunu `target_count` soruya kadar dolduracak bir arka plan işi kuyruğa alır (`202`).
    Havuz kısmen doluysa yalnızca eksik kadar soru üretilir. İlerleme: `GET /api/v1/jobs/{id}`.
    """
    return job_payload(_populate_job(book_id, chapter_index, difficulty.value, target_count))


@router.post(
    "/reverify",
    status_code=202,
    response_model=JobResponse,
    summary="Doğrulanmamış havuz sorularını yeniden doğrula (Arka Plan / Admin)",
    dependencies=[Depends(require_admin)],
)
def reverify_pool_endpoint(
    book_id: Optional[str] = Query(None, description="Boşsa tüm kitaplar"),
    chapter_index: Optional[int] = Query(None, ge=1, description="Boşsa kitabın tüm bölümleri"),
):
    """
    Doğrulama eklenmeden önce üretilmiş (kanıtı olmayan) havuz sorularını doğrular.
    Her soru için bölüm metninden en ilgili parçalar bge-m3 ile bulunur ve doğrulayıcıya verilir.
    Geçenlere kanıt yazılır ve sunulmaya başlar; geçemeyenler emekliye ayrılır (silinmez).
    """
    params = {"book_id": book_id, "chapter_index": chapter_index}
    return job_payload(enqueue("reverify", params, dedup_key=f"reverify:{book_id}:{chapter_index}"))

