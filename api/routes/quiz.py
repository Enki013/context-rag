"""
api/routes/quiz.py
------------------
Kitap ve PDF üzerinden quiz üretimi ve cevap değerlendirme endpoint'leri.
"""

import logging
import os
import shutil
import sqlite3
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from config import CATALOG_DB_PATH, MAX_FILE_SIZE_MB, UPLOAD_DIR
from api.schemas import (
    BookQuizRequest,
    QuestionEvaluationResult,
    QuizEvaluationRequest,
    QuizEvaluationResponse,
    QuizQuestion,
    QuizResponse,
)
from src.chunker import chunk_text
from src.contextual_chunker import contextualize_chunks, generate_summary
from src.dataset_search import get_chunks_for_book
from src.embedder import generate_collection_name, get_all_chunks, store_chunks
from src.pdf_processor import extract_text_from_bytes, get_full_text
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
        raw_quiz = generate_quiz(
            quiz_chunks,
            num_questions=req.num_questions,
            difficulty=req.difficulty,
            question_type=req.question_type,
        )
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
async def generate_quiz_from_pdf(
    file: UploadFile = File(..., description="Quiz üretilecek PDF dosyası"),
    num_questions: int = Form(5, ge=1, le=10, description="Soru sayısı (1-10)"),
    difficulty: str = Form("orta", description="kolay | orta | zor"),
    question_type: str = Form("multiple_choice", description="Soru türü"),
):
    """
    Mobil cihazdan yüklenen bir PDF dosyasını okur, Contextual Retrieval ile
    chunk'lara bağlam ekler, Chroma vektör veritabanına kaydeder ve quiz üretir.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Yalnızca .pdf uzantılı dosyalar kabul edilir.")

    contents = await file.read()
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

    # 3. Contextual Retrieval
    summary = generate_summary(full_text)
    ctx_chunks = contextualize_chunks(chunks, summary)

    # 4. Vektör deposuna kaydetme
    collection_name = generate_collection_name(file.filename)
    store_chunks(ctx_chunks, collection_name)

    # 5. Quiz üretimi
    all_stored_chunks = get_all_chunks(collection_name, with_metadata=True)
    raw_quiz = generate_quiz(
        all_stored_chunks,
        num_questions=num_questions,
        difficulty=difficulty,
        question_type=question_type,
    )

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
        badge = "Harika! 🏆"
    elif pct >= 50:
        badge = "İyi 👍"
    else:
        badge = "Daha Çok Çalışmalısın 📚"

    return QuizEvaluationResponse(
        total_questions=total,
        correct_count=correct_count,
        score_percentage=round(pct, 1),
        result_badge=badge,
        details=details,
    )
