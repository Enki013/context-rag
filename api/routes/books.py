"""
api/routes/books.py
-------------------
Kitap arama ve detaylarını getiren endpoint'ler.
"""

import logging
import os
import re
import sqlite3
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from config import BOOKS_DIR, CATALOG_DB_PATH
from api.security import require_admin
from api.schemas import (
    BookChapterCountResponse,
    BookChaptersResponse,
    BookDetailResponse,
    BookImportResponse,
    BookItem,
    BookSearchResponse,
    ChapterItem,
    EngineChapterItem,
)
from src.dataset_search import detect_chapters, extract_chapters, get_book_chapters, search_books
from src.epub_processor import extract_text_from_epub_bytes
from src.pdf_processor import extract_text_from_bytes, get_full_text
from src.question_bank import (
    get_book as get_engine_book,
    get_book_chapters as get_engine_chapters,
    refresh_gutenberg_book,
    register_book_content,
)

router = APIRouter(prefix="/books", tags=["Books"])
logger = logging.getLogger(__name__)


def _is_book_cached(book_id: str) -> bool:
    path = os.path.join(BOOKS_DIR, f"{book_id}.txt")
    return os.path.isfile(path) and os.path.getsize(path) > 100


@router.get("/search", response_model=BookSearchResponse, summary="Katalogda anlık kitap ara")
def search_books_endpoint(
    q: str = Query(..., min_length=1, description="Arama kelimesi (kitap adı, yazar, konu)"),
    limit: int = Query(20, ge=1, le=100, description="Döndürülecek maksimum kitap sayısı"),
):
    """
    Yerel SQLite FTS5 indeksi üzerinden **anlık (<1ms)** kitap araması yapar.
    Ağ araması veya indirme başlatmaz.
    """
    results = search_books(q, limit=limit)
    items = []
    for r in results:
        b_id = str(r["book_id"])
        items.append(
            BookItem(
                book_id=b_id,
                title=r["title"],
                author=r.get("author", "Bilinmiyor"),
                subjects=r.get("subjects", ""),
                bookshelves=r.get("bookshelves", ""),
                language=r.get("language", "en"),
                is_cached_locally=_is_book_cached(b_id),
            )
        )

    return BookSearchResponse(query=q, total=len(items), books=items)


@router.get("/{book_id}", response_model=BookDetailResponse, summary="Tek bir kitabın detaylarını getir")
def get_book_detail(book_id: str):
    """
    Belirli bir Gutenberg kitabının metadata ve yerel önbellek durumunu döner.
    Metin indirilmişse toplam bölüm sayısını (total_chapters) da içerir.
    """
    if not os.path.isfile(CATALOG_DB_PATH):
        raise HTTPException(status_code=503, detail="Kitap kataloğu henüz kurulmadı.")

    with sqlite3.connect(CATALOG_DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT book_id, title, author, subjects, bookshelves, language FROM books WHERE book_id = ?",
            (book_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Kitap bulunamadı (ID: {book_id})")

    cached = _is_book_cached(book_id)
    preview = None
    total_chaps = None
    if cached:
        try:
            with open(os.path.join(BOOKS_DIR, f"{book_id}.txt"), "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                preview = content[:500] + ("..." if len(content) > 500 else "")
                # Ağ isteği yapmadan: EPUB önbellekteyse içindekiler, değilse regex
                total_chaps = len(detect_chapters(book_id, content, book_title=row[1] or "", allow_download=False))
        except Exception:
            pass

    return BookDetailResponse(
        book_id=str(row[0]),
        title=row[1],
        author=row[2] or "Bilinmiyor",
        subjects=row[3] or "",
        bookshelves=row[4] or "",
        language=row[5] or "en",
        is_cached_locally=cached,
        text_preview=preview,
        total_chapters=total_chaps,
    )


@router.get("/{book_id}/chapter-count", response_model=BookChapterCountResponse, summary="Kitaptaki toplam bölüm ve indeks listesini getir")
def get_book_chapter_count_endpoint(book_id: str):
    """
    Kitabın kaç bölümden oluştuğunu ve quiz üretiminde kullanılabilecek
    bölüm indekslerini ([1, 2, ..., N]) döndürür.
    Mobil uygulamada bölüm seçici (dropdown / picker) doldurmak için kullanılır.
    """
    if not os.path.isfile(CATALOG_DB_PATH):
        raise HTTPException(status_code=503, detail="Kitap kataloğu henüz kurulmadı.")

    with sqlite3.connect(CATALOG_DB_PATH) as conn:
        row = conn.cursor().execute(
            "SELECT book_id, title, author FROM books WHERE book_id = ?", (book_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Kitap bulunamadı (ID: {book_id})")

    try:
        data = get_book_chapters(book_id)
        count = data["total_chapters"]
        return BookChapterCountResponse(
            book_id=data["book_id"],
            title=data["title"],
            total_chapters=count,
            chapter_indices=list(range(1, count + 1)),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Kitap metnine ulaşılamadı (ID: {book_id})")
    except Exception as e:
        logger.error(f"Bölüm sayısı alınırken hata oluştu ({book_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bölüm sayısı hesaplanırken hata oluştu: {e}")


@router.post(
    "/import",
    response_model=BookImportResponse,
    summary="EPUB veya PDF kitabını sisteme kaydet ve bölümlerini ayrıştır (Admin)",
    dependencies=[Depends(require_admin)],
)
def import_book_endpoint(
    file: UploadFile = File(..., description="Yüklenecek EPUB veya PDF dosyası"),
    book_id: Optional[str] = Form(None, description="Özel kitap ID (boş bırakılırsa dosya adından üretilir)"),
    title: Optional[str] = Form(None, description="Kitap adı (boşsa dosyadan çıkarılır)"),
    author: Optional[str] = Form(None, description="Yazar adı"),
    prepopulate: bool = Form(False, description="true ise her bölümün soru havuzu arka planda doldurulur (medium)"),
):
    """
    Mobil backend'in yeni bir kitabı (EPUB / PDF) bu quiz motoruna kaydetmesi için kullanılır.
    Kitabın bölümlerini, sayfa aralıklarını ve metinlerini çıkararak kalıcı soru bankası veritabanına indeksler.
    """
    filename = file.filename.lower()
    if not (filename.endswith(".epub") or filename.endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Yalnızca .epub ve .pdf dosyaları desteklenir.")

    contents = file.file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Dosya içeriği boş.")

    fmt = "epub" if filename.endswith(".epub") else "pdf"
    clean_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", os.path.splitext(file.filename)[0]).strip("_")
    assigned_book_id = (book_id or f"book_{clean_name}").strip()

    chapters = []
    total_pages = 0
    detected_title = title or file.filename
    detected_author = author

    if fmt == "epub":
        epub_data = extract_text_from_epub_bytes(contents, file.filename)
        detected_title = title or epub_data.get("title") or file.filename
        detected_author = author or epub_data.get("author")
        total_pages = epub_data.get("total_pages", 0)
        pages = epub_data.get("pages", [])
        raw_chapters = epub_data.get("chapters", [])

        if raw_chapters:
            for ch in raw_chapters:
                sp = ch.get("start_page", 1)
                ep = ch.get("end_page", total_pages)
                ch_pages = [p["text"] for p in pages if sp <= p["page"] <= ep]
                ch_text = "\n\n".join(ch_pages)
                if not ch_text.strip() and pages:
                    ch_text = pages[min(sp - 1, len(pages) - 1)]["text"]
                progress_pct = round((sp / max(1, total_pages)) * 100, 1)
                chapters.append({
                    "chapter_index": ch["chapter_index"],
                    "title": ch["title"],
                    "start_page": sp,
                    "end_page": ep,
                    "progress_percent": progress_pct,
                    "raw_text": ch_text,
                })
        else:
            full_text = epub_data.get("full_text", "")
            chunk_len = max(1, len(full_text) // 4)
            for idx in range(4):
                c_text = full_text[idx * chunk_len : (idx + 1) * chunk_len]
                chapters.append({
                    "chapter_index": idx + 1,
                    "title": f"Bölüm {idx + 1}",
                    "start_page": idx * max(1, total_pages // 4) + 1,
                    "end_page": (idx + 1) * max(1, total_pages // 4),
                    "progress_percent": round(idx * 25.0, 1),
                    "raw_text": c_text,
                })
    else:
        pages = extract_text_from_bytes(contents, file.filename)
        total_pages = len(pages)
        full_text = get_full_text(pages)
        parsed_chapters = extract_chapters(full_text)
        for ch in parsed_chapters:
            chapters.append({
                "chapter_index": ch["chapter_index"],
                "title": ch["title"],
                "start_page": 1,
                "end_page": total_pages,
                "progress_percent": ch["progress_percent"],
                "raw_text": ch.get("text", ""),
            })

    register_book_content(
        book_id=assigned_book_id,
        title=detected_title,
        author=detected_author,
        source_format=fmt,
        chapters=chapters,
        total_pages=total_pages,
    )

    queued_jobs = 0
    if prepopulate:
        from src.jobs import enqueue

        for ch in chapters:
            enqueue(
                "populate",
                {"book_id": assigned_book_id, "chapter_index": ch["chapter_index"], "difficulty": "medium",
                 "target_count": 10},
                dedup_key=f"populate:{assigned_book_id}:{ch['chapter_index']}:medium",
            )
            queued_jobs += 1

    db_chapters = get_engine_chapters(assigned_book_id)
    chapter_items = [
        EngineChapterItem(
            chapter_index=c["chapter_index"],
            title=c["title"],
            start_page=c.get("start_page"),
            end_page=c.get("end_page"),
            progress_percent=c.get("progress_percent", 0.0),
            char_count=c.get("char_count", 0),
            pool_count=c.get("pool_count", 0),
        )
        for c in db_chapters
    ]

    return BookImportResponse(
        book_id=assigned_book_id,
        title=detected_title,
        author=detected_author,
        source_format=fmt,
        total_chapters=len(chapter_items),
        total_pages=total_pages,
        chapters=chapter_items,
        queued_jobs=queued_jobs,
    )


@router.get("/{book_id}/chapters", response_model=BookChaptersResponse, summary="Kitabın bölümlerini ve ilerleme yüzdelerini getir")
def get_book_chapters_endpoint(book_id: str):
    """
    Belirtilen kitabın metnindeki tüm bölümleri ve her bölümün kitap içindeki ilerleme yüzdesini döndürür.
    Önce kalıcı soru bankası veritabanına bakar, yoksa Gutenberg kataloğundan çeker.
    """
    # Eski yöntemle kaydedilmiş Gutenberg kitaplarını güncel bölüm tespitiyle eşitle
    refresh_gutenberg_book(book_id)
    engine_chapters = get_engine_chapters(book_id)
    if engine_chapters:
        engine_book = get_engine_book(book_id) or {}
        chapter_items = [
            ChapterItem(
                chapter_index=c["chapter_index"],
                title=c["title"],
                progress_percent=c["progress_percent"],
                char_count=c["char_count"],
            )
            for c in engine_chapters
        ]
        return BookChaptersResponse(
            book_id=book_id,
            title=engine_book.get("title", f"Kitap #{book_id}"),
            author=engine_book.get("author"),
            total_chapters=len(chapter_items),
            chapters=chapter_items,
        )

    if not os.path.isfile(CATALOG_DB_PATH):
        raise HTTPException(status_code=503, detail="Kitap kataloğu henüz kurulmadı.")

    with sqlite3.connect(CATALOG_DB_PATH) as conn:
        row = conn.cursor().execute(
            "SELECT book_id, title, author FROM books WHERE book_id = ?", (book_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Kitap bulunamadı (ID: {book_id})")

    try:
        data = get_book_chapters(book_id)
        chapter_items = [
            ChapterItem(
                chapter_index=c["chapter_index"],
                title=c["title"],
                progress_percent=c["progress_percent"],
                char_count=c["char_count"],
            )
            for c in data["chapters"]
        ]
        return BookChaptersResponse(
            book_id=data["book_id"],
            title=data["title"],
            author=data.get("author"),
            total_chapters=data["total_chapters"],
            chapters=chapter_items,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Kitap metnine ulaşılamadı (ID: {book_id}). Bu Gutenberg eseri sesli kitap (audiobook) veya desteklenmeyen formatta olabilir. Lütfen metin formatındaki bir kitabı seçin (örn: Frankenstein için ID 84).",
        )
    except Exception as e:
        logger.error(f"Bölümler çıkarılırken hata oluştu ({book_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bölümler işlenirken hata oluştu: {e}")
