"""
api/routes/books.py
-------------------
Kitap arama ve detaylarını getiren endpoint'ler.
"""

import logging
import os
import sqlite3
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from config import BOOKS_DIR, CATALOG_DB_PATH
from api.schemas import (
    BookChapterCountResponse,
    BookChaptersResponse,
    BookDetailResponse,
    BookItem,
    BookSearchResponse,
    ChapterItem,
)
from src.dataset_search import extract_chapters, get_book_chapters, search_books

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
                total_chaps = len(extract_chapters(content))
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


@router.get("/{book_id}/chapters", response_model=BookChaptersResponse, summary="Kitabın bölümlerini ve ilerleme yüzdelerini getir")
def get_book_chapters_endpoint(book_id: str):
    """
    Belirtilen Gutenberg kitabının metnindeki tüm bölümleri (Chapter, Letter, Scene vb.)
    ve her bölümün kitap içindeki ilerleme yüzdesini (%0 - %100) tespit edip döndürür.
    Mobil uygulamanın içindekiler tablosu (Table of Contents) ve okuma ilerlemesi (reading progress)
    için doğrudan kullanılabilir.
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
        raise HTTPException(status_code=404, detail=f"Kitap metnine ulaşılamadı (ID: {book_id})")
    except Exception as e:
        logger.error(f"Bölümler çıkarılırken hata oluştu ({book_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bölümler işlenirken hata oluştu: {e}")
