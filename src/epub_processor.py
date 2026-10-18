"""
src/epub_processor.py
----------------------
EPUB dosyalarından (PyMuPDF / fitz kullanarak) sayfa/bölüm bazlı metin,
metadata ve içindekiler (TOC) çıkaran modül.
"""

import logging
import os
import re
from typing import Any, Dict, List
import fitz

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """EPUB'dan çıkarılan ham metni temizler."""
    # Kontrol karakterlerini temizle (tab ve newline hariç)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Satır sonu tire bölmelerini birleştir: "keli-\nme" -> "kelime"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Tek newline'ları boşluğa çevir (paragraf = çift newline korunur)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 3+ newline'ları çifte indir
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Çoklu boşlukları teke indir
    text = re.sub(r" {2,}", " ", text)

    # Satır başı/sonu boşlukları temizle
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


def extract_epub_data(doc: fitz.Document, filename: str = "book.epub") -> Dict[str, Any]:
    """
    fitz.Document nesnesinden metin, metadata ve içindekiler tablosunu çıkarır.
    """
    meta = doc.metadata or {}
    title = meta.get("title") or filename
    author = meta.get("author") or ""

    total_pages = len(doc)
    pages: List[Dict[str, Any]] = []

    for p_idx in range(total_pages):
        page = doc[p_idx]
        raw = page.get_text("text")
        cleaned = _clean_text(raw)
        if cleaned:
            pages.append({
                "page": p_idx + 1,
                "text": cleaned,
            })

    # TOC (İçindekiler) çıkar
    raw_toc = doc.get_toc() or []  # [[lvl, title, page_num], ...]
    chapters: List[Dict[str, Any]] = []
    if raw_toc:
        for idx, item in enumerate(raw_toc):
            lvl, ch_title, start_p = item
            next_p = raw_toc[idx + 1][2] if idx + 1 < len(raw_toc) else total_pages + 1
            end_p = max(start_p, next_p - 1 if next_p > start_p else start_p)
            chapters.append({
                "chapter_index": idx + 1,
                "title": ch_title.strip() if ch_title else f"Bölüm {idx + 1}",
                "start_page": start_p,
                "end_page": end_p,
                "level": lvl,
            })

    # Her sayfanın ait olduğu bölümü ve ilerleme yüzdesini eşle
    for p in pages:
        p_num = p["page"]
        if chapters:
            matching = [ch for ch in chapters if ch["start_page"] <= p_num <= ch["end_page"]]
            if matching:
                p["source_chapter"] = matching[-1]["title"]
        if total_pages > 0:
            p["progress_percent"] = round((p_num / total_pages) * 100, 1)

    full_text = "\n\n".join(p["text"] for p in pages)

    return {
        "title": title,
        "author": author if author else None,
        "total_pages": total_pages,
        "pages": pages,
        "chapters": chapters,
        "full_text": full_text,
    }


def extract_text_from_epub_bytes(epub_bytes: bytes, filename: str = "book.epub") -> Dict[str, Any]:
    """Byte akışından EPUB verilerini çıkarır."""
    try:
        doc = fitz.open(stream=epub_bytes, filetype="epub")
        data = extract_epub_data(doc, filename=filename)
        doc.close()
        logger.info(
            f"EPUB'dan {len(data['pages'])} sayfa, {len(data['chapters'])} bölüm çıkarıldı: {filename}"
        )
        return data
    except Exception as e:
        logger.error(f"EPUB işleme hatası ({filename}): {e}", exc_info=True)
        raise


def extract_text_from_epub(epub_path: str) -> Dict[str, Any]:
    """Dosya yolundan EPUB verilerini çıkarır."""
    if not os.path.exists(epub_path):
        raise FileNotFoundError(f"EPUB dosyası bulunamadı: {epub_path}")
    try:
        doc = fitz.open(epub_path)
        data = extract_epub_data(doc, filename=os.path.basename(epub_path))
        doc.close()
        return data
    except Exception as e:
        logger.error(f"EPUB işleme hatası ({epub_path}): {e}", exc_info=True)
        raise


def get_full_text(pages: List[Dict[str, Any]]) -> str:
    """Sayfa listesini tek bir metin string'i olarak birleştirir."""
    return "\n\n".join(p["text"] for p in pages)
