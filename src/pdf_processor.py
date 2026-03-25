import re
import fitz
import logging
import os

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    """
    PDF'den çıkarılan ham metni temizler.
    - Satır sonu tire bölmelerini birleştirir (ör: "kara-\nkter" -> "karakter")
    - Gereksiz satır kırılmalarını kaldırır (paragraf kırılmalarını korur)
    - Çoklu boşlukları teke indirir
    - Kontrol karakterlerini temizler
    - Sayfa üst/alt bilgilerindeki tekrarlayan satırları kaldırır
    """
    # Kontrol karakterlerini temizle (tab ve newline hariç)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Satır sonu tire bölmelerini birleştir: "keli-\nme" -> "kelime"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Tek newline'ları boşluğa çevir (paragraf = çift newline korunur)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 3+ newline'ları çifte indir
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Çoklu boşlukları teke indir
    text = re.sub(r' {2,}', ' ', text)

    # Satır başı/sonu boşlukları temizle
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    PDF dosyasından sayfa bazlı metin çıkarır ve temizler.

    Args:
        pdf_path: PDF dosyasının yolu

    Returns:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_path}")

    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned = _clean_text(text)
            if cleaned:
                pages.append({
                    "page": page_num + 1,
                    "text": cleaned
                })
        doc.close()
        logger.info(f"PDF'den {len(pages)} sayfa metin çıkarıldı: {pdf_path}")
    except Exception as e:
        logger.error(f"PDF işleme hatası: {e}")
        raise

    return pages


def extract_text_from_bytes(pdf_bytes: bytes, filename: str = "upload.pdf") -> list[dict]:
    """
    Byte olarak verilen PDF'den sayfa bazlı metin çıkarır ve temizler.
    Streamlit file_uploader ile uyumlu.

    Args:
        pdf_bytes: PDF dosyasının byte içeriği
        filename: Loglama için dosya adı

    Returns:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned = _clean_text(text)
            if cleaned:
                pages.append({
                    "page": page_num + 1,
                    "text": cleaned
                })
        doc.close()
        logger.info(f"PDF'den {len(pages)} sayfa metin çıkarıldı: {filename}")
    except Exception as e:
        logger.error(f"PDF işleme hatası ({filename}): {e}")
        raise

    return pages


def get_full_text(pages: list[dict]) -> str:
    """
    Sayfa bazlı metin listesini tek bir string olarak birleştirir.

    Args:
        pages: extract_text_from_pdf çıktısı

    Returns:
        Tüm sayfaların metni birleşik string
    """
    return "\n\n".join(page["text"] for page in pages)
