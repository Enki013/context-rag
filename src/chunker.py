import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS

logger = logging.getLogger(__name__)


def _build_page_index(pages: list[dict]) -> list[tuple[int, int, int]]:
    """
    Her sayfanın birleşik metindeki başlangıç ve bitiş offset'ini hesaplar.

    Returns:
        [(page_num, start_offset, end_offset), ...]
    """
    index = []
    offset = 0
    for page in pages:
        length = len(page["text"])
        index.append((page["page"], offset, offset + length))
        offset += length + 2  # +2 for "\n\n" separator in get_full_text
    return index


def _find_source_pages(chunk_start: int, chunk_end: int, page_index: list[tuple[int, int, int]]) -> list[int]:
    """Bir chunk'ın hangi sayfalara denk geldiğini bulur."""
    pages = []
    for page_num, pg_start, pg_end in page_index:
        if chunk_start < pg_end and chunk_end > pg_start:
            pages.append(page_num)
    return pages if pages else [1]


def chunk_text(
    text: str,
    pages: list[dict] = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[dict]:
    """
    Metni overlap ile chunk'lara ayırır.
    pages verilirse her chunk'a kaynak sayfa numaraları eklenir.

    Args:
        text: Chunking yapılacak metin
        pages: PDF sayfa listesi [{"page": 1, "text": "..."}, ...]. Opsiyonel.
        chunk_size: Chunk boyutu (karakter). None ise config'den alınır.
        chunk_overlap: Overlap boyutu (karakter). None ise config'den alınır.

    Returns:
        [{"index": 0, "text": "...", "length": 950, "source_pages": [3, 4]}, ...]
    """
    size = chunk_size or CHUNK_SIZE
    overlap = chunk_overlap or CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    raw_chunks = splitter.split_text(text)

    page_index = _build_page_index(pages) if pages else None

    chunks = []
    search_start = 0
    for i, chunk in enumerate(raw_chunks):
        chunk_data = {
            "index": i,
            "text": chunk,
            "length": len(chunk),
        }

        if page_index:
            pos = text.find(chunk, search_start)
            if pos == -1:
                pos = text.find(chunk)
            if pos >= 0:
                source_pages = _find_source_pages(pos, pos + len(chunk), page_index)
                search_start = pos + 1
            else:
                source_pages = []
            chunk_data["source_pages"] = source_pages

        chunks.append(chunk_data)

    logger.info(
        f"Metin {len(chunks)} chunk'a ayrıldı "
        f"(chunk_size={size}, overlap={overlap})"
    )

    return chunks
