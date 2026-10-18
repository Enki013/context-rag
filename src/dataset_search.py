"""
src/dataset_search.py
---------------------
Project Gutenberg kitap kataloğu üzerinde yerel, anlık ve çevrimdışı arama
yapan ve seçilen kitabın chunk'larını quiz için hazırlayan modül.

Mimari:
  1. Arama tamamen yerel SQLite FTS5 indeksi (`data/dataset_cache/catalog.db`)
     üzerinde çalışır (~0.2 milisaniye). Aramada ASLA internet indirmesi başlatılmaz.
  2. Kitap seçildiğinde metin diskte (`data/books/{book_id}.txt`) varsa anında okunur,
     yoksa bir kereye mahsus indirilip diske kalıcı olarak kaydedilir.
  3. Lisans başlık/altlıkları temizlenir, metin chunk'lara bölünür ve Ollama ile
     yalnızca gerekli chunk'lar embed edilip quiz üretimine iletilir.
"""

import logging
import os
import re
import sqlite3
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOKS_DIR, CATALOG_DB_PATH, DATASET_CACHE_DIR, DATASET_MAX_RESULTS
from src.chunker import chunk_text
from src.embedder import _get_embeddings_batch

logger = logging.getLogger(__name__)


# ─── Metin Temizleme ───────────────────────────────────────────────────────


def clean_gutenberg_text(text: str) -> str:
    """Project Gutenberg lisans başlık ve sonluklarını temizler."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT!",
    ]
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            eol = text.find("\n", pos)
            if eol != -1:
                text = text[eol + 1 :]
            break

    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]
            break

    return text.strip()


# ─── Katalog Durumu ve Oluşturma ───────────────────────────────────────────


def get_cache_status() -> dict:
    """Yerel katalog veritabanının durumunu döndürür."""
    if not os.path.isfile(CATALOG_DB_PATH):
        return {
            "exists": False,
            "num_books": 0,
            "dataset": "Project Gutenberg (Katalog indirilmedi)",
            "path": CATALOG_DB_PATH,
        }

    try:
        with sqlite3.connect(CATALOG_DB_PATH) as conn:
            cnt = conn.cursor().execute("SELECT count(*) FROM books").fetchone()[0]
        return {
            "exists": True,
            "num_books": cnt,
            "dataset": "Project Gutenberg (Yerel FTS5 İndeksi)",
            "path": CATALOG_DB_PATH,
        }
    except Exception as e:
        logger.warning(f"Katalog durumu okunamadı: {e}")
        return {
            "exists": False,
            "num_books": 0,
            "dataset": "Project Gutenberg (Hata)",
            "path": CATALOG_DB_PATH,
        }


def build_index(progress_callback=None, force: bool = False) -> int:
    """
    Katalog veritabanını indirme scriptini çağırarak oluşturur.
    Zaten varsa ve force=False ise doğrudan mevcut kitap sayısını döner.
    """
    from scripts.download_dataset import build_catalog_database, download_catalog_content

    if not force and os.path.isfile(CATALOG_DB_PATH):
        try:
            with sqlite3.connect(CATALOG_DB_PATH) as conn:
                cnt = conn.cursor().execute("SELECT count(*) FROM books").fetchone()[0]
                if cnt > 1000:
                    logger.info(f"Katalog zaten mevcut ({cnt:,} kitap). Yeniden indirilmeyecek.")
                    return cnt
        except Exception:
            pass

    logger.info("Katalog oluşturuluyor...")
    csv_bytes = download_catalog_content(progress_callback=progress_callback)
    return build_catalog_database(csv_bytes, CATALOG_DB_PATH)


# ─── Arama (Yerel SQLite FTS5) ──────────────────────────────────────────────


def _clean_fts_query(query: str) -> str:
    """FTS5 sözdizimi hatalarını önlemek için sorguyu temizler."""
    tokens = re.findall(r"\w+", query, re.UNICODE)
    if not tokens:
        return ""
    # Her token sonuna prefix wildcard ekle: "sher*" AND "holm*"
    return " ".join(f'"{t}"*' for t in tokens)


def search_books(query: str, limit: int = 20) -> list[dict]:
    """
    Kullanıcının sorgusunu yerel SQLite indeksinde arar.
    Tamamen çevrimdışıdır, ağ isteği yapmaz (<1ms sürer).

    Returns:
        [{"book_id": str, "title": str, "author": str, "subjects": str, "bookshelves": str}, ...]
    """
    if not os.path.isfile(CATALOG_DB_PATH):
        logger.warning(f"Katalog veritabanı bulunamadı: {CATALOG_DB_PATH}")
        return []

    query = query.strip()
    if not query:
        return []

    fts_query = _clean_fts_query(query)
    results = []

    try:
        with sqlite3.connect(CATALOG_DB_PATH) as conn:
            cur = conn.cursor()

            if fts_query:
                sql = """
                    SELECT b.book_id, b.title, b.author, b.subjects, b.bookshelves, b.language, rank
                    FROM books_fts f
                    JOIN books b ON f.rowid = b.id
                    WHERE books_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """
                cur.execute(sql, (fts_query, limit))
                rows = cur.fetchall()
                for r in rows:
                    results.append(
                        {
                            "book_id": str(r[0]),
                            "title": r[1],
                            "author": r[2] or "Bilinmiyor",
                            "subjects": r[3] or "",
                            "bookshelves": r[4] or "",
                            "language": r[5] or "en",
                        }
                    )

            # FTS sonuç vermediyse veya yetersizse standart LIKE aramasıyla destekle
            if len(results) < limit:
                needed = limit - len(results)
                seen_ids = {r["book_id"] for r in results}
                like_pattern = f"%{query}%"

                like_sql = """
                    SELECT book_id, title, author, subjects, bookshelves, language
                    FROM books
                    WHERE title LIKE ? OR author LIKE ?
                    LIMIT ?
                """
                cur.execute(like_sql, (like_pattern, like_pattern, needed * 2))
                for r in cur.fetchall():
                    b_id = str(r[0])
                    if b_id not in seen_ids:
                        seen_ids.add(b_id)
                        results.append(
                            {
                                "book_id": b_id,
                                "title": r[1],
                                "author": r[2] or "Bilinmiyor",
                                "subjects": r[3] or "",
                                "bookshelves": r[4] or "",
                                "language": r[5] or "en",
                            }
                        )
                        if len(results) >= limit:
                            break

    except Exception as e:
        logger.error(f"Arama sırasında SQLite hatası: {e}", exc_info=True)

    logger.info(f"Yerel arama: '{query}' → {len(results)} kitap bulundu.")
    return results


# ─── Bölüm (Chapter) ve İlerleme Tespiti ────────────────────────────────────

# 1-89 arası geçerli Roma rakamları (M., C., D. gibi isim kısaltmalarını kesin olarak dışlar)
_ROMAN_1_TO_89 = r"(?:L?X{0,3}(?:IX|IV|V?I{1,3}|V)|L?X{1,3}|XL|L)"

# Yazıyla yazılan bölüm numaraları (İngilizce, Fransızca, İspanyolca, Almanca, Türkçe)
_WORD_NUMBERS = (
    r"(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|"
    r"Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|"
    r"Twenty(?:[- ](?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine))?|Thirty|Forty|Fifty|"
    r"First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth|"
    r"Un|Une|Deux|Trois|Quatre|Cinq|Six|Sept|Huit|Neuf|Dix|Premier|Première|Second|Seconde|"
    r"Uno|Dos|Tres|Cuatro|Cinco|Seis|Siete|Ocho|Nueve|Diez|Primero|Segundo|Tercero|"
    r"Eins|Zwei|Drei|Vier|Fünf|Sechs|Sieben|Acht|Neun|Zehn|Erstes|Zweites|Drittes|"
    r"Bir|İki|Üç|Dört|Beş|Altı|Yedi|Sekiz|Dokuz|On|Birinci|İkinci|Üçüncü)"
)

# Çok dilli bölüm anahtar kelimeleri
_CHAPTER_KEYWORDS = (
    r"(?:CHAPTER|Chapter|CHAPITRE|Chapitre|CAP[ÍI]TULO|Cap[íi]tulo|"
    r"KAPITEL|Kapitel|CAPITOLO|Capitolo|BÖLÜM|Bölüm|"
    r"LETTER|Letter|LETTRE|Lettre|BRIEF|Brief|"
    r"ACT|Act|ACTE|Acte|ACTO|Acto|AUFZUG|Aufzug|"
    r"SCENE|Scene|SCÈNE|Scène|SZENE|Szene|"
    r"BOOK|Book|LIVRE|Livre|BUCH|Buch|"
    r"PART|Part|PARTIE|Partie|TEIL|Teil|"
    r"STORY|Story|CANTO|Canto|STAVE|Stave)"
)

# Ana Bölüm Yakalama Regex'i
CHAPTER_PATTERN = re.compile(
    rf"(?:^|\n\s*\n)[ \t]*(?P<heading>"
    rf"(?:\[\s*)?{_CHAPTER_KEYWORDS}\s+(?:[0-9]+|[IVXLCDM]+|{_WORD_NUMBERS})\b[^\r\n]{{0,70}}|"
    rf"{_ROMAN_1_TO_89}\.[ \t]+[A-ZÇĞİÖŞÜÂÊÎÔÛÉÈÀÙÄÖÜ][A-Za-z0-9ÇĞİÖŞÜçğıöşüÂÊÎÔÛâêîôûÉÈÀÙéèàùÄÖÜäöüß \t,\'’\-]{{2,65}}|"
    rf"(?:[0-9]{{1,3}}|{_ROMAN_1_TO_89})\.?(?=[ \t]*\r?\n[ \t]*\r?\n)"
    rf")",
    0,
)


def _normalize_heading_key(title: str) -> str:
    """TOC tekrarlarını yakalamak için başlığın çekirdek anahtarını çıkarır (örn. 'ACT I. Scene 1' -> 'act i')."""
    t = re.sub(r"[\[\]\.\:\-\—\–]+", " ", title).strip().lower()
    m_roman = re.match(rf"^({_ROMAN_1_TO_89.lower()})\b", t)
    if m_roman:
        return m_roman.group(1)
    m = re.match(
        rf"^({_CHAPTER_KEYWORDS.lower()})\s+([0-9]+|[ivxlcdm]+|{_WORD_NUMBERS.lower()})\b",
        t,
    )
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return re.sub(r"\s+", " ", t)[:25]


def extract_chapters(raw_text: str) -> list[dict]:
    """
    Gutenberg metnindeki bölümleri ve her bölümün kitap içindeki ilerleme yüzdesini (%0 - %100) çıkarır.
    TOC (İçindekiler) bloklarını ve paragraf içi sahte Roma rakamlarını (örn. 'M. Waldman') akıllı filtreler.
    Eğer metinde belirgin bölüm başlığı yoksa, metni orantılı 4 eşit parçaya böler.
    """
    clean = clean_gutenberg_text(raw_text)
    total_len = len(clean)
    if total_len == 0:
        return []

    # 1. Ham adayları topla
    raw_matches = []
    for m in CHAPTER_PATTERN.finditer(clean):
        pos = m.start("heading")
        line = m.group("heading").strip().strip("[]").strip()
        if not (1 <= len(line) <= 85):
            continue
        raw_matches.append((pos, line))

    # Eğer kitapta belirgin başlıklar (örn. 'CHAPTER I' veya 'I. A SCANDAL IN BOHEMIA') varsa,
    # boş satırdaki tekil numaralar ('II.', '3.') alt bölümdür, ana bölüm değildir.
    has_explicit_chapter_titles = sum(
        1 for _, l in raw_matches
        if any(l.upper().startswith(kw) for kw in ["CHAPTER", "CHAPITRE", "CAPÍTULO", "KAPITEL", "ACT", "BOOK", "LETTER"])
        or (re.match(rf"^{_ROMAN_1_TO_89}\.[ \t]+[A-Z]", l) and len(l) > 6)
    ) >= 2

    filtered_raw = []
    for pos, line in raw_matches:
        is_bare = bool(re.match(rf"^(?:[0-9]{{1,3}}|{_ROMAN_1_TO_89})\.?$", line))
        if has_explicit_chapter_titles and is_bare:
            continue
        filtered_raw.append((pos, line))

    # 2. Akıllı TOC (İçindekiler) ve Sahte Başlık Filtrelemesi
    keys = [_normalize_heading_key(line) for _, line in filtered_raw]
    filtered = []

    for i, (pos, title) in enumerate(filtered_raw):
        prev_d = pos - filtered_raw[i - 1][0] if i > 0 else 999999
        next_d = filtered_raw[i + 1][0] - pos if i < len(filtered_raw) - 1 else total_len - pos

        # TOC başlıkları kitabın ilerisinde (metin gövdesinde) tekrar geçer
        repeats_later = any(
            keys[j] == keys[i] and (100 < (filtered_raw[j][0] - pos) < 60000)
            for j in range(i + 1, len(filtered_raw))
        )
        if repeats_later:
            continue

        # İçeriği 400 karakterden az olan satırlar (üst-alt kapsayıcı konteyner hariç)
        if next_d < 400:
            if next_d < 120 and prev_d < 120:
                continue
            next_title = filtered_raw[i + 1][1].upper() if i + 1 < len(filtered_raw) else ""
            is_parent_container = title.upper().startswith(("ACT", "BOOK", "PART", "PARTE", "LIVRE")) and (
                next_title.startswith(("SCENE", "SCÈNE", "CHAPTER", "CHAPITRE"))
            )
            if not is_parent_container:
                continue

        filtered.append((pos, title))

    # 3. Yeterli bölüm bulunamadıysa (<= 1) %25'lik 4 parçalı Fallback uygula
    if len(filtered) <= 1:
        chapters = []
        chunk_size = max(1, total_len // 4)
        for idx in range(4):
            start_c = idx * chunk_size
            end_c = total_len if idx == 3 else (idx + 1) * chunk_size
            chunk_text = clean[start_c:end_c]
            chapters.append({
                "chapter_index": idx + 1,
                "title": f"Bölüm {idx + 1} (%{idx * 25})",
                "start_char": start_c,
                "end_char": end_c,
                "char_count": len(chunk_text),
                "progress_percent": round((start_c / total_len) * 100, 1),
            })
        return chapters

    # 4. Bölümleri oluştur
    chapters = []
    for idx, (start_c, title) in enumerate(filtered):
        end_c = filtered[idx + 1][0] if idx + 1 < len(filtered) else total_len
        chunk_text = clean[start_c:end_c]
        chapters.append({
            "chapter_index": idx + 1,
            "title": title,
            "start_char": start_c,
            "end_char": end_c,
            "char_count": len(chunk_text),
            "progress_percent": round((start_c / total_len) * 100, 1),
        })

    return chapters


def get_book_chapters(book_id: str) -> dict:
    """
    Belirli bir Gutenberg kitabının bölümlerini ve ilerleme yüzdelerini döndürür.
    Kitap metni yerel diskte yoksa otomatik olarak indirilir.
    """
    from scripts.download_dataset import download_book_text

    title = f"Kitap #{book_id}"
    author = "Bilinmiyor"
    if os.path.isfile(CATALOG_DB_PATH):
        try:
            with sqlite3.connect(CATALOG_DB_PATH) as conn:
                row = conn.cursor().execute(
                    "SELECT title, author FROM books WHERE book_id = ?", (book_id,)
                ).fetchone()
                if row:
                    title = row[0]
                    author = row[1] or "Bilinmiyor"
        except Exception as e:
            logger.warning(f"Katalogdan kitap metadata okunamadı: {e}")

    raw_text = download_book_text(book_id, title, author)
    chapters = extract_chapters(raw_text)
    return {
        "book_id": str(book_id),
        "title": title,
        "author": author,
        "total_chapters": len(chapters),
        "chapters": chapters,
    }


# ─── Kitap Metni Getirme ve Chunking ────────────────────────────────────────


def get_chunks_for_book(
    book_id: str,
    title: str,
    author: str,
    max_chunks: int = DATASET_MAX_RESULTS,
    chapter_index: Optional[int] = None,
) -> list[dict]:
    """
    Belirtilen kitabın metnini yerel diskten okur (yoksa bir defaya mahsus indirip
    diske kaydeder), chunk'lar ve Ollama ile embed edip quiz için döndürür.

    Args:
        book_id: Gutenberg kitap ID'si
        title: Kitap adı
        author: Yazar adı
        max_chunks: Seçilecek maksimum chunk adedi
        chapter_index: Belirli bir bölümden soru üretmek için bölüm numarası (1-based)

    Returns:
        [{"text": str, "title": str, "author": str, "source_pages": [], "source_chapter": str, "progress_percent": float, "embedding": [...]}, ...]
    """
    from scripts.download_dataset import download_book_text

    # 1. Metni yerel diskten al veya indirip diske yaz
    raw_text = download_book_text(book_id, title, author)
    if not raw_text or not raw_text.strip():
        raise ValueError(f"Kitap metni boş: {title} (ID: {book_id})")

    # 2. Lisans kısımlarını temizle
    clean_text = clean_gutenberg_text(raw_text)
    if len(clean_text) < 100:
        clean_text = raw_text

    all_chapters = extract_chapters(raw_text)

    # 3. Bölüm seçimi veya tüm kitap
    if chapter_index is not None:
        if chapter_index < 1 or chapter_index > len(all_chapters):
            raise ValueError(
                f"Geçersiz bölüm numarası: {chapter_index}. Bu kitapta toplam {len(all_chapters)} bölüm bulunuyor."
            )
        target_chap = all_chapters[chapter_index - 1]
        target_text = clean_text[target_chap["start_char"] : target_chap["end_char"]]
        if not target_text.strip():
            target_text = clean_text

        raw_chunks = chunk_text(target_text)
        chap_title = target_chap["title"]
        chap_progress = target_chap["progress_percent"]
        for c in raw_chunks:
            c["source_chapter"] = chap_title
            c["progress_percent"] = chap_progress
    else:
        raw_chunks = chunk_text(clean_text)

        def _match_chap(char_pos: int):
            for ch in reversed(all_chapters):
                if char_pos >= ch["start_char"]:
                    return ch["title"], ch["progress_percent"]
            return all_chapters[0]["title"], all_chapters[0]["progress_percent"]

        curr_offset = 0
        for c in raw_chunks:
            t_len = len(c["text"])
            c_title, c_pct = _match_chap(curr_offset)
            c["source_chapter"] = c_title
            c["progress_percent"] = c_pct
            curr_offset += t_len

    logger.info(f"Kitap '{title}' ({len(clean_text):,} karakter) → {len(raw_chunks)} chunk.")

    # 4. Quiz için temsil edici chunk'ları seç
    if len(raw_chunks) <= max_chunks:
        selected_chunks = raw_chunks
    else:
        step = max(1, len(raw_chunks) // max_chunks)
        selected_chunks = raw_chunks[::step][:max_chunks]

    # 5. Embedding'leri üret
    texts = [c["text"] for c in selected_chunks]
    logger.info(f"Quiz için {len(texts)} chunk embed ediliyor...")
    embeddings = _get_embeddings_batch(texts)

    result = []
    for chunk, emb in zip(selected_chunks, embeddings):
        result.append(
            {
                "text": chunk["text"],
                "title": title,
                "author": author,
                "source_pages": chunk.get("pages", []),
                "source_chapter": chunk.get("source_chapter"),
                "progress_percent": chunk.get("progress_percent"),
                "embedding": emb,
            }
        )

    return result
