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
  3. Lisans başlık/altlıkları temizlenir, metin chunk'lara bölünür ve kitabın tamamını
     kapsayan katmanlı örneklemeyle seçilen chunk'lar quiz üretimine iletilir.
"""

import logging
import random
import os
import re
import sqlite3
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BOOKS_DIR, CATALOG_DB_PATH, DATASET_CACHE_DIR, DATASET_MAX_RESULTS
from src.chunker import chunk_text

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

    candidate_limit = max(limit * 5, 80)
    seen_ids = set()

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
                cur.execute(sql, (fts_query, candidate_limit))
                rows = cur.fetchall()
                for r in rows:
                    b_id = str(r[0])
                    tit = r[1] or ""
                    auth = r[2] or "Bilinmiyor"
                    subj = r[3] or ""
                    # Müzik, ses kayıtları ve sesli kitapları filtrele
                    sub_low = subj.lower()
                    auth_low = auth.lower()
                    if (
                        "sound recording" in sub_low
                        or "popular music" in sub_low
                        or "audio book" in sub_low
                        or "songs and music" in sub_low
                        or "[performer]" in auth_low
                        or "[composer]" in auth_low
                        or "midi file" in tit.lower()
                    ):
                        continue
                    if b_id not in seen_ids:
                        seen_ids.add(b_id)
                        results.append(
                            {
                                "book_id": b_id,
                                "title": tit,
                                "author": auth,
                                "subjects": subj,
                                "bookshelves": r[4] or "",
                                "language": r[5] or "en",
                            }
                        )
                        if len(results) >= candidate_limit:
                            break

            # FTS sonuç vermediyse veya yetersizse standart LIKE aramasıyla destekle
            if len(results) < candidate_limit:
                needed = candidate_limit - len(results)
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
                    tit = r[1] or ""
                    auth = r[2] or "Bilinmiyor"
                    subj = r[3] or ""
                    sub_low = subj.lower()
                    auth_low = auth.lower()
                    if (
                        "sound recording" in sub_low
                        or "popular music" in sub_low
                        or "audio book" in sub_low
                        or "songs and music" in sub_low
                        or "[performer]" in auth_low
                        or "[composer]" in auth_low
                        or "midi file" in tit.lower()
                    ):
                        continue
                    if b_id not in seen_ids:
                        seen_ids.add(b_id)
                        results.append(
                            {
                                "book_id": b_id,
                                "title": tit,
                                "author": auth,
                                "subjects": subj,
                                "bookshelves": r[4] or "",
                                "language": r[5] or "en",
                            }
                        )
                        if len(results) >= candidate_limit:
                            break

    except Exception as e:
        logger.error(f"Arama sırasında SQLite hatası: {e}", exc_info=True)

    # Yerel diskte hazır/indirilmiş olanları ve kanonik ana edisyonları öne al
    def _rank_book(b):
        local_path = os.path.join(BOOKS_DIR, f"{b['book_id']}.txt")
        is_cached = os.path.isfile(local_path) and os.path.getsize(local_path) > 100
        numeric_id = int(b["book_id"]) if b["book_id"].isdigit() else 999999
        return (0 if is_cached else 1, numeric_id)

    results.sort(key=_rank_book)
    final_results = results[:limit]
    logger.info(f"Yerel arama: '{query}' → {len(final_results)} kitap döndürüldü (toplam {len(results)} adaydan).")
    return final_results


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
                "raw_text": chunk_text,
                "text": chunk_text,
                "is_fallback": True,
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
            "raw_text": chunk_text,
            "text": chunk_text,
        })

    return chapters


# ─── EPUB içindekiler tablosuyla bölüm tespiti ──────────────────────────────

# İçindekilerde bölüm olmayan girdiler
_NON_CHAPTER_TOC = re.compile(
    r"^(contents|table of contents|illustrations|list of illustrations|footnotes|transcriber.?s note.*|"
    r"index|colophon|.*project gutenberg.*|.*gutenberg.*license.*)$",
    re.IGNORECASE,
)
MIN_CHAPTER_CHARS = 500  # bundan kısa bölümler (örn. "BOOK ONE" kapsayıcı başlığı) bir sonrakiyle birleşir


def download_book_epub(book_id: str, allow_download: bool = True) -> Optional[str]:
    """Gutenberg EPUB'ını data/books/{id}.epub olarak önbelleğe alır ve yolunu döner; yoksa None."""
    import urllib.request

    local_path = os.path.join(BOOKS_DIR, f"{book_id}.epub")
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
    if not allow_download or not str(book_id).isdigit():
        return None
    os.makedirs(BOOKS_DIR, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (ContextRAG/1.0)"}
    for url in (
        f"https://www.gutenberg.org/ebooks/{book_id}.epub3.images",
        f"https://www.gutenberg.org/ebooks/{book_id}.epub.noimages",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}-images-3.epub",
    ):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as resp:
                data = resp.read()
            if data[:2] == b"PK" and len(data) > 1000:  # EPUB bir zip dosyasıdır
                with open(local_path, "wb") as f:
                    f.write(data)
                logger.info(f"EPUB indirildi: {book_id} ({len(data):,} bayt)")
                return local_path
        except Exception as e:
            logger.debug(f"EPUB indirilemedi ({url}): {e}")
    return None


def _toc_levels(epub_path: str, book_title: str = "") -> dict[int, list[str]]:
    """EPUB içindekilerini seviyelere göre döner; bölüm olmayan girdiler atılır."""
    import fitz

    with fitz.open(epub_path) as doc:
        toc = doc.get_toc() or []
    levels: dict[int, list[str]] = {}
    title_key = re.sub(r"\W+", " ", book_title).strip().lower()
    for level, title, _page in toc:
        title = re.sub(r"\s+", " ", title or "").strip()
        if not title or _NON_CHAPTER_TOC.match(title):
            continue
        if title_key and re.sub(r"\W+", " ", title).strip().lower() == title_key:
            continue  # kitabın kendi adı
        levels.setdefault(level, []).append(title)
    return levels


_HEADING_KEYWORD = re.compile(
    r"\b(chapter|letter|book|part|stave|act|scene|canto|chapitre|lettre|livre|cap[ií]tulo|kapitel|b[öo]l[üu]m)"
    r"\s*([ivxlcdm]+\b|\d+)",
    re.IGNORECASE,
)


def _heading_core(title: str) -> str:
    """
    İçindekiler girdisinden metinde aranacak başlığı çıkarır. Bazı EPUB'larda resim alt yazısı
    başlıkla birleşik yazılır ("I hope Mr. Bingley will like it. CHAPTER II.") veya boşluk eksiktir
    ("CHAPTERXXVII."); anahtar kelime + numara varsa arama oradan başlar.
    """
    m = _HEADING_KEYWORD.search(title)
    if m and m.start() > 0:
        title = title[m.start():]
    return _HEADING_KEYWORD.sub(lambda x: f"{x.group(1)} {x.group(2)}", title, count=1)


def _title_pattern(title: str) -> Optional[re.Pattern]:
    """Başlığı, noktalama ve satır sonu farklarına dayanıklı bir regex'e çevirir."""
    tokens = re.findall(r"\w+", _heading_core(title), re.UNICODE)
    if not tokens:
        return None
    return re.compile(r"(?<!\w)" + r"\W+".join(map(re.escape, tokens)) + r"(?!\w)", re.IGNORECASE)


def _locate_titles(text: str, titles: list[str]) -> list[tuple[int, str]]:
    """
    Başlıkları metindeki gerçek konumlarında bulur.
    - Metnin başındaki içindekiler listesindeki (başlıkların birbirine çok yakın geçtiği) eşleşmeler atlanır.
    - Her başlığın tüm geçişleri aday olur; içindekiler sırasını koruyan en uzun dizi seçilir
      (en uzun artan alt dizi). Böylece metnin çok ilerisinde yanlış eşleşen tek bir girdi
      (örn. kitap adının bir parçası) aradaki bölümleri kaybettirmez.
    """
    import bisect

    patterns = [(t, _title_pattern(t)) for t in titles]
    patterns = [(t, p) for t, p in patterns if p]
    if not patterns:
        return []

    # İçindekiler bloğu: başlıkların ilk geçişlerinin art arda çok kısa aralıklarla (<300 karakter)
    # dizildiği, metnin ilk %20'sindeki en az 3 girdilik bölge. Gerçek başlıklar arasında bölüm metni olur.
    firsts = sorted(m.start() for _, p in patterns if (m := p.search(text)))
    skip_until = 0
    i = 0
    while i < len(firsts):
        j = i
        while j + 1 < len(firsts) and firsts[j + 1] - firsts[j] < 300:
            j += 1
        if j - i + 1 >= min(3, len(patterns)) and firsts[i] < len(text) * 0.2:
            skip_until = firsts[j] + 1
            break
        i = j + 1

    # Adaylar: (başlık sırası, konum). Aynı başlığın konumları azalan sırada eklenir ki
    # bir başlıktan en fazla bir geçiş seçilsin.
    candidates = []
    for order, (title, pattern) in enumerate(patterns):
        positions = [m.start() for m in pattern.finditer(text) if m.start() >= skip_until]
        candidates.extend((order, pos, title) for pos in sorted(positions, reverse=True))

    # Konuma göre en uzun artan alt dizi (patience sorting + geri izleme)
    tails, tail_idx, parent = [], [], [None] * len(candidates)
    for i, (_order, pos, _title) in enumerate(candidates):
        k = bisect.bisect_left(tails, pos)
        if k == len(tails):
            tails.append(pos)
            tail_idx.append(i)
        else:
            tails[k] = pos
            tail_idx[k] = i
        parent[i] = tail_idx[k - 1] if k > 0 else None
    chain, i = [], (tail_idx[-1] if tail_idx else None)
    while i is not None:
        chain.append((candidates[i][1], candidates[i][2]))
        i = parent[i]
    return chain[::-1]


def chapters_from_epub_toc(raw_text: str, epub_path: str, book_title: str = "") -> list[dict]:
    """
    Bölüm başlıklarını EPUB içindekilerinden alır, sınırlarını düz metinde bulur (karakter
    konumları extract_chapters ile aynı temizlenmiş metne göredir). Bulunamazsa boş liste döner.
    """
    clean = clean_gutenberg_text(raw_text)
    total_len = len(clean)
    if not total_len:
        return []
    try:
        levels = _toc_levels(epub_path, book_title)
    except Exception as e:
        logger.warning(f"EPUB içindekileri okunamadı ({epub_path}): {e}")
        return []

    best: list[tuple[int, str]] = []
    for level in sorted(levels):  # en çok başlığı bulunan seviye; eşitlikte üst seviye
        located = _locate_titles(clean, levels[level])
        if len(located) > len(best):
            best = located
    if len(best) < 2:
        return []

    # Çok kısa bölümleri (kapsayıcı başlıklar) bir sonrakiyle birleştir
    merged: list[tuple[int, str]] = []
    for i, (start, title) in enumerate(best):
        end = best[i + 1][0] if i + 1 < len(best) else total_len
        if end - start < MIN_CHAPTER_CHARS and i + 1 < len(best):
            continue
        merged.append((start, title))

    chapters = []
    for idx, (start_c, title) in enumerate(merged):
        end_c = merged[idx + 1][0] if idx + 1 < len(merged) else total_len
        body = clean[start_c:end_c]
        chapters.append({
            "chapter_index": idx + 1,
            "title": title,
            "start_char": start_c,
            "end_char": end_c,
            "char_count": len(body),
            "progress_percent": round((start_c / total_len) * 100, 1),
            "raw_text": body,
            "text": body,
            "detection": "epub_toc",
        })
    return chapters


def detect_chapters(book_id: str, raw_text: str, book_title: str = "", allow_download: bool = True) -> list[dict]:
    """Önce Gutenberg EPUB içindekilerini dener, olmazsa metin üzerinde regex ile bölüm tespitine döner."""
    regex_chapters = extract_chapters(raw_text)
    regex_is_fallback = bool(regex_chapters) and all(ch.get("is_fallback") for ch in regex_chapters)
    for ch in regex_chapters:
        ch.setdefault("detection", "regex_fallback" if regex_is_fallback else "regex")

    epub_path = download_book_epub(book_id, allow_download=allow_download)
    epub_chapters = chapters_from_epub_toc(raw_text, epub_path, book_title) if epub_path else []

    # Daha çok gerçek bölüm bulan yöntem kazanır; regex'in "4 eşit parça" yedeği 0 sayılır
    regex_score = 0 if regex_is_fallback else len(regex_chapters)
    if epub_chapters and len(epub_chapters) >= regex_score:
        return epub_chapters
    return regex_chapters


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
    chapters = detect_chapters(book_id, raw_text, book_title=title)
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
    diske kaydeder), chunk'lara böler ve quiz için temsil edici chunk'ları döndürür.

    Args:
        book_id: Gutenberg kitap ID'si
        title: Kitap adı
        author: Yazar adı
        max_chunks: Seçilecek maksimum chunk adedi
        chapter_index: Belirli bir bölümden soru üretmek için bölüm numarası (1-based)

    Returns:
        [{"index": int, "text": str, "title": str, "author": str, "source_pages": [], "source_chapter": str, "progress_percent": float}, ...]
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

    all_chapters = detect_chapters(book_id, raw_text, book_title=title)

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

        for c in raw_chunks:
            c_title, c_pct = _match_chap(c.get("start_char") or 0)
            c["source_chapter"] = c_title
            c["progress_percent"] = c_pct

    logger.info(f"Kitap '{title}' ({len(clean_text):,} karakter) → {len(raw_chunks)} chunk.")

    # 4. Quiz için temsil edici chunk'ları seç: metnin farklı yerlerinden ardışık pencereler
    #    (tek tek dağınık chunk'lar birleştirilince LLM birbirinden kopuk sahneleri karıştırıyordu)
    from src.quiz_generator import MAX_CHUNKS_PER_CALL
    window = max(1, min(MAX_CHUNKS_PER_CALL, max_chunks))
    selected_chunks = _sample_windows(raw_chunks, n_windows=max(1, max_chunks // window), window=window)

    return [
        {
            "index": chunk["index"],
            "text": chunk["text"],
            "title": title,
            "author": author,
            "source_pages": chunk.get("source_pages", []),
            "source_chapter": chunk.get("source_chapter"),
            "progress_percent": chunk.get("progress_percent"),
        }
        for chunk in selected_chunks
    ]


def _sample_windows(items: list, n_windows: int, window: int, rng: Optional[random.Random] = None) -> list:
    """
    Listeyi n_windows eşit dilime böler ve her dilimden `window` uzunluğunda ardışık bir pencere seçer.
    Böylece metnin başı, ortası ve sonu temsil edilir, her pencere kendi içinde bütünlüklü kalır.
    Pencereler çakışmaz ve sıralama korunur.
    """
    n = len(items)
    if n <= n_windows * window:
        return list(items)
    rng = rng or random
    picks = []
    for i in range(n_windows):
        lo = (i * n) // n_windows
        hi = ((i + 1) * n) // n_windows  # dilimin sonu (hariç)
        start = rng.randint(lo, max(lo, hi - window))
        picks.extend(items[start:start + window])
    return picks
