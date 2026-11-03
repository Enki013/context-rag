"""
src/question_bank.py
---------------------
Kitap ve Bölüm Odaklı Kalıcı Soru Bankası ve Quiz Dağıtım Motoru.
Kullanıcı verisi tutmaz; tamamen kitaplar, bölümler ve soru havuzları
üzerine odaklanmış headless mikroservis mantığıyla çalışır.

Temel Yetenekler:
  1. Kitap ve Bölüm Kaydı (Metin, sayfa aralıkları, ilerleme yüzdesi)
  2. Soru Havuzlama & Önbellek (Question Pooling & Cache Hit - 0ms)
  3. İhtiyaç Anında Üretim (On-Demand / Cache Miss)
  4. Hile Korumalı Quiz Servisi (hide_answers=True)
  5. Soru ID Bazlı Otomatik Cevap Değerlendirme
"""

import json
import logging
import os
import random
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QUIZ_ENGINE_DB_PATH
from src.chunker import chunk_text
from src.difficulty import normalize_difficulty
from src.quiz_generator import generate_quiz

logger = logging.getLogger(__name__)


def _get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    target_path = db_path or QUIZ_ENGINE_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


_initialized_dbs: set[str] = set()
_init_lock = threading.Lock()

# Aynı bölüm için eşzamanlı cache miss'lerde LLM'in iki kez çalışmasını önler (tek süreç içinde).
_pool_locks: dict[tuple, threading.Lock] = {}
_pool_locks_guard = threading.Lock()


def _pool_lock(db_path: Optional[str], book_id: str, chapter_index: int, difficulty: str) -> threading.Lock:
    key = (os.path.abspath(db_path or QUIZ_ENGINE_DB_PATH), book_id, chapter_index, normalize_difficulty(difficulty))
    with _pool_locks_guard:
        return _pool_locks.setdefault(key, threading.Lock())


def _add_missing_columns(cur: sqlite3.Cursor, table: str, columns: Dict[str, str]):
    existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, decl in columns.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_question_bank_db(db_path: Optional[str] = None):
    """Veritabanı tablolarını ve indekslerini başlatır (her veritabanı için süreç başına bir kez)."""
    target = os.path.abspath(db_path or QUIZ_ENGINE_DB_PATH)
    if target in _initialized_dbs and os.path.exists(target):
        return
    with _init_lock:
        if target in _initialized_dbs and os.path.exists(target):
            return
        _create_schema(db_path)
        _initialized_dbs.add(target)


def _create_schema(db_path: Optional[str] = None):
    with _get_connection(db_path) as conn:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS books (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                author TEXT,
                source_format TEXT DEFAULT 'epub',
                total_chapters INTEGER DEFAULT 0,
                total_pages INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                chapter_index INTEGER NOT NULL,
                title TEXT NOT NULL,
                start_page INTEGER,
                end_page INTEGER,
                progress_percent REAL DEFAULT 0.0,
                char_count INTEGER DEFAULT 0,
                raw_text TEXT,
                UNIQUE(book_id, chapter_index)
            );

            CREATE TABLE IF NOT EXISTS question_bank (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                chapter_index INTEGER NOT NULL,
                question TEXT NOT NULL,
                options TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                explanation TEXT NOT NULL,
                difficulty TEXT DEFAULT 'medium',
                source_pages TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_qb_lookup ON question_bank(book_id, chapter_index, difficulty);
            """
        )
        # Eski veritabanları için geriye uyumlu şema göçü
        _add_missing_columns(cur, "question_bank", {
            "model": "TEXT",
            "prompt_version": "TEXT",
            "source_chunk_indices": "TEXT",
            "evidence": "TEXT",
        })
        # Zorluk değerleri İngilizceye taşındı (kolay/orta/zor -> easy/medium/hard)
        cur.execute(
            """
            UPDATE question_bank SET difficulty = CASE lower(difficulty)
                WHEN 'kolay' THEN 'easy' WHEN 'orta' THEN 'medium' WHEN 'zor' THEN 'hard'
                ELSE difficulty END
            WHERE lower(difficulty) IN ('kolay', 'orta', 'zor')
            """
        )
        conn.commit()
    logger.info(f"Soru bankası veritabanı hazır: {db_path or QUIZ_ENGINE_DB_PATH}")


# ─── Kitap & Bölüm Yönetimi ─────────────────────────────────────────────────


def register_book_content(
    book_id: str,
    title: str,
    author: Optional[str],
    source_format: str,
    chapters: List[Dict[str, Any]],
    total_pages: int = 0,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kitap ve bölümlerini kalıcı veritabanına kaydeder.
    Her bölümün metni ve sayfa aralıkları saklanır.

    Sorular (book_id, chapter_index) ile bağlı olduğu için, bölümleme değiştiğinde
    (örn. bölüm regex'i güncellendi) eski sorular yanlış bölüme düşer ve spoiler olur.
    Bu yüzden metni değişen bölümlerin havuzu silinir, artık var olmayan bölümler kaldırılır.
    """
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        cur = conn.cursor()

        existing_texts = {
            row["chapter_index"]: row["raw_text"] or ""
            for row in cur.execute(
                "SELECT chapter_index, raw_text FROM chapters WHERE book_id = ?", (book_id,)
            ).fetchall()
        }
        new_indices = {ch.get("chapter_index", 1) for ch in chapters}

        stale_indices = [idx for idx in existing_texts if idx not in new_indices]
        for ch in chapters:
            idx = ch.get("chapter_index", 1)
            new_text = ch.get("text") or ch.get("raw_text") or ""
            if idx in existing_texts and existing_texts[idx] != new_text:
                stale_indices.append(idx)

        for idx in stale_indices:
            deleted = cur.execute(
                "DELETE FROM question_bank WHERE book_id = ? AND chapter_index = ?", (book_id, idx)
            ).rowcount
            if deleted:
                logger.info(f"Bölüm içeriği değişti, {deleted} eski soru silindi (Kitap: {book_id}, Bölüm: {idx})")
        for idx in existing_texts:
            if idx not in new_indices:
                cur.execute("DELETE FROM chapters WHERE book_id = ? AND chapter_index = ?", (book_id, idx))
        cur.execute(
            """
            INSERT INTO books (id, title, author, source_format, total_chapters, total_pages)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                author = excluded.author,
                source_format = excluded.source_format,
                total_chapters = excluded.total_chapters,
                total_pages = excluded.total_pages
            """,
            (book_id, title, author or "Bilinmiyor", source_format, len(chapters), total_pages),
        )

        for ch in chapters:
            ch_idx = ch.get("chapter_index", 1)
            ch_title = ch.get("title", f"Bölüm {ch_idx}")
            raw_text = ch.get("text") or ch.get("raw_text") or ""
            char_count = len(raw_text)
            start_p = ch.get("start_page") or ch.get("start_p")
            end_p = ch.get("end_page") or ch.get("end_p")
            progress_pct = ch.get("progress_percent", 0.0)

            cur.execute(
                """
                INSERT INTO chapters (book_id, chapter_index, title, start_page, end_page, progress_percent, char_count, raw_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(book_id, chapter_index) DO UPDATE SET
                    title = excluded.title,
                    start_page = excluded.start_page,
                    end_page = excluded.end_page,
                    progress_percent = excluded.progress_percent,
                    char_count = excluded.char_count,
                    raw_text = excluded.raw_text
                """,
                (book_id, ch_idx, ch_title, start_p, end_p, progress_pct, char_count, raw_text),
            )

        conn.commit()

    return {
        "book_id": book_id,
        "title": title,
        "author": author,
        "total_chapters": len(chapters),
        "total_pages": total_pages,
    }


def get_book(book_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Kitap metaverilerini döner."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        if not row:
            return None
        return dict(row)


def get_book_chapters(book_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Kitabın bölümlerini ve havuzdaki soru adetlerini döner."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.*, 
                   (SELECT COUNT(*) FROM question_bank q 
                    WHERE q.book_id = c.book_id AND q.chapter_index = c.chapter_index) as pool_count
            FROM chapters c
            WHERE c.book_id = ?
            ORDER BY c.chapter_index ASC
            """,
            (book_id,),
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d.pop("raw_text", None)  # Bölüm listesinde dev metinleri gönderme
            results.append(d)
        return results


def get_chapter(book_id: str, chapter_index: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Tek bir bölümün metnini ve bilgilerini döner."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM chapters WHERE book_id = ? AND chapter_index = ?",
            (book_id, chapter_index),
        ).fetchone()
        if not row:
            return None
        return dict(row)


# ─── Soru Havuzu (Question Pool) Yönetimi ───────────────────────────────────


def count_pool_questions(
    book_id: str,
    chapter_index: int,
    difficulty: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Belirli bir bölümün havuzundaki soru adedini döner."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        if difficulty:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM question_bank WHERE book_id = ? AND chapter_index = ? AND difficulty = ?",
                (book_id, chapter_index, normalize_difficulty(difficulty)),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM question_bank WHERE book_id = ? AND chapter_index = ?",
                (book_id, chapter_index),
            ).fetchone()
        return row["cnt"] if row else 0


def save_questions_to_pool(
    book_id: str,
    chapter_index: int,
    questions: List[Dict[str, Any]],
    difficulty: str = "medium",
    db_path: Optional[str] = None,
) -> int:
    """Üretilen soruları soru bankası tablosuna kaydeder."""
    init_question_bank_db(db_path)
    saved_count = 0
    with _get_connection(db_path) as conn:
        cur = conn.cursor()
        for q in questions:
            opts_json = json.dumps(q.get("options", []), ensure_ascii=False)
            pages_json = json.dumps(q.get("source_pages", []), ensure_ascii=False)
            diff = normalize_difficulty(q.get("difficulty") or difficulty)

            cur.execute(
                """
                INSERT INTO question_bank
                    (book_id, chapter_index, question, options, correct_answer, explanation, difficulty,
                     source_pages, model, prompt_version, source_chunk_indices, evidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book_id,
                    chapter_index,
                    q["question"],
                    opts_json,
                    q["correct_answer"],
                    q["explanation"],
                    diff,
                    pages_json,
                    q.get("model"),
                    q.get("prompt_version"),
                    json.dumps(q.get("source_chunk_indices", [])),
                    q.get("evidence"),
                ),
            )
            saved_count += 1
        conn.commit()

    logger.info(f"{saved_count} soru havuza kaydedildi (Kitap: {book_id}, Bölüm: {chapter_index})")
    return saved_count


def get_cached_questions(
    book_id: str,
    chapter_index: int,
    count: int = 5,
    difficulty: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Havuzdan rastgele soru çeker."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        query = "SELECT * FROM question_bank WHERE book_id = ? AND chapter_index = ?"
        params = [book_id, chapter_index]
        if difficulty:
            query += " AND difficulty = ?"
            params.append(normalize_difficulty(difficulty))

        rows = conn.execute(query, params).fetchall()
        if not rows:
            return []

        all_items = [dict(r) for r in rows]
        for item in all_items:
            try:
                item["options"] = json.loads(item["options"])
            except Exception:
                item["options"] = []
            try:
                item["source_pages"] = json.loads(item.get("source_pages") or "[]")
            except Exception:
                item["source_pages"] = []

        if len(all_items) <= count:
            random.shuffle(all_items)
            return all_items

        return random.sample(all_items, count)


# ─── Quiz Servis Etme & Değerlendirme (Mobile Headless Core) ────────────────


def serve_chapter_quiz(
    book_id: str,
    chapter_index: int,
    count: int = 5,
    difficulty: str = "medium",
    min_pool_target: int = 10,
    hide_answers: bool = True,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Mobil uygulama için quiz getirir.
    1. Havuzda yeterli soru varsa 0ms içinde döner (Cache Hit).
    2. Havuz boş veya yetersizse bölüm metninden LLM ile soru üretip havuza ekler (Cache Miss).
    3. Hile koruması (hide_answers=True) ile doğru şık ve açıklamayı mobil kullanıcıdan gizler.
    """
    pool_count = count_pool_questions(book_id, chapter_index, difficulty=difficulty, db_path=db_path)
    cache_hit = pool_count >= count

    if not cache_hit:
        generated = _ensure_pool(
            book_id,
            chapter_index,
            difficulty=difficulty,
            target_count=max(count, min_pool_target),
            db_path=db_path,
        )
        # Başka bir istek havuzu bu arada doldurduysa yine cache hit sayılır
        cache_hit = generated == 0

    selected_questions = get_cached_questions(
        book_id, chapter_index, count=count, difficulty=difficulty, db_path=db_path
    )

    return _format_served_quiz(book_id, chapter_index, difficulty, cache_hit, selected_questions, hide_answers, db_path)


def populate_chapter_pool(
    book_id: str,
    chapter_index: int,
    target_count: int = 10,
    difficulty: str = "medium",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Bölüm havuzunu target_count soruya kadar doldurur (havuz kısmen doluysa eksik kadar üretir)."""
    generated = _ensure_pool(book_id, chapter_index, difficulty, target_count, db_path=db_path)
    return {
        "generated": generated,
        "total_pool_size": count_pool_questions(book_id, chapter_index, difficulty=difficulty, db_path=db_path),
    }


def _ensure_pool(
    book_id: str,
    chapter_index: int,
    difficulty: str,
    target_count: int,
    db_path: Optional[str] = None,
) -> int:
    """
    Havuzda target_count kadar soru yoksa eksik kadarını LLM ile üretip kaydeder.
    Aynı bölüm için kilit tutar; kilidi bekleyen istek, önceki üretim bitince havuzu yeniden sayar.
    Üretilen soru sayısını döner.
    """
    with _pool_lock(db_path, book_id, chapter_index, difficulty):
        pool_count = count_pool_questions(book_id, chapter_index, difficulty=difficulty, db_path=db_path)
        needed = target_count - pool_count
        if needed <= 0:
            return 0

        raw_text = _load_chapter_text(book_id, chapter_index, db_path=db_path)
        chunks = chunk_text(raw_text)

        logger.info(
            f"Havuz yetersiz ({pool_count}/{target_count}). LLM ile {needed} soru üretiliyor..."
        )
        new_questions = generate_quiz(
            chunks,
            num_questions=needed,
            difficulty=difficulty,
            randomize=True,
            exclude_questions=_pool_questions(book_id, chapter_index, difficulty, db_path=db_path),
        )
        return save_questions_to_pool(
            book_id=book_id,
            chapter_index=chapter_index,
            questions=new_questions,
            difficulty=difficulty,
            db_path=db_path,
        )


def _pool_questions(book_id: str, chapter_index: int, difficulty: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Bölüm havuzundaki tüm sorular (tekrar kontrolü için: soru, şıklar, doğru cevap)."""
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT question, options, correct_answer FROM question_bank "
            "WHERE book_id = ? AND chapter_index = ? AND difficulty = ?",
            (book_id, chapter_index, normalize_difficulty(difficulty)),
        ).fetchall()
    result = []
    for r in rows:
        try:
            options = json.loads(r["options"])
        except Exception:
            options = []
        result.append({"question": r["question"], "options": options, "correct_answer": r["correct_answer"]})
    return result


def _load_chapter_text(book_id: str, chapter_index: int, db_path: Optional[str] = None) -> str:
    """Bölüm metnini veritabanından okur; yoksa yerel dosyadan veya Gutenberg'den çıkarıp kaydeder."""
    ch = get_chapter(book_id, chapter_index, db_path=db_path)
    if ch and (ch.get("raw_text") or "").strip():
        return ch["raw_text"]

    from config import BOOKS_DIR
    clean_id = str(book_id).strip("'\"")

    # 1. Öncelik: data/books/{clean_id}.txt yerel dosyasından oku
    local_path = os.path.join(BOOKS_DIR, f"{clean_id}.txt")
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 100:
        try:
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
            from src.dataset_search import extract_chapters
            extracted = extract_chapters(full_text)
            if extracted:
                register_book_content(
                    book_id=book_id,
                    title=f"Kitap #{clean_id}",
                    author="Bilinmiyor",
                    source_format="gutenberg",
                    chapters=extracted,
                    db_path=db_path,
                )
                ch = get_chapter(book_id, chapter_index, db_path=db_path)
        except Exception as e:
            logger.warning(f"Yerel metinden bölüm çıkarma hatası ({clean_id}): {e}")

    # 2. Öncelik: Gutenberg kataloğundan indir
    if not ch or not (ch.get("raw_text") or "").strip():
        from src.dataset_search import get_book_chapters as get_gutenberg_chapters
        try:
            g_info = get_gutenberg_chapters(clean_id)
            g_chapters = g_info.get("chapters", [])
            if g_chapters:
                register_book_content(
                    book_id=book_id,
                    title=g_info.get("title", f"Kitap #{clean_id}"),
                    author=g_info.get("author", ""),
                    source_format="gutenberg",
                    chapters=g_chapters,
                    db_path=db_path,
                )
                ch = get_chapter(book_id, chapter_index, db_path=db_path)
        except Exception as e:
            logger.warning(f"Gutenberg fallback hatası ({clean_id}): {e}")

    if not ch or not (ch.get("raw_text") or "").strip():
        raise ValueError(f"Bölüm metni bulunamadı: Kitap {book_id}, Bölüm {chapter_index}")
    return ch["raw_text"]


def _format_served_quiz(
    book_id: str,
    chapter_index: int,
    difficulty: str,
    cache_hit: bool,
    selected_questions: List[Dict[str, Any]],
    hide_answers: bool,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    bk = get_book(book_id, db_path=db_path) or {}
    ch = get_chapter(book_id, chapter_index, db_path=db_path) or {}

    out_questions = []
    for q in selected_questions:
        q_item = {
            "id": q["id"],
            "question": q["question"],
            "options": q["options"],
            "source_pages": q.get("source_pages", []),
            "source_chapter": ch.get("title"),
            "progress_percent": ch.get("progress_percent"),
        }
        if not hide_answers:
            q_item["correct_answer"] = q["correct_answer"]
            q_item["explanation"] = q["explanation"]
        out_questions.append(q_item)

    return {
        "book_id": book_id,
        "title": bk.get("title", f"Kitap #{book_id}"),
        "author": bk.get("author", ""),
        "chapter_index": chapter_index,
        "chapter_title": ch.get("title", f"Bölüm {chapter_index}"),
        "difficulty": difficulty,
        "cache_hit": cache_hit,
        "total_pool_size": count_pool_questions(book_id, chapter_index, difficulty=difficulty, db_path=db_path),
        "question_count": len(out_questions),
        "questions": out_questions,
    }


def evaluate_answers_against_bank(
    answers: List[Dict[str, Any]],
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Mobil kullanıcının cevaplarını soru ID'leri üzerinden veritabanındaki
    gerçek cevaplarla karşılaştırır. İstemciden doğru cevabı almaz (Hile yapılamaz).

    Args:
        answers: [{"question_id": 1, "selected_option": "A) ..."}, ...]

    Returns:
        { "total": 5, "correct_count": 4, "score_percentage": 80.0, "details": [...] }
    """
    init_question_bank_db(db_path)
    if not answers:
        return {
            "total_questions": 0,
            "correct_count": 0,
            "score_percentage": 0.0,
            "result_badge": "Soru Yok",
            "details": [],
        }

    details = []
    correct_count = 0

    with _get_connection(db_path) as conn:
        for idx, ans in enumerate(answers):
            q_id = ans.get("question_id")
            user_sel = (ans.get("selected_option") or "").strip()

            row = conn.execute("SELECT * FROM question_bank WHERE id = ?", (q_id,)).fetchone()
            if not row:
                details.append({
                    "question_id": q_id,
                    "question_index": idx,
                    "question": "Bilinmeyen Soru",
                    "user_selected": user_sel or "Cevaplanmadı",
                    "correct_answer": "?",
                    "correct_option": "?",
                    "is_correct": False,
                    "explanation": "Soru veritabanında bulunamadı.",
                })
                continue

            q_data = dict(row)
            try:
                options = json.loads(q_data["options"])
            except Exception:
                options = []

            correct_letter = q_data["correct_answer"].strip().upper()
            correct_opt = next((opt for opt in options if opt.startswith(f"{correct_letter})")), correct_letter)

            # Eşleşme kontrolü: "A" harfi veya "A) Baker Street"
            user_letter = user_sel[:1].upper() if user_sel and user_sel[1:2] in (")", ".", " ") else user_sel.upper()
            is_correct = (user_letter == correct_letter) or (user_sel == correct_opt)

            if is_correct:
                correct_count += 1

            details.append({
                "question_id": q_id,
                "question_index": idx,
                "question": q_data["question"],
                "user_selected": user_sel if user_sel else "Cevaplanmadı",
                "correct_answer": correct_letter,
                "correct_option": correct_opt,
                "is_correct": is_correct,
                "explanation": q_data["explanation"],
            })

    total = len(answers)
    pct = round((correct_count / total) * 100, 1) if total > 0 else 0.0

    if pct == 100.0:
        badge = "Kusursuz! 🏆"
    elif pct >= 80.0:
        badge = "Harika! 🌟"
    elif pct >= 60.0:
        badge = "İyi İş! 👍"
    else:
        badge = "Tekrar Dene! 📚"

    return {
        "total_questions": total,
        "correct_count": correct_count,
        "score_percentage": pct,
        "result_badge": badge,
        "details": details,
    }
