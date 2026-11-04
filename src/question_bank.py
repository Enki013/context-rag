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
import config
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

            -- Gözlemlenebilirlik: her üretim çalışmasının istatistikleri ve elenen sorular
            CREATE TABLE IF NOT EXISTS generation_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                book_id TEXT,
                chapter_index INTEGER,
                difficulty TEXT,
                model TEXT,
                verify_model TEXT,
                prompt_version TEXT,
                requested INTEGER,
                accepted INTEGER,
                total_seconds REAL,
                stats TEXT
            );

            CREATE TABLE IF NOT EXISTS rejected_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER REFERENCES generation_runs(id) ON DELETE CASCADE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                stage TEXT NOT NULL,
                reason TEXT,
                question TEXT,
                context TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_rejected_stage ON rejected_questions(stage, reason);

            -- Asenkron üretim işleri (tek worker sırayla işler)
            CREATE TABLE IF NOT EXISTS generation_jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                params TEXT NOT NULL,
                dedup_key TEXT,
                status TEXT NOT NULL DEFAULT 'queued',
                result TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                finished_at TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_queue ON generation_jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_jobs_dedup ON generation_jobs(dedup_key, status);
            """
        )
        # Eski veritabanları için geriye uyumlu şema göçü
        _add_missing_columns(cur, "question_bank", {
            "model": "TEXT",
            "prompt_version": "TEXT",
            "source_chunk_indices": "TEXT",
            "evidence": "TEXT",
            # 1: sunulmaz (yeniden doğrulamada elendi). Silinmez; eski istemcilerin cevap değerlendirmesi bozulmasın.
            "retired": "INTEGER DEFAULT 0",
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
            f"""
            SELECT c.*,
                   (SELECT COUNT(*) FROM question_bank q
                    WHERE q.book_id = c.book_id AND q.chapter_index = c.chapter_index{_servable_filter()}) as pool_count
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


def _servable_filter() -> str:
    """Sunulabilir sorular: emekli değil; doğrulama açıksa kanıtı olan (doğrulanmış)."""
    clause = " AND COALESCE(retired, 0) = 0"
    if config.VERIFY_ANSWERS:
        clause += " AND evidence IS NOT NULL"
    return clause


def count_pool_questions(
    book_id: str,
    chapter_index: int,
    difficulty: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """Belirli bir bölümün havuzundaki sunulabilir soru adedini döner."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        query = "SELECT COUNT(*) as cnt FROM question_bank WHERE book_id = ? AND chapter_index = ?" + _servable_filter()
        params: list = [book_id, chapter_index]
        if difficulty:
            query += " AND difficulty = ?"
            params.append(normalize_difficulty(difficulty))
        row = conn.execute(query, params).fetchone()
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
    """Havuzdan rastgele (sunulabilir) soru çeker."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        query = "SELECT * FROM question_bank WHERE book_id = ? AND chapter_index = ?" + _servable_filter()
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


def get_pool_quiz(
    book_id: str,
    chapter_index: int,
    count: int = 5,
    difficulty: str = "medium",
    hide_answers: bool = True,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Havuzda yeterli soru varsa quiz'i döner (LLM çağrısı yapmaz), yoksa None.
    API bu fonksiyonu kullanır; eksik havuz arka planda bir işle doldurulur.
    """
    difficulty = normalize_difficulty(difficulty)
    if count_pool_questions(book_id, chapter_index, difficulty=difficulty, db_path=db_path) < count:
        return None
    selected = get_cached_questions(book_id, chapter_index, count=count, difficulty=difficulty, db_path=db_path)
    return _format_served_quiz(book_id, chapter_index, difficulty, True, selected, hide_answers, db_path)


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
        from src.metrics import record_generation_run

        stats: Dict[str, Any] = {}
        try:
            new_questions = generate_quiz(
                chunks,
                num_questions=needed,
                difficulty=difficulty,
                randomize=True,
                exclude_questions=_pool_questions(book_id, chapter_index, difficulty, db_path=db_path),
                stats=stats,
            )
        finally:
            record_generation_run(stats, "pool", book_id, chapter_index, db_path=db_path)
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
            "WHERE book_id = ? AND chapter_index = ? AND difficulty = ? AND COALESCE(retired, 0) = 0",
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


REVERIFY_TOP_K = 4  # yeniden doğrulamada soru başına bulunan en ilgili chunk sayısı


def reverify_pool(
    book_id: Optional[str] = None,
    chapter_index: Optional[int] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kanıtı olmayan (doğrulama eklenmeden önce üretilmiş) havuz sorularını doğrular.
    Bu sorularda hangi metin parçasından üretildikleri kayıtlı olmadığı için, her soru için bölüm
    metninin en ilgili REVERIFY_TOP_K chunk'ı bge-m3 benzerliğiyle bulunur (retrieval) ve
    doğrulayıcıya bağlam olarak verilir. Geçenlere kanıt yazılır; geçemeyenler emekliye ayrılır.
    """
    import time

    import numpy as np

    from src.embedder import _get_embeddings_batch
    from src.metrics import record_generation_run
    from src.quiz_generator import LLM_MODEL, VERIFY_MODEL, _build_context, _dedup_text, _verify_questions

    init_question_bank_db(db_path)
    query = ("SELECT id, book_id, chapter_index, question, options, correct_answer, explanation FROM question_bank "
             "WHERE evidence IS NULL AND COALESCE(retired, 0) = 0")
    params: list = []
    if book_id:
        query += " AND book_id = ?"
        params.append(book_id)
    if chapter_index is not None:
        query += " AND chapter_index = ?"
        params.append(chapter_index)
    with _get_connection(db_path) as conn:
        rows = [dict(r) for r in conn.execute(query + " ORDER BY book_id, chapter_index, id", params)]

    by_chapter: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        r["options"] = json.loads(r["options"] or "[]")
        by_chapter.setdefault((r["book_id"], r["chapter_index"]), []).append(r)

    summary = {"checked": 0, "verified": 0, "retired": 0, "skipped_chapters": []}
    t_start = time.perf_counter()
    rejects: list = []
    for (b_id, ch_idx), questions in by_chapter.items():
        try:
            chunks = chunk_text(_load_chapter_text(b_id, ch_idx, db_path=db_path))
            chunk_vecs = np.asarray(_get_embeddings_batch([c["text"] for c in chunks]), dtype=float)
            q_vecs = np.asarray(_get_embeddings_batch([_dedup_text(q) for q in questions]), dtype=float)
        except Exception as e:
            logger.warning(f"Yeniden doğrulama atlandı ({b_id}, bölüm {ch_idx}): {e}")
            summary["skipped_chapters"].append({"book_id": b_id, "chapter_index": ch_idx, "error": str(e)})
            continue
        chunk_vecs /= np.clip(np.linalg.norm(chunk_vecs, axis=1, keepdims=True), 1e-12, None)
        q_vecs /= np.clip(np.linalg.norm(q_vecs, axis=1, keepdims=True), 1e-12, None)

        for q, q_vec in zip(questions, q_vecs):
            summary["checked"] += 1
            top = sorted(np.argsort(chunk_vecs @ q_vec)[::-1][:REVERIFY_TOP_K])
            context, _ = _build_context([chunks[i] for i in top])
            candidate = {k: q[k] for k in ("question", "options", "correct_answer", "explanation")}
            verified = _verify_questions([candidate], context, rejects=rejects)
            with _get_connection(db_path) as conn:
                if verified:
                    conn.execute("UPDATE question_bank SET evidence = ? WHERE id = ?",
                                 (verified[0]["evidence"], q["id"]))
                    summary["verified"] += 1
                else:
                    conn.execute("UPDATE question_bank SET retired = 1 WHERE id = ?", (q["id"],))
                    summary["retired"] += 1
                conn.commit()

    record_generation_run({
        "requested": summary["checked"], "generated": summary["checked"], "accepted": summary["verified"],
        "returned": summary["verified"], "verify_rejected": summary["retired"], "llm_calls": summary["checked"],
        "seconds": {"verify": round(time.perf_counter() - t_start, 2), "total": round(time.perf_counter() - t_start, 2)},
        "model": LLM_MODEL, "verify_model": VERIFY_MODEL, "rejected": rejects,
    }, "reverify", book_id, chapter_index, db_path=db_path)
    logger.info(f"Yeniden doğrulama: {summary}")
    return summary


def _register_gutenberg_book(book_id: str, db_path: Optional[str] = None) -> bool:
    """Gutenberg kitabının bölümlerini (EPUB içindekiler / regex) ve katalog başlığını kaydeder."""
    from src.dataset_search import get_book_chapters as get_gutenberg_chapters

    info = get_gutenberg_chapters(str(book_id).strip("'\""))
    chapters = info.get("chapters", [])
    if not chapters:
        return False
    register_book_content(
        book_id=book_id,
        title=info.get("title") or f"Kitap #{book_id}",
        author=info.get("author") or "",
        source_format="gutenberg",
        chapters=chapters,
        db_path=db_path,
    )
    return True


_refreshed_books: set = set()


def refresh_gutenberg_book(book_id: str, db_path: Optional[str] = None) -> bool:
    """
    Daha önce kaydedilmiş bir Gutenberg kitabının bölümlerini güncel tespitle karşılaştırır
    (süreç başına kitap başına bir kez). Bölüm başlıkları farklıysa veya kitap adı geçiciyse
    ("Kitap #36") yeniden kaydeder; register_book_content metni değişen bölümlerin eski sorularını
    siler (artık yanlış bölüme ait olurlardı). Yenilendiyse True döner.
    """
    key = (os.path.abspath(db_path or QUIZ_ENGINE_DB_PATH), book_id)
    if key in _refreshed_books:
        return False
    _refreshed_books.add(key)

    book = get_book(book_id, db_path=db_path)
    if not book or book.get("source_format") != "gutenberg":
        return False
    from src.dataset_search import get_book_chapters as get_gutenberg_chapters

    try:
        info = get_gutenberg_chapters(str(book_id))
    except Exception as e:
        logger.warning(f"Gutenberg bölümleri yenilenemedi ({book_id}): {e}")
        return False
    new_titles = [c["title"] for c in info.get("chapters", [])]
    old_titles = [c["title"] for c in get_book_chapters(book_id, db_path=db_path)]
    placeholder_title = str(book.get("title", "")).startswith("Kitap #")
    if not new_titles or (new_titles == old_titles and not placeholder_title):
        return False

    logger.info(f"Gutenberg bölümleri güncelleniyor ({book_id}): {len(old_titles)} -> {len(new_titles)} bölüm")
    register_book_content(
        book_id=book_id,
        title=info.get("title") or book.get("title"),
        author=info.get("author") or book.get("author"),
        source_format="gutenberg",
        chapters=info["chapters"],
        db_path=db_path,
    )
    return True


def _load_chapter_text(book_id: str, chapter_index: int, db_path: Optional[str] = None) -> str:
    """Bölüm metnini veritabanından okur; yoksa Gutenberg metninden (önce yerel önbellek) çıkarıp kaydeder."""
    ch = get_chapter(book_id, chapter_index, db_path=db_path)
    if ch and (ch.get("raw_text") or "").strip():
        return ch["raw_text"]

    try:
        if _register_gutenberg_book(book_id, db_path=db_path):
            ch = get_chapter(book_id, chapter_index, db_path=db_path)
    except Exception as e:
        logger.warning(f"Gutenberg bölümleri yüklenemedi ({book_id}): {e}")

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
        badge = "Kusursuz!"
    elif pct >= 80.0:
        badge = "Harika!"
    elif pct >= 60.0:
        badge = "İyi İş!"
    else:
        badge = "Tekrar Dene!"

    return {
        "total_questions": total,
        "correct_count": correct_count,
        "score_percentage": pct,
        "result_badge": badge,
        "details": details,
    }
