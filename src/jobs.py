"""
src/jobs.py
-----------
Asenkron üretim işleri. Uzun süren LLM üretimi HTTP isteğinin içinde değil, SQLite'ta tutulan bir
kuyruktan tek bir arka plan worker'ı tarafından sırayla işlenir (Ollama tek GPU'da zaten sırayla
çalıştığı için doğru eşzamanlılık 1'dir).

- Aynı iş için tekrar gelen istekler (aynı dedup_key) mevcut kuyruktaki/çalışan işe bağlanır.
- Süreç yeniden başlarsa yarıda kalan ("running") işler yeniden kuyruğa alınır.
- İş türleri: "populate" (bölüm havuzunu doldur), "quiz" (/quizzes için quiz üret),
  "reverify" (doğrulanmamış havuz sorularını yeniden doğrula).
"""

import json
import logging
import threading
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

from src.question_bank import _get_connection, init_question_bank_db

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "running")


# ─── Kuyruk işlemleri ───────────────────────────────────────────────────────


def _row_to_job(row) -> Dict[str, Any]:
    job = dict(row)
    for key in ("params", "result"):
        if job.get(key):
            try:
                job[key] = json.loads(job[key])
            except Exception:
                pass
    return job


def enqueue(kind: str, params: Dict[str, Any], dedup_key: Optional[str] = None,
            db_path: Optional[str] = None) -> Dict[str, Any]:
    """İşi kuyruğa ekler. Aynı dedup_key ile kuyrukta/çalışan bir iş varsa onu döner."""
    if kind not in HANDLERS:
        raise ValueError(f"Bilinmeyen iş türü: {kind}")
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if dedup_key:
            row = conn.execute(
                f"SELECT * FROM generation_jobs WHERE dedup_key = ? AND status IN {ACTIVE_STATUSES} "
                "ORDER BY created_at LIMIT 1",
                (dedup_key,),
            ).fetchone()
            if row:
                conn.commit()
                return _row_to_job(row)
        job_id = uuid.uuid4().hex[:16]
        conn.execute(
            "INSERT INTO generation_jobs (id, kind, params, dedup_key) VALUES (?, ?, ?, ?)",
            (job_id, kind, json.dumps(params, ensure_ascii=False), dedup_key),
        )
        row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
        conn.commit()
    logger.info(f"İş kuyruğa alındı: {kind} {job_id} {params}")
    _notify_worker()
    return _row_to_job(row)


def get_job(job_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM generation_jobs WHERE id = ?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def queue_position(job_id: str, db_path: Optional[str] = None) -> Optional[int]:
    """Kuyruktaki işin önünde kaç iş olduğu (çalışan dahil). Kuyrukta değilse None."""
    job = get_job(job_id, db_path)
    if not job or job["status"] != "queued":
        return None
    with _get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM generation_jobs WHERE status = 'running' "
            "OR (status = 'queued' AND created_at < ?) OR (status = 'queued' AND created_at = ? AND id < ?)",
            (job["created_at"], job["created_at"], job_id),
        ).fetchone()
    return row["n"]


def _claim_next(db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM generation_jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
        ).fetchone()
        if not row:
            conn.commit()
            return None
        conn.execute(
            "UPDATE generation_jobs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
            (row["id"],),
        )
        conn.commit()
    job = _row_to_job(row)
    job["status"] = "running"
    return job


def _finish(job_id: str, status: str, result: Any = None, error: Optional[str] = None,
            db_path: Optional[str] = None):
    with _get_connection(db_path) as conn:
        conn.execute(
            "UPDATE generation_jobs SET status = ?, result = ?, error = ?, finished_at = CURRENT_TIMESTAMP "
            "WHERE id = ?",
            (status, json.dumps(result, ensure_ascii=False) if result is not None else None, error, job_id),
        )
        conn.commit()


def requeue_interrupted(db_path: Optional[str] = None) -> int:
    """Süreç yeniden başladığında yarıda kalan işleri tekrar kuyruğa alır."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        n = conn.execute(
            "UPDATE generation_jobs SET status = 'queued', started_at = NULL WHERE status = 'running'"
        ).rowcount
        conn.commit()
    if n:
        logger.info(f"{n} yarıda kalmış iş yeniden kuyruğa alındı")
    return n


def run_job(job: Dict[str, Any], db_path: Optional[str] = None) -> Dict[str, Any]:
    """Tek bir işi çalıştırır ve sonucunu kaydeder."""
    handler = HANDLERS[job["kind"]]
    try:
        result = handler(job["params"], db_path)
        _finish(job["id"], "completed", result=result, db_path=db_path)
        logger.info(f"İş tamamlandı: {job['kind']} {job['id']}")
    except Exception as e:
        logger.error(f"İş başarısız: {job['kind']} {job['id']}: {e}\n{traceback.format_exc()}")
        _finish(job["id"], "failed", error=str(e), db_path=db_path)
    return get_job(job["id"], db_path)


def run_pending(db_path: Optional[str] = None, max_jobs: Optional[int] = None) -> int:
    """Kuyruktaki işleri bu thread'de sırayla çalıştırır (testler ve CLI için). Çalışan iş sayısını döner."""
    done = 0
    while max_jobs is None or done < max_jobs:
        job = _claim_next(db_path)
        if not job:
            break
        run_job(job, db_path)
        done += 1
    return done


# ─── İş türleri ─────────────────────────────────────────────────────────────


def _handle_populate(params: Dict[str, Any], db_path: Optional[str]) -> Dict[str, Any]:
    from src.question_bank import populate_chapter_pool

    return populate_chapter_pool(
        params["book_id"], int(params["chapter_index"]),
        target_count=int(params.get("target_count", 10)),
        difficulty=params.get("difficulty", "medium"),
        db_path=db_path,
    )


def _handle_quiz(params: Dict[str, Any], db_path: Optional[str]) -> Dict[str, Any]:
    from src.dataset_search import get_chunks_for_book
    from src.metrics import record_generation_run
    from src.quiz_generator import generate_quiz

    book_id = params["book_id"]
    chapter_index = params.get("chapter_index")
    chunks = get_chunks_for_book(book_id, params.get("title", ""), params.get("author", ""),
                                 max_chunks=20, chapter_index=chapter_index)
    stats: Dict[str, Any] = {}
    try:
        questions = generate_quiz(
            [{k: c.get(k) for k in ("index", "text", "source_pages", "source_chapter", "progress_percent")}
             for c in chunks],
            num_questions=int(params.get("question_count", 5)),
            difficulty=params.get("difficulty", "medium"),
            stats=stats,
        )
    finally:
        record_generation_run(stats, "quiz_job", book_id, chapter_index, db_path=db_path)
    keep = ("question", "options", "correct_answer", "explanation", "source_chapter", "progress_percent")
    return {"questions": [{**{k: q.get(k) for k in keep}, "source_pages": q.get("source_pages") or []}
                          for q in questions]}


def _handle_reverify(params: Dict[str, Any], db_path: Optional[str]) -> Dict[str, Any]:
    from src.question_bank import reverify_pool

    return reverify_pool(book_id=params.get("book_id"), chapter_index=params.get("chapter_index"),
                         db_path=db_path)


HANDLERS: Dict[str, Callable[[Dict[str, Any], Optional[str]], Dict[str, Any]]] = {
    "populate": _handle_populate,
    "quiz": _handle_quiz,
    "reverify": _handle_reverify,
}


# ─── Arka plan worker'ı ─────────────────────────────────────────────────────


class JobWorker(threading.Thread):
    """Kuyruğu sırayla işleyen tek arka plan thread'i."""

    def __init__(self, db_path: Optional[str] = None, idle_seconds: float = 5.0):
        super().__init__(name="generation-job-worker", daemon=True)
        self.db_path = db_path
        self.idle_seconds = idle_seconds
        self._wake = threading.Event()
        self._stop_event = threading.Event()

    def notify(self):
        self._wake.set()

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        self._wake.set()
        self.join(timeout)

    def run(self):
        logger.info("Üretim worker'ı başladı")
        while not self._stop_event.is_set():
            try:
                job = _claim_next(self.db_path)
            except Exception as e:
                logger.error(f"Kuyruk okunamadı: {e}")
                job = None
            if job:
                run_job(job, self.db_path)
                continue
            self._wake.wait(self.idle_seconds)
            self._wake.clear()
        logger.info("Üretim worker'ı durdu")


_worker: Optional[JobWorker] = None


def start_worker(db_path: Optional[str] = None) -> JobWorker:
    global _worker
    if _worker and _worker.is_alive():
        return _worker
    requeue_interrupted(db_path)
    _worker = JobWorker(db_path)
    _worker.start()
    return _worker


def stop_worker():
    global _worker
    if _worker:
        _worker.stop()
        _worker = None


def _notify_worker():
    if _worker:
        _worker.notify()
