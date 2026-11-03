"""
src/metrics.py
--------------
Üretim pipeline'ının gözlemlenebilirliği: her çalışmanın aşama istatistikleri ve elenen sorular
(sebepleri ve metin bağlamıyla) soru bankası veritabanına yazılır. Elenen sorular değerlendirme
setini büyütmek için dışa aktarılabilir.
"""

import json
import logging
from collections import Counter
from typing import Any, Dict, Optional

from src.question_bank import _get_connection, init_question_bank_db

logger = logging.getLogger(__name__)


def record_generation_run(
    stats: Dict[str, Any],
    source: str,
    book_id: Optional[str] = None,
    chapter_index: Optional[int] = None,
    db_path: Optional[str] = None,
) -> Optional[int]:
    """generate_quiz(stats=...) çıktısını kaydeder. Kayıt hatası üretimi asla bozmaz."""
    if not stats:
        return None
    try:
        init_question_bank_db(db_path)
        rejected = stats.get("rejected", [])
        summary = {k: v for k, v in stats.items() if k != "rejected"}
        with _get_connection(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO generation_runs (source, book_id, chapter_index, difficulty, model, verify_model,
                    prompt_version, requested, accepted, total_seconds, stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source, book_id, chapter_index, stats.get("difficulty"), stats.get("model"),
                    stats.get("verify_model"), stats.get("prompt_version"), stats.get("requested"),
                    stats.get("accepted"), (stats.get("seconds") or {}).get("total"),
                    json.dumps(summary, ensure_ascii=False),
                ),
            )
            run_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO rejected_questions (run_id, stage, reason, question, context) VALUES (?, ?, ?, ?, ?)",
                [
                    (run_id, r["stage"], r.get("reason"), json.dumps(r.get("question"), ensure_ascii=False, default=str),
                     r.get("context"))
                    for r in rejected
                ],
            )
            conn.commit()
        return run_id
    except Exception as e:
        logger.warning(f"Üretim istatistiği kaydedilemedi: {e}")
        return None


def summarize_runs(limit: int = 100, db_path: Optional[str] = None) -> Dict[str, Any]:
    """Son `limit` çalışmanın özeti: aşama başına eleme oranları, ortalama süreler, en sık eleme sebepleri."""
    init_question_bank_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, created_at, stats FROM generation_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        run_ids = [r["id"] for r in rows]
        reasons = Counter()
        if run_ids:
            marks = ",".join("?" * len(run_ids))
            for r in conn.execute(
                f"SELECT stage, reason FROM rejected_questions WHERE run_id IN ({marks})", run_ids
            ):
                reason = (r["reason"] or "").split(":")[0]
                reasons[f"{r['stage']}:{reason}"] += 1

    totals = Counter()
    seconds = Counter()
    for r in rows:
        st = json.loads(r["stats"] or "{}")
        for key in ("requested", "returned", "accepted", "generated", "llm_calls",
                    "structural_rejected", "verify_rejected", "dedup_rejected"):
            totals[key] += st.get(key) or 0
        for key, value in (st.get("seconds") or {}).items():
            seconds[key] += value or 0

    n = len(rows)
    generated = totals["generated"] or 0

    def rate(x):
        return round(x / generated, 3) if generated else None

    return {
        "runs": n,
        "last_run_at": rows[0]["created_at"] if rows else None,
        "totals": dict(totals),
        "rates_of_generated": {
            "structural_rejected": rate(totals["structural_rejected"]),
            "verify_rejected": rate(totals["verify_rejected"]),
            "dedup_rejected": rate(totals["dedup_rejected"]),
            "accepted": rate(totals["accepted"]),
        },
        "avg_seconds_per_run": {k: round(v / n, 1) for k, v in seconds.items()} if n else {},
        "top_reject_reasons": dict(reasons.most_common(10)),
    }


def list_rejected(stage: Optional[str] = None, limit: int = 100, db_path: Optional[str] = None) -> list:
    """Elenen soruları sebep ve bağlamıyla döner (değerlendirme setine aday)."""
    init_question_bank_db(db_path)
    query = "SELECT id, run_id, created_at, stage, reason, question, context FROM rejected_questions"
    params: list = []
    if stage:
        query += " WHERE stage = ?"
        params.append(stage)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _get_connection(db_path) as conn:
        out = []
        for r in conn.execute(query, params):
            item = dict(r)
            try:
                item["question"] = json.loads(item["question"])
            except Exception:
                pass
            out.append(item)
        return out
