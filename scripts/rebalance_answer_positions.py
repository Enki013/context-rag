"""
scripts/rebalance_answer_positions.py
-------------------------------------
Soru bankasındaki mevcut soruların şıklarını karıştırır ve doğru cevap harfini yeniden yazar.

Neden: Şık karıştırma eklenmeden önce üretilen sorularda LLM doğru cevabı çoğunlukla
A/B'ye koyuyordu (ör. 146 soruda A=46, B=69, C=23, D=8). Yeni sorular üretimde karıştırılıyor;
bu script eski havuzu dengeler.

Not: Bir istemci soruları almış ama cevabı henüz göndermemişse, karıştırma sonrası
gönderdiği "B) ..." gibi eski bir şık yanlış değerlendirilebilir. Trafik yokken çalıştırın.

Kullanım:
    python scripts/rebalance_answer_positions.py              # data/quiz_engine.db (önce yedek alır)
    python scripts/rebalance_answer_positions.py --db yol.db --dry-run
"""

import argparse
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import QUIZ_ENGINE_DB_PATH
from src.quiz_generator import _LETTERS, _has_duplicate_options, _shuffle_options


def rebalance(db_path: str, dry_run: bool = False, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, options, correct_answer FROM question_bank").fetchall()

    before, after = Counter(), Counter()
    skipped = []
    updates = []

    for q_id, options_json, correct in rows:
        correct = (correct or "").strip().upper()
        try:
            options = json.loads(options_json)
        except (TypeError, json.JSONDecodeError):
            skipped.append((q_id, "options JSON okunamadı"))
            continue
        if not isinstance(options, list) or len(options) != 4 or correct not in _LETTERS:
            skipped.append((q_id, "4 şık / geçerli harf değil"))
            continue
        if _has_duplicate_options(options):
            skipped.append((q_id, "tekrarlanan şık"))
            continue

        before[correct] += 1
        q = _shuffle_options({"options": options, "correct_answer": correct}, rng)
        after[q["correct_answer"]] += 1
        updates.append((json.dumps(q["options"], ensure_ascii=False), q["correct_answer"], q_id))

    if not dry_run and updates:
        with conn:
            conn.executemany("UPDATE question_bank SET options = ?, correct_answer = ? WHERE id = ?", updates)
    conn.close()

    return {"total": len(rows), "updated": len(updates), "skipped": skipped,
            "before": dict(sorted(before.items())), "after": dict(sorted(after.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=QUIZ_ENGINE_DB_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Değişiklik yazmadan sonucu göster")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        sys.exit(f"Veritabanı bulunamadı: {args.db}")

    if not args.dry_run and not args.no_backup:
        backup = f"{args.db}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(args.db, backup)
        print(f"Yedek: {backup}")

    result = rebalance(args.db, dry_run=args.dry_run, seed=args.seed)
    print(f"Toplam: {result['total']}, güncellenen: {result['updated']}, atlanan: {len(result['skipped'])}")
    print(f"Önce : {result['before']}")
    print(f"Sonra: {result['after']}")
    for q_id, reason in result["skipped"]:
        print(f"  atlandı #{q_id}: {reason}")
    if args.dry_run:
        print("(dry-run: hiçbir şey yazılmadı)")


if __name__ == "__main__":
    main()
