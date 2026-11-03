"""
scripts/eval_verifier.py
------------------------
Grounding doğrulayıcısını etiketli değerlendirme setinde (evals/verifier_eval.json) ölçer.

Her soru, kendi metin parçasıyla birlikte üretimdeki `_verify_questions` fonksiyonuna verilir.
Doğrulayıcı hatalı soruları reddetmeli, geçerli soruları kabul etmelidir.

Kullanım:
    python scripts/eval_verifier.py                                  # config'deki doğrulayıcı
    QUIZ_VERIFY_MODEL=gemma4:31b QUIZ_VERIFY_THINK=0 python scripts/eval_verifier.py --out sonuc.json
    python scripts/eval_verifier.py --combine a.json b.json          # "hepsi kabul ederse kabul" birleşimi

Not: Üretimde bir gruptaki sorular tek çağrıda doğrulanır; burada her soru ayrı çağrıyla doğrulanır.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_DATASET = os.path.join(ROOT, "evals", "verifier_eval.json")


def load_dataset(path: str = DEFAULT_DATASET) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(dataset: dict, limit: int | None = None) -> list[dict]:
    import src.quiz_generator as qg

    items = dataset["items"][:limit] if limit else dataset["items"]
    results = []
    for i, it in enumerate(items, 1):
        context = dataset["contexts"][it["context_id"]]["context"]
        q = {"question": it["question"], "options": list(it["options"]),
             "correct_answer": it["correct_answer"], "explanation": ""}
        t = time.time()
        accepted = bool(qg._verify_questions([q], context))
        expected_accept = it["label"] == "valid"
        res = {
            "id": it["id"], "label": it["label"], "error_type": it["error_type"],
            "confidence": it["confidence"], "accepted": accepted,
            "correct": accepted == expected_accept, "sec": round(time.time() - t, 1),
        }
        results.append(res)
        mark = "✓" if res["correct"] else "✗"
        print(f"{mark} {i:>2}/{len(items)} {it['id']:<24} {it['label']:<7} {it['error_type'] or '':<14} "
              f"{'KABUL' if accepted else 'RED':<5} {res['sec']}s", flush=True)
    return results


def summarize(results: list[dict], title: str) -> dict:
    def block(rs):
        valid = [r for r in rs if r["label"] == "valid"]
        invalid = [r for r in rs if r["label"] == "invalid"]
        by_type = defaultdict(lambda: [0, 0])
        for r in invalid:
            by_type[r["error_type"]][0] += not r["accepted"]
            by_type[r["error_type"]][1] += 1
        return {
            "invalid_caught": f"{sum(not r['accepted'] for r in invalid)}/{len(invalid)}",
            "valid_kept": f"{sum(r['accepted'] for r in valid)}/{len(valid)}",
            "accuracy": round(sum(r["correct"] for r in rs) / max(len(rs), 1), 3),
            "by_error_type": {k: f"{c}/{n}" for k, (c, n) in sorted(by_type.items())},
        }

    summary = {"title": title, "all": block(results),
               "high_confidence": block([r for r in results if r["confidence"] == "high"])}
    secs = [r["sec"] for r in results if r.get("sec") is not None]
    if secs:
        summary["avg_sec"] = round(sum(secs) / len(secs), 1)

    print(f"\n== {title} ==")
    for name in ("all", "high_confidence"):
        b = summary[name]
        print(f"  [{name}] hatalı yakalanan {b['invalid_caught']} | geçerli korunan {b['valid_kept']} | "
              f"doğruluk {b['accuracy']} | türler {b['by_error_type']}")
    if secs:
        print(f"  ortalama süre: {summary['avg_sec']} sn/soru")
    return summary


def combine(paths: list[str]) -> list[dict]:
    """Birden fazla doğrulayıcının sonucunu 'hepsi kabul ederse kabul' kuralıyla birleştirir."""
    runs = [{r["id"]: r for r in json.load(open(p))["results"]} for p in paths]
    combined = []
    for rid, base in runs[0].items():
        accepted = all(run[rid]["accepted"] for run in runs)
        combined.append({**base, "accepted": accepted,
                         "correct": accepted == (base["label"] == "valid"),
                         "sec": round(sum(run[rid]["sec"] for run in runs), 1)})
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", help="Sonuçları JSON olarak kaydet")
    parser.add_argument("--combine", nargs="+", metavar="RESULT_JSON")
    args = parser.parse_args()

    if args.combine:
        results = combine(args.combine)
        summary = summarize(results, "birleşim (hepsi kabul) : " + " + ".join(
            json.load(open(p))["summary"]["title"] for p in args.combine))
    else:
        logging.basicConfig(level=logging.ERROR)
        import config
        title = config.VERIFY_MODEL + {None: "", False: " (thinking kapalı)", True: " (thinking açık)"}[
            config.VERIFY_THINK]
        results = run(load_dataset(args.dataset), args.limit)
        summary = summarize(results, title)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
