"""
tests/test_architecture_fixes.py
--------------------------------
Mimari incelemede bulunan kritik hataların regresyon testleri:
şık konumu yanlılığı, context/kaynak sayfa tutarlılığı, offset kayması,
örnekleme kapsamı, stale bölüm indeksi, populate, hile koruması ve admin uçları.
"""

import random
from collections import Counter
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app
from src.chunker import chunk_text
from src.dataset_search import _stratified_sample
from src.question_bank import (
    count_pool_questions,
    get_cached_questions,
    init_question_bank_db,
    populate_chapter_pool,
    register_book_content,
    save_questions_to_pool,
)
from src.quiz_generator import (
    _build_context,
    _shuffle_options,
    _strip_overlap,
    _validate_question,
    generate_quiz,
)

client = TestClient(app)


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "engine.db")
    init_question_bank_db(db_file)
    return db_file


def _q(i=0, correct="A"):
    return {
        "question": f"Bu metinde soru numarası {i} nedir?",
        "options": ["A) Ankara", "B) İstanbul", "C) İzmir", "D) Bursa"],
        "correct_answer": correct,
        "explanation": "Metinde açıkça belirtilmiştir.",
    }


# ─── Şık karıştırma ─────────────────────────────────────────────────────────


def test_shuffle_keeps_correct_answer_text():
    rng = random.Random(0)
    for _ in range(50):
        q = _shuffle_options(_q(correct="B"), rng)
        idx = "ABCD".index(q["correct_answer"])
        assert q["options"][idx] == f"{q['correct_answer']}) İstanbul"
        assert [o[:3] for o in q["options"]] == ["A) ", "B) ", "C) ", "D) "]


def test_shuffle_balances_answer_positions():
    rng = random.Random(42)
    positions = Counter(_shuffle_options(_q(correct="A"), rng)["correct_answer"] for _ in range(2000))
    for letter in "ABCD":
        assert 0.20 < positions[letter] / 2000 < 0.30


def test_shuffle_handles_unprefixed_options():
    q = {"question": "x?", "options": ["Ankara", "İstanbul", "İzmir", "Bursa"], "correct_answer": "C",
         "explanation": "."}
    q = _shuffle_options(q, random.Random(1))
    assert q["options"]["ABCD".index(q["correct_answer"])].endswith("İzmir")


def test_duplicate_options_rejected():
    q = _q()
    q["options"] = ["A) Ankara", "B) ankara", "C) İzmir", "D) Bursa"]
    assert _validate_question(q, "Turkish") is False


def test_generate_quiz_adds_provenance_and_shuffles():
    chunks = [{"index": i, "text": f"Metin parçası {i}. " * 20, "source_pages": [i + 1]} for i in range(3)]
    with patch("src.quiz_generator._call_llm", return_value=[_q(correct="A")]):
        out = generate_quiz(chunks, num_questions=1)
    q = out[0]
    assert q["source_chunk_indices"] == [0, 1, 2]
    assert q["model"] and q["prompt_version"]
    assert q["options"]["ABCD".index(q["correct_answer"])].endswith("Ankara")


# ─── Context / kaynak sayfa tutarlılığı ─────────────────────────────────────


def test_build_context_reports_only_included_chunks():
    chunks = [{"index": i, "text": chr(65 + i) * 1000, "source_pages": [i + 1]} for i in range(5)]
    context, included = _build_context(chunks)
    assert len(context) <= 4000
    assert [c["index"] for c in included] == [0, 1, 2]
    assert "E" * 10 not in context


def test_strip_overlap_removes_splitter_overlap():
    prev = "Birinci cümle burada. İkinci cümle de burada bitiyor."
    curr = "İkinci cümle de burada bitiyor. Üçüncü cümle yeni."
    assert _strip_overlap(prev, curr) == "Üçüncü cümle yeni."
    assert _strip_overlap("tamamen farklı", curr) == curr


# ─── Offset ve örnekleme ────────────────────────────────────────────────────


def test_chunk_start_char_matches_real_position():
    text = "\n\n".join(f"Paragraf {i}. " + ("kelime " * 60) for i in range(80))
    for c in chunk_text(text):
        assert text[c["start_char"]: c["start_char"] + c["length"]] == c["text"]


def test_stratified_sample_covers_whole_range():
    items = list(range(39))
    for seed in range(20):
        picked = _stratified_sample(items, 20, random.Random(seed))
        assert len(picked) == 20
        assert picked == sorted(picked)
        assert picked[-1] >= 37  # son dilim her zaman temsil edilir
        assert picked[0] <= 1


# ─── Soru bankası: stale indeks ve populate ─────────────────────────────────


def _register(db, texts):
    register_book_content(
        book_id="b1", title="T", author="A", source_format="epub",
        chapters=[{"chapter_index": i + 1, "title": f"B{i + 1}", "raw_text": t} for i, t in enumerate(texts)],
        db_path=db,
    )


def test_reregister_same_text_keeps_pool(temp_db):
    _register(temp_db, ["bir", "iki"])
    save_questions_to_pool("b1", 1, [_q()], db_path=temp_db)
    _register(temp_db, ["bir", "iki"])
    assert count_pool_questions("b1", 1, db_path=temp_db) == 1


def test_reregister_changed_chapter_drops_stale_questions(temp_db):
    _register(temp_db, ["bir", "iki", "üç"])
    for idx in (1, 2, 3):
        save_questions_to_pool("b1", idx, [_q(idx)], db_path=temp_db)

    # Bölümleme değişti: 2. bölümün metni farklı, 3. bölüm artık yok
    _register(temp_db, ["bir", "YENİ İKİ"])

    assert count_pool_questions("b1", 1, db_path=temp_db) == 1
    assert count_pool_questions("b1", 2, db_path=temp_db) == 0
    assert count_pool_questions("b1", 3, db_path=temp_db) == 0
    from src.question_bank import get_chapter
    assert get_chapter("b1", 3, db_path=temp_db) is None


def test_saved_questions_store_version_fields(temp_db):
    _register(temp_db, ["bir"])
    q = dict(_q(), model="m", prompt_version="v1", source_chunk_indices=[3, 4])
    save_questions_to_pool("b1", 1, [q], db_path=temp_db)
    row = get_cached_questions("b1", 1, count=1, db_path=temp_db)[0]
    assert row["model"] == "m" and row["prompt_version"] == "v1"
    assert row["source_chunk_indices"] == "[3, 4]"


def test_populate_fills_partially_filled_pool(temp_db):
    _register(temp_db, ["Bölüm metni " * 50])
    save_questions_to_pool("b1", 1, [_q(0)], db_path=temp_db)

    with patch("src.question_bank.generate_quiz", side_effect=lambda chunks, num_questions, **kw:
               [_q(100 + i) for i in range(num_questions)]) as mock_gen:
        result = populate_chapter_pool("b1", 1, target_count=5, db_path=temp_db)

    assert mock_gen.call_args.kwargs["num_questions"] == 4
    assert result == {"generated": 4, "total_pool_size": 5}

    with patch("src.question_bank.generate_quiz") as mock_gen:
        result = populate_chapter_pool("b1", 1, target_count=5, db_path=temp_db)
    assert mock_gen.call_count == 0
    assert result["generated"] == 0


# ─── Hile koruması ve admin uçları ──────────────────────────────────────────


def _seed_api_book(db):
    register_book_content(
        book_id="api_sec", title="T", author="A", source_format="epub",
        chapters=[{"chapter_index": 1, "title": "B1", "raw_text": "Metin"}], db_path=db,
    )
    save_questions_to_pool("api_sec", 1, [_q(i) for i in range(3)], db_path=db)


def test_serve_with_answers_requires_admin_key(temp_db):
    _seed_api_book(temp_db)
    url = "/api/v1/quiz/serve?book_id=api_sec&chapter_index=1&count=2&hide_answers=false"
    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db), patch.object(config, "ADMIN_API_KEY", "s3cret"):
        assert client.get(url).status_code == 403
        assert client.get(url, headers={"X-Admin-Key": "wrong"}).status_code == 403
        resp = client.get(url, headers={"X-Admin-Key": "s3cret"})
        assert resp.status_code == 200
        assert resp.json()["questions"][0]["correct_answer"] is not None


def test_admin_endpoints_closed_without_configured_key():
    with patch.object(config, "ADMIN_API_KEY", None):
        resp = client.post("/api/v1/quiz/populate?book_id=x&chapter_index=1", headers={"X-Admin-Key": "anything"})
        assert resp.status_code == 403
        files = {"file": ("a.epub", b"data", "application/epub+zip")}
        assert client.post("/api/v1/books/import", files=files).status_code == 403


def test_populate_endpoint_with_admin_key(temp_db):
    _seed_api_book(temp_db)
    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db), patch.object(config, "ADMIN_API_KEY", "k"), \
            patch("src.question_bank.generate_quiz", return_value=[_q(9), _q(10)]):
        resp = client.post(
            "/api/v1/quiz/populate?book_id=api_sec&chapter_index=1&target_count=5",
            headers={"X-Admin-Key": "k"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["generated"] == 2
    assert body["total_pool_size"] == 5


# ─── Eski havuzu dengeleme scripti ──────────────────────────────────────────


def test_rebalance_script_keeps_correct_text(temp_db):
    from scripts.rebalance_answer_positions import rebalance

    _register(temp_db, ["bir"])
    save_questions_to_pool("b1", 1, [_q(i, correct="A") for i in range(200)], db_path=temp_db)
    bad = dict(_q(999), options=["A) x", "B) y"])  # 4 şık değil -> atlanmalı
    save_questions_to_pool("b1", 1, [bad], db_path=temp_db)

    result = rebalance(temp_db, seed=7)
    assert result["updated"] == 200 and len(result["skipped"]) == 1
    assert all(n > 30 for n in result["after"].values())

    for q in get_cached_questions("b1", 1, count=500, db_path=temp_db):
        if len(q["options"]) == 4:
            assert q["options"]["ABCD".index(q["correct_answer"])].endswith("Ankara")
