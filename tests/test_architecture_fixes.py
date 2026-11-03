"""
tests/test_architecture_fixes.py
--------------------------------
Mimari incelemede bulunan kritik hataların regresyon testleri:
şık konumu yanlılığı, context/kaynak sayfa tutarlılığı, offset kayması,
örnekleme kapsamı, stale bölüm indeksi, populate, hile koruması ve admin uçları.
"""

import os
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
    with patch("src.quiz_generator._call_llm", return_value=[_q(correct="A")]), \
            patch("src.quiz_generator.VERIFY_ANSWERS", False):
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


# ─── Grounding doğrulaması ──────────────────────────────────────────────────

from src.quiz_generator import _evidence_in_context, _generate_verified_for_group, _verify_questions

CTX = (
    "To-day the King of Bohemia came to Baker Street. He said: \u201cThere is the photograph, "
    "which she threatens to send to my bride.\u201d Holmes promised to recover it within three days."
)


def _chat_reply(answers):
    import json as _json
    return {"message": {"content": _json.dumps({"answers": answers})}}


def _photo_q(correct_text="A photograph", correct="A"):
    opts = ["A) A photograph", "B) Some letters", "C) A ring", "D) A crown"]
    return {"question": "What does the King want Holmes to recover?", "options": opts,
            "correct_answer": correct, "explanation": "."}


def test_evidence_matching_is_tolerant_but_strict():
    assert _evidence_in_context('"There is the photograph, which she threatens"', CTX)  # düz tırnak vs kıvrık
    assert _evidence_in_context("came to Baker Street ... recover it within three days", CTX)
    assert not _evidence_in_context("the King wanted his letters back", CTX)  # metinde yok
    assert not _evidence_in_context("Baker", CTX)  # çok kısa
    assert not _evidence_in_context("", CTX)


def test_verify_keeps_supported_question_with_evidence():
    reply = _chat_reply([{"id": 1, "answer": "A", "evidence": "There is the photograph, which she threatens"}])
    with patch("src.quiz_generator.ollama_client.chat", return_value=reply):
        out = _verify_questions([_photo_q()], CTX)
    assert len(out) == 1 and out[0]["evidence"].startswith("There is the photograph")


def test_verify_rejects_wrong_marked_answer():
    # Generator 'letters' (B) işaretlemiş, doğrulayıcı metne göre A diyor -> reddedilmeli
    reply = _chat_reply([{"id": 1, "answer": "A", "evidence": "There is the photograph, which she threatens"}])
    with patch("src.quiz_generator.ollama_client.chat", return_value=reply):
        assert _verify_questions([_photo_q(correct="B")], CTX) == []


def test_verify_rejects_none_and_fabricated_evidence():
    q1, q2 = _photo_q(), _photo_q()
    reply = _chat_reply([
        {"id": 1, "answer": "NONE", "evidence": ""},
        {"id": 2, "answer": "A", "evidence": "The King demanded the photograph immediately"},  # uydurma alıntı
    ])
    with patch("src.quiz_generator.ollama_client.chat", return_value=reply):
        assert _verify_questions([q1, q2], CTX) == []


def test_verify_fails_closed_on_unparseable_output():
    bad = {"message": {"content": "üzgünüm, yapamam"}}
    with patch("src.quiz_generator.ollama_client.chat", return_value=bad) as chat:
        assert _verify_questions([_photo_q()], CTX) == []
    assert chat.call_count == 2  # bir kez yeniden denedi


def test_group_generation_retries_when_verification_drops_questions():
    calls = {"n": 0}

    def fake_llm(context, count, language, difficulty="medium"):
        calls["n"] += 1
        return [dict(_photo_q(), question=f"Soru {calls['n']}-{i} nedir?") for i in range(count)]

    def fake_verify(questions, context):
        return questions[:1]  # her turda yalnızca biri geçiyor

    with patch("src.quiz_generator._call_llm", side_effect=fake_llm), \
            patch("src.quiz_generator._verify_questions", side_effect=fake_verify), \
            patch("src.quiz_generator.VERIFY_ANSWERS", True):
        out = _generate_verified_for_group(CTX, 3, "English", "medium")
    assert calls["n"] == 2 and len(out) == 2


def test_evidence_saved_to_pool(temp_db):
    _register(temp_db, ["bir"])
    save_questions_to_pool("b1", 1, [dict(_q(), evidence="kanıt cümlesi")], db_path=temp_db)
    assert get_cached_questions("b1", 1, count=1, db_path=temp_db)[0]["evidence"] == "kanıt cümlesi"


# ─── Doğrulayıcı değerlendirme seti ─────────────────────────────────────────


def test_verifier_eval_dataset_is_well_formed():
    from scripts.eval_verifier import load_dataset

    data = load_dataset()
    items, contexts = data["items"], data["contexts"]
    assert len(items) >= 40
    assert len({i["id"] for i in items}) == len(items)
    for it in items:
        assert it["context_id"] in contexts
        assert len(it["options"]) == 4 and it["correct_answer"] in "ABCD"
        assert it["label"] in ("valid", "invalid") and it["confidence"] in ("high", "low")
        if it["label"] == "invalid":
            assert it["error_type"] in ("ambiguous", "premise_error", "wrong_key", "unanswerable")
        else:
            assert it["error_type"] is None
    labels = [i["label"] for i in items]
    assert labels.count("valid") >= 15 and labels.count("invalid") >= 15


def test_eval_summary_and_combine(tmp_path):
    import json as _json
    from scripts.eval_verifier import combine, summarize

    base = [
        {"id": "a", "label": "invalid", "error_type": "wrong_key", "confidence": "high", "sec": 1.0},
        {"id": "b", "label": "valid", "error_type": None, "confidence": "high", "sec": 1.0},
        {"id": "c", "label": "invalid", "error_type": "ambiguous", "confidence": "low", "sec": 1.0},
    ]
    run1 = [dict(r, accepted=acc, correct=acc == (r["label"] == "valid")) for r, acc in zip(base, [False, True, True])]
    run2 = [dict(r, accepted=acc, correct=acc == (r["label"] == "valid")) for r, acc in zip(base, [True, True, False])]
    s = summarize(run1, "m1")
    assert s["all"]["invalid_caught"] == "1/2" and s["all"]["valid_kept"] == "1/1"
    assert s["high_confidence"]["invalid_caught"] == "1/1"

    p1, p2 = tmp_path / "1.json", tmp_path / "2.json"
    p1.write_text(_json.dumps({"summary": {"title": "m1"}, "results": run1}))
    p2.write_text(_json.dumps({"summary": {"title": "m2"}, "results": run2}))
    both = {r["id"]: r["accepted"] for r in combine([str(p1), str(p2)])}
    assert both == {"a": False, "b": True, "c": False}


def test_verify_passes_think_flag_only_when_configured():
    reply = _chat_reply([{"id": 1, "answer": "A", "evidence": "There is the photograph, which she threatens"}])
    with patch("src.quiz_generator.ollama_client.chat", return_value=reply) as chat, \
            patch("src.quiz_generator.VERIFY_THINK", None):
        _verify_questions([_photo_q()], CTX)
    assert "think" not in chat.call_args.kwargs
    with patch("src.quiz_generator.ollama_client.chat", return_value=reply) as chat, \
            patch("src.quiz_generator.VERIFY_THINK", False):
        _verify_questions([_photo_q()], CTX)
    assert chat.call_args.kwargs["think"] is False


# ─── Varsayılan doğrulayıcı ve health ───────────────────────────────────────


def test_default_verifier_is_gemma_without_thinking(monkeypatch):
    import importlib
    monkeypatch.delenv("QUIZ_VERIFY_MODEL", raising=False)
    monkeypatch.delenv("QUIZ_VERIFY_THINK", raising=False)
    fresh = importlib.reload(config)
    try:
        assert fresh.VERIFY_MODEL == "gemma4:31b"
        assert fresh.VERIFY_THINK is False
        monkeypatch.setenv("QUIZ_VERIFY_THINK", "auto")
        assert importlib.reload(config).VERIFY_THINK is None
    finally:
        monkeypatch.delenv("QUIZ_VERIFY_THINK", raising=False)
        importlib.reload(config)


def test_health_reports_missing_verifier_model():
    with patch("api.routes.health._installed_models", return_value={"qwen2.5:14b", "bge-m3"}), \
            patch.object(config, "VERIFY_ANSWERS", True), patch.object(config, "VERIFY_MODEL", "gemma4:31b"), \
            patch("api.routes.health.get_cache_status", return_value={"exists": True, "num_books": 1}):
        body = client.get("/api/v1/health").json()
    assert body["verify_model"] == "gemma4:31b"
    assert body["verify_model_installed"] is False
    assert body["status"] == "degraded"


def test_health_ok_when_verifier_installed():
    with patch("api.routes.health._installed_models", return_value={"qwen2.5:14b", "gemma4:31b"}), \
            patch.object(config, "VERIFY_ANSWERS", True), patch.object(config, "VERIFY_MODEL", "gemma4:31b"), \
            patch("api.routes.health.get_cache_status", return_value={"exists": True, "num_books": 1}):
        body = client.get("/api/v1/health").json()
    assert body["verify_model_installed"] is True and body["status"] == "ok"


# ─── İngilizce zorluk değerleri ─────────────────────────────────────────────


def test_normalize_difficulty_accepts_english_and_legacy():
    from src.difficulty import normalize_difficulty

    assert normalize_difficulty("easy") == "easy"
    assert normalize_difficulty("HARD") == "hard"
    assert normalize_difficulty("orta") == "medium"
    assert normalize_difficulty(" Zor ") == "hard"
    assert normalize_difficulty(None) == "medium"
    with pytest.raises(ValueError):
        normalize_difficulty("extreme")


def test_prompt_difficulty_legacy_alias_matches_english():
    from src.prompts import build_quiz_prompt

    assert build_quiz_prompt("ctx", difficulty="kolay") == build_quiz_prompt("ctx", difficulty="easy")
    assert "Hard / Challenging" in build_quiz_prompt("ctx", difficulty="hard")


def test_openapi_lists_only_english_difficulties():
    schema = client.get("/openapi.json").json()
    assert schema["components"]["schemas"]["Difficulty"]["enum"] == ["easy", "medium", "hard"]
    serve_params = schema["paths"]["/api/v1/quiz/serve"]["get"]["parameters"]
    diff = next(p for p in serve_params if p["name"] == "difficulty")
    assert diff["schema"]["default"] == "medium"
    assert "orta" not in str(schema) and "kolay" not in str(schema)


def test_serve_accepts_english_and_legacy_difficulty(temp_db):
    register_book_content(
        book_id="diff_book", title="T", author="A", source_format="epub",
        chapters=[{"chapter_index": 1, "title": "B1", "raw_text": "Metin"}], db_path=temp_db,
    )
    save_questions_to_pool("diff_book", 1, [_q(i) for i in range(3)], difficulty="hard", db_path=temp_db)
    base = "/api/v1/quiz/serve?book_id=diff_book&chapter_index=1&count=2"
    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db):
        r_en = client.get(base + "&difficulty=hard")
        r_legacy = client.get(base + "&difficulty=zor")  # eski Android sürümleri
        r_bad = client.get(base + "&difficulty=extreme")
    assert r_en.status_code == 200 and r_en.json()["difficulty"] == "hard" and r_en.json()["cache_hit"] is True
    assert r_legacy.status_code == 200 and r_legacy.json()["difficulty"] == "hard"
    assert r_bad.status_code == 422


def test_legacy_difficulty_rows_are_migrated(tmp_path):
    import sqlite3
    import src.question_bank as qb

    db = str(tmp_path / "legacy.db")
    qb.init_question_bank_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO books (id, title) VALUES ('b', 't')")
        for d in ("orta", "zor", "kolay", "medium"):
            conn.execute(
                "INSERT INTO question_bank (book_id, chapter_index, question, options, correct_answer, explanation,"
                " difficulty) VALUES ('b', 1, 'q', '[]', 'A', 'e', ?)", (d,))
    qb._initialized_dbs.discard(os.path.abspath(db))  # süreç yeniden başlamış gibi
    qb.init_question_bank_db(db)
    with sqlite3.connect(db) as conn:
        values = sorted(r[0] for r in conn.execute("SELECT difficulty FROM question_bank"))
    assert values == ["easy", "hard", "medium", "medium"]
