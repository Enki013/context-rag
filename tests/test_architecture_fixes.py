"""
tests/test_architecture_fixes.py
--------------------------------
Mimari incelemede bulunan kritik hataların regresyon testleri:
şık konumu yanlılığı, context/kaynak sayfa tutarlılığı, offset kayması,
örnekleme kapsamı, stale bölüm indeksi, populate, hile koruması ve admin uçları.
"""

import os
import random
import re
from collections import Counter
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import config
from api.main import app
from src.chunker import chunk_text
from src.dataset_search import _sample_windows
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
        "evidence": "Metinden birebir alınmış kanıt cümlesi.",
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


def test_sample_windows_are_contiguous_and_cover_whole_range():
    items = list(range(100))
    for seed in range(20):
        picked = _sample_windows(items, n_windows=4, window=5, rng=random.Random(seed))
        assert len(picked) == 20 and picked == sorted(picked)
        windows = [picked[i:i + 5] for i in range(0, 20, 5)]
        for w in windows:
            assert w == list(range(w[0], w[0] + 5))  # her pencere ardışık
        for i, w in enumerate(windows):  # her pencere kendi çeyreğinde: baş, orta, son temsil edilir
            assert i * 25 <= w[0] and w[-1] < (i + 1) * 25
    assert _sample_windows(list(range(12)), n_windows=4, window=5) == list(range(12))


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
        assert resp.status_code == 202
        job = resp.json()
        assert job["kind"] == "populate" and job["status"] == "queued"

        from src.jobs import run_pending
        assert run_pending() == 1
        done = client.get(job["poll_url"]).json()
    assert done["status"] == "completed"
    assert done["result"] == {"generated": 2, "total_pool_size": 5}


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

from src.quiz_generator import _evidence_in_context, _verify_questions

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


def _indexed_chunks(n, start=0):
    return [{"index": start + i, "text": f"Paragraf {start + i}. " + "kelime " * 120} for i in range(n)]


def test_generation_runs_in_phases_and_retries_short_groups():
    """Önce tüm gruplar üretilir, sonra hepsi doğrulanır; eksik kalanlar için ikinci tur yapılır."""
    events = []
    counter = {"n": 0}

    def fake_llm(context, count, language, difficulty="medium", rejects=None):
        counter["n"] += 1
        events.append("gen")
        return [dict(_photo_q(), question=f"Soru {counter['n']}-{i} nedir acaba?") for i in range(count)]

    def fake_verify(questions, context, rejects=None):
        events.append("ver")
        return questions[:1]  # her grupta tur başına yalnızca bir soru geçiyor

    chunks = _indexed_chunks(10)  # 2 ardışık grup
    with patch("src.quiz_generator._call_llm", side_effect=fake_llm), \
            patch("src.quiz_generator._verify_questions", side_effect=fake_verify), \
            patch("src.quiz_generator.VERIFY_ANSWERS", True):
        out = generate_quiz(chunks, num_questions=4)
    assert events == ["gen", "gen", "ver", "ver", "gen", "gen", "ver", "ver"]
    assert len(out) == 4


def test_groups_never_mix_non_adjacent_chunks():
    from src.quiz_generator import _select_chunk_groups

    chunks = _indexed_chunks(5, 0) + _indexed_chunks(5, 100) + _indexed_chunks(3, 200)
    groups = _select_chunk_groups(chunks, num_questions=10, randomize=False)
    for g in groups:
        idx = [c["index"] for c in g]
        assert idx == list(range(idx[0], idx[0] + len(idx)))
    assert sorted(len(g) for g in groups) == [3, 5, 5]


def test_dedup_removes_exact_and_semantic_duplicates(monkeypatch):
    from src.quiz_generator import _deduplicate

    monkeypatch.setattr("src.quiz_generator.DEDUP_SIMILARITY_THRESHOLD", 0.85)
    a = dict(_q(1), question="Why did Victor read Agrippa?")
    a_paraphrase = dict(_q(2), question="For what reason did Victor study Agrippa?")
    a_exact = dict(_q(3), question="  why did victor read AGRIPPA? ")
    b = dict(_q(4), question="Who is the King of Bohemia?")
    pool = [dict(_q(5), question="What did the King want back?")]
    pool_like = dict(_q(6), question="What was the King trying to recover?")

    vectors = {  # sahte embedding: benzer sorular aynı yöne bakıyor
        "Why did Victor": [1, 0, 0], "For what reason": [0.95, 0.31, 0],
        "Who is the King": [0, 1, 0], "What did the King": [0, 0, 1], "What was the King": [0.1, 0, 0.99],
    }

    def fake_embed(texts):
        return [next(v for k, v in vectors.items() if t.startswith(k)) for t in texts]

    with patch("src.quiz_generator._get_embeddings_batch", side_effect=fake_embed):
        kept = _deduplicate([a, a_paraphrase, a_exact, b, pool_like], existing=pool)
    assert [q["question"] for q in kept] == [a["question"], b["question"]]


def test_dedup_falls_back_to_exact_when_embedding_fails(monkeypatch):
    from src.quiz_generator import _deduplicate

    monkeypatch.setattr("src.quiz_generator.DEDUP_SIMILARITY_THRESHOLD", 0.85)
    qs = [dict(_q(1), question="Soru bir nedir?"), dict(_q(2), question="soru bir nedir?"),
          dict(_q(3), question="Soru iki nedir?")]
    with patch("src.quiz_generator._get_embeddings_batch", side_effect=RuntimeError("ollama yok")):
        kept = _deduplicate(qs)
    assert [q["question"] for q in kept] == ["Soru bir nedir?", "Soru iki nedir?"]


def test_pool_generation_excludes_existing_questions(temp_db):
    _register(temp_db, ["Bölüm metni " * 50])
    save_questions_to_pool("b1", 1, [dict(_q(0), question="Havuzdaki soru nedir?")], db_path=temp_db)
    with patch("src.question_bank.generate_quiz", return_value=[_q(1)]) as gen:
        populate_chapter_pool("b1", 1, target_count=2, db_path=temp_db)
    excluded = gen.call_args.kwargs["exclude_questions"]
    assert [q["question"] for q in excluded] == ["Havuzdaki soru nedir?"]


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
    import json as _json
    dumped = _json.dumps(schema, ensure_ascii=False)
    assert '"orta"' not in dumped and '"kolay"' not in dumped and '"zor"' not in dumped


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


def test_dedup_catches_same_question_with_conflicting_answer(monkeypatch):
    from src.quiz_generator import _deduplicate

    monkeypatch.setattr("src.quiz_generator.DEDUP_SIMILARITY_THRESHOLD", 0.85)
    monkeypatch.setattr("src.quiz_generator.DEDUP_QUESTION_THRESHOLD", 0.95)
    first = dict(_q(1), question="What is Holmes' opinion of love?")
    second = dict(_q(2), question="What does Holmes think about love?")
    second["options"] = ["A) Useful for observers", "B) x", "C) y", "D) z"]

    def fake_embed(texts):
        # Soru metinleri neredeyse aynı yöne, soru+cevap metinleri farklı yönlere bakıyor
        out = []
        for t in texts:
            if "Ankara" in t:
                out.append([1, 0, 0])
            elif "Useful" in t:
                out.append([0, 1, 0])
            else:
                out.append([0, 0, 1] if "opinion" in t else [0.1, 0, 0.995])
        return out

    with patch("src.quiz_generator._get_embeddings_batch", side_effect=fake_embed):
        kept = _deduplicate([first, second])
    assert [q["question"] for q in kept] == [first["question"]]


# ─── Asenkron üretim işleri ─────────────────────────────────────────────────


def test_serve_enqueues_job_when_pool_is_short_and_serves_after(temp_db):
    from src.jobs import run_pending

    register_book_content(
        book_id="async_book", title="T", author="A", source_format="epub",
        chapters=[{"chapter_index": 1, "title": "B1", "raw_text": "Bölüm metni " * 80}], db_path=temp_db,
    )
    url = "/api/v1/quiz/serve?book_id=async_book&chapter_index=1&count=3"
    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db), \
            patch("src.question_bank.generate_quiz", side_effect=lambda chunks, num_questions, **kw:
                  [dict(_q(100 + i), evidence="kanıt") for i in range(num_questions)]) as gen:
        first = client.get(url)
        second = client.get(url)  # aynı bölüm için ikinci istek aynı işe bağlanmalı
        assert first.status_code == 202 and first.headers["Retry-After"]
        assert first.json()["job_id"] == second.json()["job_id"]
        assert gen.call_count == 0  # istek sırasında LLM çağrılmadı

        assert run_pending() == 1
        job = client.get(first.json()["poll_url"]).json()
        assert job["status"] == "completed" and job["result"]["total_pool_size"] == 10
        served = client.get(url)
    assert served.status_code == 200 and served.json()["cache_hit"] is True
    assert len(served.json()["questions"]) == 3


def test_quizzes_endpoint_is_async(temp_db):
    from src.jobs import run_pending

    fake_chunks = [{"index": i, "text": f"Metin {i} " * 40, "source_pages": [], "source_chapter": "B1",
                    "progress_percent": 0.0} for i in range(3)]
    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db), \
            patch("src.dataset_search.get_chunks_for_book", return_value=fake_chunks), \
            patch("src.quiz_generator.generate_quiz", return_value=[_q(1), _q(2)]):
        created = client.post("/api/v1/quizzes", json={"source": {"type": "book", "book_id": "1661"},
                                                       "question_count": 2})
        assert created.status_code == 202
        body = created.json()
        assert body["id"].startswith("quiz_") and body["status"] == "queued"
        assert client.get(body["poll_url"]).status_code == 202

        run_pending()
        done = client.get(body["poll_url"])
    assert done.status_code == 200
    quiz = done.json()
    assert quiz["status"] == "completed" and quiz["question_count"] == 2
    assert quiz["questions"][0]["correct_answer"] in "ABCD"


def test_failed_job_reports_error_and_interrupted_jobs_are_requeued(temp_db):
    from src import jobs

    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db):
        job = jobs.enqueue("populate", {"book_id": "missing", "chapter_index": 1})
        jobs.run_pending()
        failed = jobs.get_job(job["id"])
        assert failed["status"] == "failed" and "Bölüm metni bulunamadı" in failed["error"]

        stuck = jobs.enqueue("populate", {"book_id": "x", "chapter_index": 2})
        jobs._claim_next()  # süreç bu iş çalışırken çöktü
        assert jobs.get_job(stuck["id"])["status"] == "running"
        assert jobs.requeue_interrupted() == 1
        assert jobs.get_job(stuck["id"])["status"] == "queued"


def test_background_worker_processes_queue(temp_db):
    import time as _time
    from src import jobs

    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db), \
            patch("src.jobs.HANDLERS", {**jobs.HANDLERS, "populate": lambda params, db: {"ok": params["chapter_index"]}}):
        worker = jobs.JobWorker(idle_seconds=0.05)
        worker.start()
        try:
            job = jobs.enqueue("populate", {"book_id": "b", "chapter_index": 7})
            worker.notify()
            for _ in range(100):
                if jobs.get_job(job["id"])["status"] == "completed":
                    break
                _time.sleep(0.02)
        finally:
            worker.stop()
        assert jobs.get_job(job["id"])["result"] == {"ok": 7}


def test_generation_runs_are_recorded_with_rejections(temp_db):
    from src.metrics import list_rejected, record_generation_run, summarize_runs

    stats = {"requested": 5, "returned": 2, "accepted": 2, "generated": 6, "llm_calls": 2,
             "structural_rejected": 1, "verify_rejected": 2, "dedup_rejected": 1,
             "seconds": {"generate": 10.0, "verify": 8.0, "dedup": 0.5, "total": 19.0},
             "rejected": [{"stage": "verify", "reason": "no_supported_answer:NONE", "question": _q(1), "context": "ctx"},
                          {"stage": "dedup", "reason": "exact_duplicate", "question": _q(2), "context": None}]}
    run_id = record_generation_run(stats, "pool", "b1", 1, db_path=temp_db)
    assert run_id
    summary = summarize_runs(db_path=temp_db)
    assert summary["runs"] == 1 and summary["totals"]["generated"] == 6
    assert summary["rates_of_generated"]["verify_rejected"] == round(2 / 6, 3)
    assert summary["top_reject_reasons"] == {"verify:no_supported_answer": 1, "dedup:exact_duplicate": 1}
    rejected = list_rejected(stage="verify", db_path=temp_db)
    assert rejected[0]["context"] == "ctx" and rejected[0]["question"]["question"] == _q(1)["question"]


def test_generate_quiz_fills_stats_with_stage_counts():
    chunks = _indexed_chunks(5)
    bad = {"question": "x", "options": ["A) a"], "correct_answer": "A", "explanation": "e"}

    def fake_llm(context, count, language, difficulty="medium", rejects=None):
        rejects.append({"stage": "structural", "reason": "wrong_option_count", "question": bad, "context": context})
        return [_q(1), _q(2)]

    def fake_verify(questions, context, rejects=None):
        rejects.append({"stage": "verify", "reason": "evidence_not_found", "question": questions[1], "context": context})
        return questions[:1]

    stats = {}
    with patch("src.quiz_generator._call_llm", side_effect=fake_llm), \
            patch("src.quiz_generator._verify_questions", side_effect=fake_verify), \
            patch("src.quiz_generator.VERIFY_ANSWERS", True), patch("src.quiz_generator.MAX_GROUP_ATTEMPTS", 1):
        generate_quiz(chunks, num_questions=2, stats=stats)
    assert stats["llm_calls"] == 1 and stats["generated"] == 3
    assert stats["structural_rejected"] == 1 and stats["verify_rejected"] == 1 and stats["accepted"] == 1
    assert set(stats["seconds"]) == {"generate", "verify", "dedup", "total"}


# ─── Doğrulanmamış soruların ayıklanması ve yeniden doğrulama ───────────────


def _unverified(i, text="Soru"):
    q = dict(_q(i), question=f"{text} {i} nedir acaba?")
    q.pop("evidence")
    return q


def test_unverified_and_retired_questions_are_not_served(temp_db):
    import sqlite3

    _register(temp_db, ["bir"])
    save_questions_to_pool("b1", 1, [_q(1), _unverified(2)], db_path=temp_db)
    assert count_pool_questions("b1", 1, db_path=temp_db) == 1
    with patch.object(config, "VERIFY_ANSWERS", False):  # doğrulama kapalıysa eski sorular da sunulur
        assert count_pool_questions("b1", 1, db_path=temp_db) == 2
    with sqlite3.connect(temp_db) as conn:
        conn.execute("UPDATE question_bank SET retired = 1 WHERE evidence IS NOT NULL")
    assert count_pool_questions("b1", 1, db_path=temp_db) == 0
    assert get_cached_questions("b1", 1, count=5, db_path=temp_db) == []


def test_reverify_uses_retrieved_context_and_retires_failures(temp_db):
    import sqlite3
    from src.question_bank import reverify_pool

    chapter = "\n\n".join([f"Paragraf {i}. " + ("dolgu " * 150) for i in range(8)])
    _register(temp_db, [chapter])
    save_questions_to_pool("b1", 1, [_unverified(1, "Geçecek"), _unverified(2, "Kalacak")], db_path=temp_db)
    seen_contexts = []

    def fake_embed(texts):
        # chunk'lar: kendi paragraf numarası yönünde; sorular: hep "Paragraf 5" chunk'ına benzer
        vecs = []
        for t in texts:
            v = [0.0] * 10
            m = re.search(r"Paragraf (\d)", t)
            v[int(m.group(1)) if m else 5] = 1.0
            vecs.append(v)
        return vecs

    def fake_verify(questions, context, rejects=None):
        seen_contexts.append(context)
        q = questions[0]
        if q["question"].startswith("Geçecek"):
            return [dict(q, evidence="Paragraf 5")]
        rejects.append({"stage": "verify", "reason": "no_supported_answer:NONE", "question": q, "context": context})
        return []

    with patch("src.embedder._get_embeddings_batch", side_effect=fake_embed), \
            patch("src.quiz_generator._verify_questions", side_effect=fake_verify):
        summary = reverify_pool(db_path=temp_db)

    assert summary == {"checked": 2, "verified": 1, "retired": 1, "skipped_chapters": []}
    assert all("Paragraf 5" in c for c in seen_contexts)  # en ilgili parça bağlama girdi
    with sqlite3.connect(temp_db) as conn:
        rows = dict(conn.execute("SELECT substr(question, 1, 7), COALESCE(evidence, '') || '|' || retired FROM question_bank"))
    assert rows == {"Geçecek": "Paragraf 5|0", "Kalacak": "|1"}
    assert count_pool_questions("b1", 1, db_path=temp_db) == 1


def test_reverify_skips_chapters_without_text(temp_db):
    from src.question_bank import reverify_pool

    register_book_content(book_id="ghost", title="T", author="A", source_format="epub",
                          chapters=[{"chapter_index": 1, "title": "B1", "raw_text": "bir"}], db_path=temp_db)
    save_questions_to_pool("ghost", 3, [_unverified(1)], db_path=temp_db)
    with patch("src.question_bank._load_chapter_text", side_effect=ValueError("Bölüm metni bulunamadı")):
        summary = reverify_pool(book_id="ghost", db_path=temp_db)
    assert summary["checked"] == 0 and summary["skipped_chapters"][0]["book_id"] == "ghost"


def test_reverify_endpoint_enqueues_job(temp_db):
    with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db), patch.object(config, "ADMIN_API_KEY", "k"):
        assert client.post("/api/v1/quiz/reverify").status_code == 403
        resp = client.post("/api/v1/quiz/reverify?book_id=1661", headers={"X-Admin-Key": "k"})
    assert resp.status_code == 202
    assert resp.json()["kind"] == "reverify" and resp.json()["params"] == {"book_id": "1661", "chapter_index": None}


# ─── EPUB içindekilerle bölüm tespiti ───────────────────────────────────────


def test_heading_core_handles_captions_and_missing_space():
    from src.dataset_search import _heading_core

    assert _heading_core("I hope Mr. Bingley will like it. CHAPTER II.") == "CHAPTER II."
    assert _heading_core("“On the Stairs.” CHAPTERXXVII.") == "CHAPTER XXVII."
    assert _heading_core("I. A SCANDAL IN BOHEMIA") == "I. A SCANDAL IN BOHEMIA"


def _book_text(n_chapters, para=900):
    contents = "Contents\n\n" + "\n".join(f"CHAPTER {i}" for i in range(1, n_chapters + 1)) + "\n\n"
    body = "".join(f"\n\nCHAPTER {i}\n\n" + (f"Story of part {i}. " * (para // 18)) for i in range(1, n_chapters + 1))
    return contents + body


def test_locate_titles_skips_contents_block_and_ignores_far_decoy():
    from src.dataset_search import _locate_titles

    text = _book_text(6) + "\n\nAfterword mentions The Book Title again."
    titles = ["The Book Title"] + [f"CHAPTER {i}" for i in range(1, 7)]
    located = _locate_titles(text, titles)
    names = [t for _, t in located]
    assert names == [f"CHAPTER {i}" for i in range(1, 7)]  # sondaki sahte eşleşme bölümleri yutmadı
    contents_end = text.index("\n\n\n")
    assert all(pos > contents_end for pos, _ in located)  # içindekiler listesindeki geçişler atlandı


def test_chapters_from_epub_toc_merges_container_headings(tmp_path):
    from src.dataset_search import chapters_from_epub_toc

    text = _book_text(4).replace("\n\nCHAPTER 3\n", "\n\nPART TWO\n\nCHAPTER 3\n")
    toc = {1: ["Contents", "CHAPTER 1", "CHAPTER 2", "PART TWO", "CHAPTER 3", "CHAPTER 4"]}
    with patch("src.dataset_search._toc_levels", return_value={1: [t for t in toc[1] if t != "Contents"]}):
        chapters = chapters_from_epub_toc(text, str(tmp_path / "x.epub"))
    assert [c["title"] for c in chapters] == ["CHAPTER 1", "CHAPTER 2", "CHAPTER 3", "CHAPTER 4"]
    assert all(c["detection"] == "epub_toc" for c in chapters)
    assert chapters[2]["text"].startswith("CHAPTER 3")
    assert text[chapters[1]["start_char"]:chapters[1]["end_char"]] == chapters[1]["text"]


def test_detect_chapters_prefers_method_with_more_real_chapters(tmp_path):
    from src.dataset_search import detect_chapters

    epub = tmp_path / "1.epub"
    epub.write_bytes(b"PK" + b"0" * 2000)
    no_headings = "Just prose without any headings. " * 2000
    two_toc = [{"chapter_index": 1, "title": "A", "detection": "epub_toc"},
               {"chapter_index": 2, "title": "B", "detection": "epub_toc"}]

    with patch("src.dataset_search.download_book_epub", return_value=str(epub)), \
            patch("src.dataset_search.chapters_from_epub_toc", return_value=two_toc):
        # regex yalnızca "4 eşit parça" yedeğine düşüyorsa EPUB kazanır
        assert [c["title"] for c in detect_chapters("1", no_headings)] == ["A", "B"]
        # regex daha çok gerçek bölüm bulduysa regex kazanır
        assert len(detect_chapters("1", _book_text(5))) == 5
    with patch("src.dataset_search.download_book_epub", return_value=None):
        fallback = detect_chapters("1", no_headings)
    assert len(fallback) == 4 and fallback[0]["detection"] == "regex_fallback"


# ─── Eski kayıtlı Gutenberg kitaplarının bölümlerini yenileme ────────────────


def _gutenberg_info(titles, title="The War of the Worlds"):
    return {"book_id": "36", "title": title, "author": "Wells, H. G.",
            "chapters": [{"chapter_index": i + 1, "title": t, "raw_text": f"{t} metni " * 50,
                          "progress_percent": i * 10.0} for i, t in enumerate(titles)]}


def test_refresh_gutenberg_book_replaces_stale_chapters_and_title(temp_db):
    from src import question_bank as qb

    register_book_content(
        book_id="36", title="Kitap #36", author="Bilinmiyor", source_format="gutenberg",
        chapters=[{"chapter_index": i, "title": f"Bölüm {i} (%{(i - 1) * 25})", "raw_text": f"eski {i}"}
                  for i in range(1, 5)],
        db_path=temp_db,
    )
    save_questions_to_pool("36", 1, [_q(1)], db_path=temp_db)
    qb._refreshed_books.clear()
    new_titles = [f"{r}. CHAPTER" for r in ("I", "II", "III", "IV", "V", "VI")]
    with patch("src.dataset_search.get_book_chapters", return_value=_gutenberg_info(new_titles)):
        assert qb.refresh_gutenberg_book("36", db_path=temp_db) is True
        assert qb.refresh_gutenberg_book("36", db_path=temp_db) is False  # süreç başına bir kez

    book = qb.get_book("36", db_path=temp_db)
    assert book["title"] == "The War of the Worlds"
    assert [c["title"] for c in qb.get_book_chapters("36", db_path=temp_db)] == new_titles
    assert count_pool_questions("36", 1, db_path=temp_db) == 0  # sınırı değişen bölümün eski soruları silindi


def test_refresh_is_noop_when_chapters_match_or_not_gutenberg(temp_db):
    from src import question_bank as qb

    titles = ["CHAPTER I", "CHAPTER II"]
    register_book_content(book_id="36", title="The War of the Worlds", author="Wells", source_format="gutenberg",
                          chapters=_gutenberg_info(titles)["chapters"], db_path=temp_db)
    save_questions_to_pool("36", 1, [_q(1)], db_path=temp_db)
    register_book_content(book_id="mine", title="Benim Kitabım", author="A", source_format="epub",
                          chapters=[{"chapter_index": 1, "title": "B1", "raw_text": "x"}], db_path=temp_db)
    qb._refreshed_books.clear()
    with patch("src.dataset_search.get_book_chapters", return_value=_gutenberg_info(titles)) as gut:
        assert qb.refresh_gutenberg_book("36", db_path=temp_db) is False
        assert qb.refresh_gutenberg_book("mine", db_path=temp_db) is False
    assert gut.call_count == 1  # içe aktarılmış (epub) kitap için Gutenberg'e hiç bakılmadı
    assert count_pool_questions("36", 1, db_path=temp_db) == 1
