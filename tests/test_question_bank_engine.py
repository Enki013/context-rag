"""
tests/test_question_bank_engine.py
----------------------------------
Kalıcı Soru Bankası, Kitap Kaydı ve Headless Quiz Dağıtım Motoru testleri.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app
from src.question_bank import (
    count_pool_questions,
    evaluate_answers_against_bank,
    get_book,
    get_book_chapters,
    get_cached_questions,
    init_question_bank_db,
    register_book_content,
    save_questions_to_pool,
    serve_chapter_quiz,
)

client = TestClient(app)


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_engine.db")
    init_question_bank_db(db_file)
    return db_file


def test_book_registration_and_chapters(temp_db):
    chapters = [
        {
            "chapter_index": 1,
            "title": "Bölüm 1: Giriş",
            "start_page": 1,
            "end_page": 15,
            "progress_percent": 0.0,
            "raw_text": "Bu birinci bölümün metnidir. Karakterler tanıtılır ve yolculuk başlar.",
        },
        {
            "chapter_index": 2,
            "title": "Bölüm 2: Karşılaşma",
            "start_page": 16,
            "end_page": 35,
            "progress_percent": 15.5,
            "raw_text": "Bu ikinci bölümün metnidir. Gizemli bir yabancı kasabaya gelir.",
        },
    ]

    reg = register_book_content(
        book_id="book_test_101",
        title="Test Romanı",
        author="Yazar Adı",
        source_format="epub",
        chapters=chapters,
        total_pages=35,
        db_path=temp_db,
    )

    assert reg["book_id"] == "book_test_101"
    assert reg["total_chapters"] == 2

    book_meta = get_book("book_test_101", db_path=temp_db)
    assert book_meta is not None
    assert book_meta["title"] == "Test Romanı"

    ch_list = get_book_chapters("book_test_101", db_path=temp_db)
    assert len(ch_list) == 2
    assert ch_list[0]["chapter_index"] == 1
    assert ch_list[0]["pool_count"] == 0


def test_question_pooling_and_cache_hit(temp_db):
    # Kitap ve bölüm ekle
    register_book_content(
        book_id="book_pool_test",
        title="Havuz Testi",
        author="Yazar",
        source_format="pdf",
        chapters=[{
            "chapter_index": 1,
            "title": "Bölüm 1",
            "raw_text": "Metin...",
        }],
        db_path=temp_db,
    )

    # 10 soru üretip havuza kaydet
    mock_questions = [
        {
            "question": f"Soru {i}?",
            "options": ["A) Şık 1", "B) Şık 2", "C) Şık 3", "D) Şık 4"],
            "correct_answer": "A",
            "explanation": f"Açıklama {i}",
            "source_pages": [i],
        }
        for i in range(1, 11)
    ]

    saved = save_questions_to_pool(
        book_id="book_pool_test",
        chapter_index=1,
        questions=mock_questions,
        difficulty="orta",
        db_path=temp_db,
    )
    assert saved == 10
    assert count_pool_questions("book_pool_test", 1, difficulty="orta", db_path=temp_db) == 10

    # Havuzdan 5 soru iste (Cache Hit olmalı ve LLM çağrılmamalı)
    with patch("src.question_bank.generate_quiz") as mock_llm:
        quiz = serve_chapter_quiz(
            book_id="book_pool_test",
            chapter_index=1,
            count=5,
            difficulty="orta",
            hide_answers=True,
            db_path=temp_db,
        )

        assert quiz["cache_hit"] is True
        assert quiz["question_count"] == 5
        assert len(quiz["questions"]) == 5
        assert mock_llm.call_count == 0  # LLM ASLA çağrılmadı!

        # Hile koruması: doğru cevaplar gizlenmeli
        for q in quiz["questions"]:
            assert "correct_answer" not in q
            assert "explanation" not in q
            assert "id" in q


def test_answer_evaluation_against_bank(temp_db):
    # Soru ekle
    register_book_content(
        book_id="book_eval",
        title="Eval Kitabı",
        author="Yazar",
        source_format="epub",
        chapters=[{"chapter_index": 1, "title": "Bölüm 1", "raw_text": "..."}],
        db_path=temp_db,
    )

    mock_q = [
        {
            "question": "Başkent neresidir?",
            "options": ["A) Ankara", "B) İstanbul", "C) İzmir", "D) Bursa"],
            "correct_answer": "A",
            "explanation": "Türkiye Cumhuriyeti'nin başkenti Ankara'dır.",
        },
        {
            "question": "Hangisi bir asal sayıdır?",
            "options": ["A) 4", "B) 6", "C) 7", "D) 9"],
            "correct_answer": "C",
            "explanation": "7 bir asal sayıdır.",
        },
    ]
    save_questions_to_pool("book_eval", 1, mock_q, db_path=temp_db)
    cached = get_cached_questions("book_eval", 1, count=2, db_path=temp_db)

    q1 = next(q for q in cached if "Başkent" in q["question"])
    q2 = next(q for q in cached if "asal" in q["question"])

    # Kullanıcı cevapları: 1. Doğru, 2. Yanlış
    answers = [
        {"question_id": q1["id"], "selected_option": "A) Ankara"},
        {"question_id": q2["id"], "selected_option": "B) 6"},
    ]

    result = evaluate_answers_against_bank(answers, db_path=temp_db)
    assert result["total_questions"] == 2
    assert result["correct_count"] == 1
    assert result["score_percentage"] == 50.0

    d1 = next(d for d in result["details"] if d["question_id"] == q1["id"])
    assert d1["is_correct"] is True
    assert d1["correct_answer"] == "A"
    assert "Ankara" in d1["explanation"]

    d2 = next(d for d in result["details"] if d["question_id"] == q2["id"])
    assert d2["is_correct"] is False
    assert d2["correct_answer"] == "C"


class TestAPIHeadlessEndpoints:
    def test_serve_quiz_endpoint(self, temp_db):
        register_book_content(
            book_id="api_book",
            title="API Kitap",
            author="Yazar",
            source_format="epub",
            chapters=[{"chapter_index": 1, "title": "Bölüm 1", "raw_text": "Metin"}],
            db_path=temp_db,
        )
        save_questions_to_pool(
            "api_book",
            1,
            [
                {
                    "question": f"Soru {i}?",
                    "options": ["A) 1", "B) 2", "C) 3", "D) 4"],
                    "correct_answer": "A",
                    "explanation": "Açıklama",
                }
                for i in range(5)
            ],
            db_path=temp_db,
        )

        with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db):
            resp = client.get("/api/v1/quiz/serve?book_id=api_book&chapter_index=1&count=3")
            assert resp.status_code == 200
            data = resp.json()
            assert data["book_id"] == "api_book"
            assert data["cache_hit"] is True
            assert len(data["questions"]) == 3
            assert data["questions"][0]["correct_answer"] is None  # Gizli

    def test_evaluate_answers_endpoint(self, temp_db):
        register_book_content(
            book_id="api_eval_book",
            title="Eval",
            author="Y",
            source_format="epub",
            chapters=[{"chapter_index": 1, "title": "B1", "raw_text": "M"}],
            db_path=temp_db,
        )
        save_questions_to_pool(
            "api_eval_book",
            1,
            [{
                "question": "Soru 1?",
                "options": ["A) Doğru", "B) Yanlış", "C) X", "D) Y"],
                "correct_answer": "A",
                "explanation": "A doğru.",
            }],
            db_path=temp_db,
        )
        q = get_cached_questions("api_eval_book", 1, count=1, db_path=temp_db)[0]

        with patch("src.question_bank.QUIZ_ENGINE_DB_PATH", temp_db):
            payload = {
                "answers": [{"question_id": q["id"], "selected_option": "A"}]
            }
            resp = client.post("/api/v1/quiz/evaluate-answers", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert data["correct_count"] == 1
            assert data["score_percentage"] == 100.0
