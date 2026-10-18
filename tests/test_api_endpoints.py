"""
tests/test_api_endpoints.py
---------------------------
FastAPI REST API endpoint'leri için entegrasyon testleri.
TestClient ile HTTP istekleri, durum kodları, yanıt şemaları
ve hata senaryoları (400, 404, 422) test edilir.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_health_check(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ollama_connected" in data
        assert "llm_model" in data
        assert "embedding_model" in data
        assert data["catalog_installed"] is True
        assert data["total_books"] > 1000

    def test_api_root(self):
        resp = client.get("/api/v1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Context RAG Quiz API"
        assert data["documentation"] == "/docs"
        assert data["status"] == "online"


class TestBookEndpoints:
    def test_search_books_valid(self):
        resp = client.get("/api/v1/books/search?q=Sherlock&limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "Sherlock"
        assert data["total"] > 0
        assert len(data["books"]) <= 5
        book = data["books"][0]
        assert "book_id" in book
        assert "title" in book
        assert "is_cached_locally" in book

    def test_search_books_empty_query_validation(self):
        # min_length=1 kuralı nedeniyle 422 dönmelidir
        resp = client.get("/api/v1/books/search?q=")
        assert resp.status_code == 422

    def test_get_book_detail_existing(self):
        resp = client.get("/api/v1/books/1661")
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_id"] == "1661"
        assert "Sherlock Holmes" in data["title"]
        assert data["is_cached_locally"] is True
        assert data["total_chapters"] == 12

    def test_get_book_detail_not_found(self):
        resp = client.get("/api/v1/books/999999999")
        assert resp.status_code == 404
        assert "Kitap bulunamadı" in resp.json()["detail"]

    def test_get_book_chapter_count_existing(self):
        resp = client.get("/api/v1/books/1661/chapter-count")
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_id"] == "1661"
        assert data["total_chapters"] == 12
        assert len(data["chapter_indices"]) == 12
        assert data["chapter_indices"] == list(range(1, 13))

    def test_get_book_chapter_count_not_found(self):
        resp = client.get("/api/v1/books/999999999/chapter-count")
        assert resp.status_code == 404

    def test_get_book_chapters_existing(self):
        resp = client.get("/api/v1/books/1661/chapters")
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_id"] == "1661"
        assert data["total_chapters"] == 12
        assert len(data["chapters"]) == 12
        chap = data["chapters"][0]
        assert chap["chapter_index"] == 1
        assert "title" in chap
        assert "progress_percent" in chap
        assert "char_count" in chap

    def test_get_book_chapters_not_found(self):
        resp = client.get("/api/v1/books/999999999/chapters")
        assert resp.status_code == 404


class TestQuizEvaluationEndpoint:
    def test_evaluate_all_correct(self):
        payload = {
            "user_answers": [
                {"question_index": 0, "selected_option": "A) Baker Street 221B"},
                {"question_index": 1, "selected_option": "B"},
            ],
            "quiz": [
                {
                    "question": "Sherlock Holmes nerede yaşar?",
                    "options": ["A) Baker Street 221B", "B) Oxford Street", "C) Fleet Street", "D) Abbey Road"],
                    "correct_answer": "A",
                    "explanation": "Baker Street 221B'de yaşar.",
                    "source_pages": [1],
                },
                {
                    "question": "Watson'ın mesleği nedir?",
                    "options": ["A) Avukat", "B) Doktor", "C) Mühendis", "D) Polis"],
                    "correct_answer": "B",
                    "explanation": "Dr. Watson bir doktordur.",
                    "source_pages": [2],
                },
            ],
        }
        resp = client.post("/api/v1/quiz/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_questions"] == 2
        assert data["correct_count"] == 2
        assert data["score_percentage"] == 100.0
        assert "🏆" in data["result_badge"]
        assert all(d["is_correct"] is True for d in data["details"])

    def test_evaluate_partial_score(self):
        payload = {
            "user_answers": [
                {"question_index": 0, "selected_option": "A"},
                {"question_index": 1, "selected_option": "C"},
            ],
            "quiz": [
                {
                    "question": "Soru 1?",
                    "options": ["A) Doğru", "B) Yanlış", "C) Yanlış", "D) Yanlış"],
                    "correct_answer": "A",
                    "explanation": "A doğru.",
                },
                {
                    "question": "Soru 2?",
                    "options": ["A) Yanlış", "B) Doğru", "C) Yanlış", "D) Yanlış"],
                    "correct_answer": "B",
                    "explanation": "B doğru.",
                },
            ],
        }
        resp = client.post("/api/v1/quiz/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["correct_count"] == 1
        assert data["score_percentage"] == 50.0
        assert data["details"][0]["is_correct"] is True
        assert data["details"][1]["is_correct"] is False

    def test_evaluate_unanswered_questions(self):
        payload = {
            "user_answers": [],
            "quiz": [
                {
                    "question": "Soru 1?",
                    "options": ["A) 1", "B) 2", "C) 3", "D) 4"],
                    "correct_answer": "A",
                    "explanation": "A",
                }
            ],
        }
        resp = client.post("/api/v1/quiz/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["correct_count"] == 0
        assert data["score_percentage"] == 0.0
        assert data["details"][0]["user_selected"] == "Cevaplanmadı"

    def test_evaluate_missing_payload(self):
        resp = client.post("/api/v1/quiz/evaluate", json={})
        assert resp.status_code == 422


class TestQuizValidationAndErrorHandling:
    def test_generate_from_book_invalid_chapter_index(self):
        # 1661'de 12 bölüm var, 999 istenirse 400 Bad Request dönmelidir
        payload = {
            "book_id": "1661",
            "title": "The Adventures of Sherlock Holmes",
            "author": "Arthur Conan Doyle",
            "num_questions": 3,
            "chapter_index": 999,
        }
        resp = client.post("/api/v1/quiz/generate-from-book", json=payload)
        assert resp.status_code == 400
        assert "Geçersiz bölüm numarası" in resp.json()["detail"]

    def test_generate_from_book_missing_required_fields(self):
        resp = client.post("/api/v1/quiz/generate-from-book", json={"num_questions": 5})
        assert resp.status_code == 422

    def test_generate_from_book_num_questions_out_of_bounds(self):
        # 0 soru geçersizdir (ge=1)
        resp1 = client.post(
            "/api/v1/quiz/generate-from-book",
            json={"book_id": "1661", "title": "Test", "num_questions": 0},
        )
        assert resp1.status_code == 422

        # 15 soru geçersizdir (le=10)
        resp2 = client.post(
            "/api/v1/quiz/generate-from-book",
            json={"book_id": "1661", "title": "Test", "num_questions": 15},
        )
        assert resp2.status_code == 422

    def test_generate_from_pdf_invalid_file_extension(self):
        files = {"file": ("test.txt", b"plain text content", "text/plain")}
        resp = client.post("/api/v1/quiz/generate-from-pdf", files=files)
        assert resp.status_code == 400
        assert "Yalnızca .pdf" in resp.json()["detail"]

    def test_generate_from_pdf_missing_file(self):
        resp = client.post("/api/v1/quiz/generate-from-pdf")
        assert resp.status_code == 422

    def test_generate_from_book_only_book_id_accepted(self):
        # Yalnızca book_id göndermek geçerli bir şemadır, 422 doğrulama hatası vermez
        from api.schemas import BookQuizRequest
        req = BookQuizRequest(book_id="1661")
        assert req.book_id == "1661"
        assert req.title is None
        assert req.author is None
        assert req.num_questions == 5


# ─── POST /api/v1/quizzes  (Yeni RESTful Endpoint) ───────────────────────────


class TestQuizzesRESTfulEndpoint:
    """POST /api/v1/quizzes — Yeni RESTful quiz kaynağı endpoint testleri."""

    # ── Şema & Validasyon ────────────────────────────────────────────────────

    def test_create_quiz_missing_source_returns_422(self):
        """source alanı olmadan istek 422 döndürmelidir."""
        resp = client.post("/api/v1/quizzes", json={"question_count": 3})
        assert resp.status_code == 422

    def test_create_quiz_missing_book_id_returns_422(self):
        """source içinde book_id zorunludur."""
        resp = client.post(
            "/api/v1/quizzes",
            json={"source": {"type": "book"}, "question_count": 3},
        )
        assert resp.status_code == 422

    def test_create_quiz_invalid_question_count_too_low(self):
        """question_count 0 geçersizdir (ge=1)."""
        resp = client.post(
            "/api/v1/quizzes",
            json={
                "source": {"type": "book", "book_id": "1661"},
                "question_count": 0,
            },
        )
        assert resp.status_code == 422

    def test_create_quiz_invalid_question_count_too_high(self):
        """question_count 11 geçersizdir (le=10)."""
        resp = client.post(
            "/api/v1/quizzes",
            json={
                "source": {"type": "book", "book_id": "1661"},
                "question_count": 11,
            },
        )
        assert resp.status_code == 422

    def test_create_quiz_invalid_difficulty_returns_422(self):
        """Geçersiz difficulty değeri 422 döndürmelidir."""
        resp = client.post(
            "/api/v1/quizzes",
            json={
                "source": {"type": "book", "book_id": "1661"},
                "difficulty": "extreme",
            },
        )
        assert resp.status_code == 422

    def test_create_quiz_invalid_chapter_index_returns_400(self):
        """Kitapta bulunmayan bölüm numarası 400 döndürmelidir."""
        resp = client.post(
            "/api/v1/quizzes",
            json={
                "source": {"type": "book", "book_id": "1661", "chapter_index": 999},
                "question_count": 3,
                "difficulty": "medium",
            },
        )
        assert resp.status_code == 400
        assert "Geçersiz bölüm numarası" in resp.json()["detail"]

    # ── Schema Modeli Testleri ────────────────────────────────────────────────

    def test_create_quiz_request_schema_defaults(self):
        """CreateQuizRequest varsayılan değerleri doğru olmalı."""
        from api.schemas import CreateQuizRequest, QuizSource
        req = CreateQuizRequest(source=QuizSource(type="book", book_id="1661"))
        assert req.question_count == 5
        assert req.difficulty == "medium"
        assert req.source.chapter_index is None

    def test_create_quiz_request_schema_with_chapter(self):
        """CreateQuizRequest chapter_index ile düzgün çalışmalı."""
        from api.schemas import CreateQuizRequest, QuizSource
        req = CreateQuizRequest(
            source=QuizSource(type="book", book_id="1661", chapter_index=2),
            question_count=3,
            difficulty="hard",
        )
        assert req.source.chapter_index == 2
        assert req.difficulty == "hard"

    def test_create_quiz_request_difficulty_enum(self):
        """difficulty alanı yalnızca easy/medium/hard kabul etmelidir."""
        from pydantic import ValidationError
        from api.schemas import CreateQuizRequest, QuizSource
        with pytest.raises(ValidationError):
            CreateQuizRequest(
                source=QuizSource(type="book", book_id="1661"),
                difficulty="zor",  # Türkçe değer API'de geçersiz
            )

