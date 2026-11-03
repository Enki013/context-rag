"""
Ortak test ayarları: hiçbir test gerçek data/quiz_engine.db dosyasına dokunmasın.
db_path vermeden soru bankasını kullanan kod (örn. API uçları) her test için geçici bir veritabanına yönlenir.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_question_bank(tmp_path, monkeypatch):
    monkeypatch.setattr("src.question_bank.QUIZ_ENGINE_DB_PATH", str(tmp_path / "quiz_engine_test.db"))


@pytest.fixture(autouse=True)
def _no_semantic_dedup(monkeypatch):
    """Testler Ollama embedding çağırmasın; semantik tekrar eleme testleri bunu kendisi açar."""
    monkeypatch.setattr("src.quiz_generator.DEDUP_SIMILARITY_THRESHOLD", 1.0)
    monkeypatch.setattr("src.quiz_generator.DEDUP_QUESTION_THRESHOLD", 1.0)
