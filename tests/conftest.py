"""
Ortak test ayarları: hiçbir test gerçek data/quiz_engine.db dosyasına dokunmasın.
db_path vermeden soru bankasını kullanan kod (örn. API uçları) her test için geçici bir veritabanına yönlenir.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_question_bank(tmp_path, monkeypatch):
    monkeypatch.setattr("src.question_bank.QUIZ_ENGINE_DB_PATH", str(tmp_path / "quiz_engine_test.db"))
