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


@pytest.fixture(autouse=True)
def _no_epub_download(monkeypatch):
    """Testler Gutenberg'den EPUB indirmesin; yalnızca önbellekteki EPUB kullanılır."""
    import src.dataset_search as ds

    original = ds.download_book_epub
    monkeypatch.setattr(ds, "download_book_epub", lambda book_id, allow_download=True: original(book_id, allow_download=False))


@pytest.fixture(autouse=True)
def _no_book_download(monkeypatch):
    """Testler Gutenberg'den kitap metni indirmesin; yalnızca data/books önbelleği okunur."""
    import os

    import scripts.download_dataset as dd

    def cached_only(book_id, title="", author=""):
        path = os.path.join(dd.BOOKS_DIR, f"{book_id}.txt")
        if os.path.isfile(path) and os.path.getsize(path) > 100:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        raise FileNotFoundError(f"Kitap metni önbellekte yok (testlerde indirme kapalı): {book_id}")

    monkeypatch.setattr(dd, "download_book_text", cached_only)
