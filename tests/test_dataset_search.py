"""
tests/test_dataset_search.py
----------------------------
Yerel kitap arama, lisans temizleme, bölüm (chapter) çıkarma
ve chunking işlevleri için birim testler.
"""

import os
import pytest
from src.dataset_search import (
    clean_gutenberg_text,
    extract_chapters,
    _clean_fts_query,
    search_books,
    get_book_chapters,
    get_chunks_for_book,
)


class TestTextCleaning:
    def test_clean_gutenberg_header_and_footer(self):
        sample = (
            "Random Gutenberg preamble line\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "This is the actual story text.\n"
            "Chapter 1: The Beginning.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            "Gutenberg legal footer\n"
        )
        cleaned = clean_gutenberg_text(sample)
        assert "This is the actual story text." in cleaned
        assert "Chapter 1: The Beginning." in cleaned
        assert "*** START OF THE" not in cleaned
        assert "*** END OF THE" not in cleaned
        assert "Gutenberg legal footer" not in cleaned

    def test_clean_text_without_markers(self):
        plain = "Just normal story text without any headers."
        cleaned = clean_gutenberg_text(plain)
        assert cleaned == plain

    def test_clean_empty_text(self):
        assert clean_gutenberg_text("") == ""
        assert clean_gutenberg_text("   \n\t  ") == ""


class TestChapterExtraction:
    def test_extract_chapters_from_standard_headings(self):
        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            "CHAPTER I.\n" + ("Story text line in chapter one. " * 50) + "\n\n"
            "CHAPTER II.\n" + ("Story text line in chapter two. " * 50) + "\n\n"
            "CHAPTER III.\n" + ("Story text line in chapter three. " * 50) + "\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***"
        )
        chapters = extract_chapters(text)
        assert len(chapters) == 3
        assert chapters[0]["chapter_index"] == 1
        assert "CHAPTER I" in chapters[0]["title"]
        assert chapters[0]["progress_percent"] == 0.0
        assert chapters[1]["chapter_index"] == 2
        assert "CHAPTER II" in chapters[1]["title"]
        assert chapters[2]["chapter_index"] == 3
        assert "CHAPTER III" in chapters[2]["title"]

    def test_extract_chapters_filters_table_of_contents(self):
        toc_and_body = (
            "*** START OF THE PROJECT GUTENBERG EBOOK ***\n"
            "CONTENTS\n"
            "CHAPTER I. Introduction\n"
            "CHAPTER II. The Journey\n"
            "CHAPTER III. The Climax\n"
            "\n" + ("Introductory preface text... " * 30) + "\n\n"
            "CHAPTER I. Introduction\n" + ("Lots of body text for chapter 1. " * 50) + "\n\n"
            "CHAPTER II. The Journey\n" + ("Lots of body text for chapter 2. " * 50) + "\n\n"
            "CHAPTER III. The Climax\n" + ("Lots of body text for chapter 3. " * 50) + "\n"
        )
        chapters = extract_chapters(toc_and_body)
        assert len(chapters) == 3
        assert chapters[0]["chapter_index"] == 1
        assert chapters[0]["char_count"] > 500

    def test_extract_chapters_fallback_on_unstructured_text(self):
        plain_essay = "This is a long essay without any explicit chapter markers. " * 80
        chapters = extract_chapters(plain_essay)
        assert len(chapters) == 4
        assert chapters[0]["chapter_index"] == 1
        assert chapters[0]["progress_percent"] == 0.0
        assert chapters[1]["progress_percent"] == 25.0
        assert chapters[2]["progress_percent"] == 50.0
        assert chapters[3]["progress_percent"] == 75.0

    def test_extract_chapters_empty_text(self):
        assert extract_chapters("") == []

    def test_extract_chapters_word_numbers(self):
        text = "\n\nChapter One\n" + ("Story text line in chapter one. " * 30) + "\n\nChapter Two\n" + ("Story text line in chapter two. " * 30)
        chapters = extract_chapters(text)
        assert len(chapters) == 2
        assert "Chapter One" in chapters[0]["title"]
        assert "Chapter Two" in chapters[1]["title"]

    def test_extract_chapters_multilingual(self):
        # Fransızca
        fr = "\n\nChapitre I\n" + ("Texte français pour le premier chapitre. " * 30) + "\n\nChapitre II\n" + ("Texte pour le deuxième chapitre. " * 30)
        ch_fr = extract_chapters(fr)
        assert len(ch_fr) == 2
        assert "Chapitre I" in ch_fr[0]["title"]

        # İspanyolca
        es = "\n\nCapítulo 1\n" + ("Texto en español para el primer capítulo. " * 30) + "\n\nCapítulo 2\n" + ("Texto para el segundo capítulo. " * 30)
        ch_es = extract_chapters(es)
        assert len(ch_es) == 2
        assert "Capítulo 1" in ch_es[0]["title"]

    def test_extract_chapters_cached_frankenstein(self):
        import os
        path = "data/books/84.txt"
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                chaps = extract_chapters(f.read())
            assert len(chaps) == 28
            assert "Letter 1" in chaps[0]["title"]
            assert "Chapter 24" in chaps[-1]["title"]

    def test_extract_chapters_cached_alice_play(self):
        import os
        path = "data/books/35688.txt"
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                chaps = extract_chapters(f.read())
            assert len(chaps) == 3
            assert "ACT I" in chaps[0]["title"]
            assert "ACT III" in chaps[-1]["title"]



class TestFtsQueryCleaning:
    def test_clean_fts_query_simple(self):
        q = _clean_fts_query("Sherlock Holmes")
        assert '"Sherlock"*' in q
        assert '"Holmes"*' in q

    def test_clean_fts_query_special_chars(self):
        q = _clean_fts_query("Sherlock! & (Holmes) OR *")
        assert '"Sherlock"*' in q
        assert '"Holmes"*' in q
        assert "&" not in q

    def test_clean_fts_query_empty(self):
        assert _clean_fts_query("") == ""
        assert _clean_fts_query("   !@#$%^&*()   ") == ""


class TestBookSearch:
    def test_search_existing_book(self):
        results = search_books("Sherlock", limit=5)
        assert isinstance(results, list)
        assert len(results) > 0
        book = results[0]
        assert "book_id" in book
        assert "title" in book
        assert "author" in book
        assert "Sherlock" in book["title"] or "Sherlock" in book["author"] or "Sherlock" in book.get("subjects", "")

    def test_search_empty_query(self):
        assert search_books("") == []
        assert search_books("   ") == []

    def test_search_non_existent_query(self):
        results = search_books("xyznonexistentbooktitle99999abc", limit=5)
        assert results == []

    def test_search_limit_respected(self):
        results = search_books("the", limit=3)
        assert len(results) <= 3


class TestBookChaptersAndChunks:
    def test_get_book_chapters_for_cached_book(self):
        data = get_book_chapters("1661")
        assert data["book_id"] == "1661"
        assert "The Adventures of Sherlock Holmes" in data["title"]
        assert data["total_chapters"] == 12
        assert len(data["chapters"]) == 12
        first_chap = data["chapters"][0]
        assert first_chap["chapter_index"] == 1
        assert "SCANDAL IN BOHEMIA" in first_chap["title"]

    def test_get_chunks_for_book_whole_book(self):
        chunks = get_chunks_for_book(
            book_id="1661",
            title="The Adventures of Sherlock Holmes",
            author="Arthur Conan Doyle",
            max_chunks=4,
        )
        assert len(chunks) <= 4
        assert len(chunks) > 0
        first = chunks[0]
        assert "text" in first
        assert "embedding" in first
        assert len(first["embedding"]) > 0
        assert "source_chapter" in first
        assert first["progress_percent"] is not None

    def test_get_chunks_for_book_single_chapter(self):
        chunks = get_chunks_for_book(
            book_id="1661",
            title="The Adventures of Sherlock Holmes",
            author="Arthur Conan Doyle",
            max_chunks=3,
            chapter_index=2,
        )
        assert len(chunks) > 0
        for c in chunks:
            assert "RED-HEADED LEAGUE" in c["source_chapter"]

    def test_get_chunks_for_book_invalid_chapter_raises_error(self):
        with pytest.raises(ValueError, match="Geçersiz bölüm numarası"):
            get_chunks_for_book(
                book_id="1661",
                title="The Adventures of Sherlock Holmes",
                author="Arthur Conan Doyle",
                chapter_index=999,
            )

        with pytest.raises(ValueError, match="Geçersiz bölüm numarası"):
            get_chunks_for_book(
                book_id="1661",
                title="The Adventures of Sherlock Holmes",
                author="Arthur Conan Doyle",
                chapter_index=0,
            )
