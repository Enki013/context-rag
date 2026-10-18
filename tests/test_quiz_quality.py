"""
tests/test_quiz_quality.py
--------------------------
Quiz kalitesi, künye filtreleme ve rastgele katmanlı örnekleme testleri.
"""

from src.quiz_generator import _is_boilerplate_chunk, _select_chunk_groups


def test_boilerplate_detection():
    # Künye / Telif chunk'ı
    boilerplate_1 = "Kronik Kitap Yayın No: 154, ISBN: 978-605-7635-12-3, Baskı: İstanbul 2021, Matbaa: Deniz Matbaacılık"
    assert _is_boilerplate_chunk(boilerplate_1) is True

    boilerplate_2 = "Table of contents:\nChapter 1... 5\nChapter 2... 25\nAll rights reserved."
    assert _is_boilerplate_chunk(boilerplate_2) is True

    # Gerçek hikaye / anlatı metni
    story_text = (
        "Prens Andrew odasına girdiğinde Pierre'i pencerenin kenarında otururken gördü. "
        "Savaşın getireceği yıkımı ve ordunun durumunu uzun uzun tartıştılar."
    )
    assert _is_boilerplate_chunk(story_text) is False


def test_stratified_random_sampling_filters_boilerplate():
    # İlk 5 chunk künye, sonraki 30 chunk gerçek metin
    chunks = [
        {"text": f"Kronik Kitap Yayın no {i} ISBN: 1234 Baskı: 1", "source_pages": [1]}
        for i in range(5)
    ] + [
        {"text": f"Gerçek kitap içeriği paragrafı {i} - olaylar gelişiyor ve kahraman ilerliyor.", "source_pages": [i + 2]}
        for i in range(30)
    ]

    # 3 soru için 3 grup seç
    groups = _select_chunk_groups(chunks, num_questions=3, randomize=True)
    assert len(groups) == 3

    # Seçilen hiçbir grupta künye olmamalı
    for group in groups:
        for chunk in group:
            assert _is_boilerplate_chunk(chunk["text"]) is False


def test_stratified_sampling_spreads_across_document():
    # 50 chunk'lık döküman
    chunks = [{"text": f"Paragraf {i}", "source_pages": [i]} for i in range(50)]

    # 5 soru için gruplar seç
    groups = _select_chunk_groups(chunks, num_questions=5, randomize=False)
    assert len(groups) == 5

    # Grupların başlangıç sayfaları dökümana yayılmış olmalı
    first_pages = [g[0]["source_pages"][0] for g in groups]
    assert first_pages == sorted(first_pages)
    assert first_pages[-1] > first_pages[0] + 30
