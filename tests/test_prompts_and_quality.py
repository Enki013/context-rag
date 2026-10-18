"""
tests/test_prompts_and_quality.py
---------------------------------
Prompt üretimi, dil algılama, JSON parse etme ve
quiz soru kalite doğrulama mantığı için birim testler.
"""

import pytest
from src.prompts import detect_language, build_quiz_prompt, build_system_prompt
from src.quiz_generator import _validate_question, _extract_json, _has_vague_options


class TestPromptGeneration:
    def test_detect_language_turkish(self):
        text = "Bu kitap Sherlock Holmes'un maceralarını anlatmaktadır. Çok güzel bir hikaye."
        assert detect_language(text) == "Turkish"

    def test_detect_language_english(self):
        text = "To Sherlock Holmes she is always the woman. I have seldom heard him mention her."
        assert detect_language(text) == "English"

    def test_build_quiz_prompt_contains_parameters(self):
        prompt = build_quiz_prompt(
            context="Sherlock Holmes was a detective.",
            count=5,
            language="Turkish",
            difficulty="zor",
        )
        assert "Sherlock Holmes was a detective." in prompt
        assert "5 multiple-choice" in prompt
        assert "hard" in prompt.lower()
        assert "Turkish" in prompt

    def test_build_system_prompt_json_requirement(self):
        sys_prompt = build_system_prompt("Turkish")
        assert "JSON" in sys_prompt
        assert "Turkish" in sys_prompt


class TestQuestionValidationAndQuality:
    def test_validate_question_success(self):
        valid_q = {
            "question": "Sherlock Holmes nerede yaşamaktadır?",
            "options": [
                "A) Baker Street 221B",
                "B) Oxford Caddesi 10",
                "C) Fleet Sokağı 5",
                "D) Abbey Yolu 4",
            ],
            "correct_answer": "A",
            "explanation": "Holmes, Baker Street 221B adresinde ikamet etmektedir.",
        }
        assert _validate_question(valid_q, "Turkish") is True

    def test_validate_question_missing_field(self):
        invalid_q = {
            "question": "Eksik soru?",
            "options": ["A) 1", "B) 2", "C) 3", "D) 4"],
            # correct_answer missing
            "explanation": "Açıklama",
        }
        assert _validate_question(invalid_q, "Turkish") is False

    def test_validate_question_invalid_option_count(self):
        invalid_q = {
            "question": "Soru?",
            "options": ["A) 1", "B) 2", "C) 3"],  # Sadece 3 şık var
            "correct_answer": "A",
            "explanation": "Açıklama",
        }
        assert _validate_question(invalid_q, "Turkish") is False

    def test_validate_question_duplicate_options(self):
        invalid_q = {
            "question": "Soru?",
            "options": ["A) Aynı", "B) Aynı", "C) Farklı", "D) Başka"],
            "correct_answer": "A",
            "explanation": "Açıklama",
        }
        assert _validate_question(invalid_q, "Turkish") is False

    def test_validate_question_vague_options(self):
        vague_q = {
            "question": "Soru?",
            "options": ["A) Doğru", "B) Yanlış", "C) Bilinmiyor", "D) Hiçbiri"],
            "correct_answer": "A",
            "explanation": "Açıklama",
        }
        assert _validate_question(vague_q, "Turkish") is False
        assert _has_vague_options(vague_q["options"]) is True

    def test_validate_question_mixed_language_rejection(self):
        # Türkçe olması gerekirken İngilizce stopword içeren soru
        mixed_q = {
            "question": "Where is the place where Sherlock Holmes was living with Watson?",
            "options": ["A) Baker", "B) Oxford", "C) Fleet", "D) Abbey"],
            "correct_answer": "A",
            "explanation": "Because he lived there with them and was always there.",
        }
        # Hedef dil Turkish ise bu soru dil filtresine takılmalıdır
        assert _validate_question(mixed_q, "Turkish") is False


class TestJsonExtraction:
    def test_extract_pure_json(self):
        raw = '{"questions": [{"question": "Test?"}]}'
        data = _extract_json(raw)
        assert data is not None
        assert "questions" in data

    def test_extract_markdown_fence_json(self):
        raw = 'Here is the response:\n```json\n{"questions": [{"question": "Fence test"}]}\n```\nHope it helps!'
        data = _extract_json(raw)
        assert data is not None
        assert data["questions"][0]["question"] == "Fence test"

    def test_extract_malformed_returns_none(self):
        raw = "This is not json at all {broken"
        data = _extract_json(raw)
        assert data is None
