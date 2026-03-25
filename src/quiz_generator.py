import json
import re
import math
import random
import logging
import ollama as ollama_client

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL
from src.prompts import build_quiz_prompt, build_system_prompt, detect_language

logger = logging.getLogger(__name__)

MAX_CHARS_PER_CALL = 4000
MAX_CHUNKS_PER_CALL = 5

# Dil karisimi tespiti icin yasakli kelime listeleri
_ENGLISH_STOPWORDS = {
    "the", "and", "is", "was", "were", "are", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall",
    "not", "but", "or", "if", "then", "than", "that", "which",
    "who", "whom", "what", "where", "when", "how", "why",
    "some", "any", "all", "each", "every", "both", "few",
    "more", "most", "other", "into", "with", "from", "about",
    "between", "through", "during", "before", "after",
    "above", "below", "again", "further", "once",
    "gave", "kind", "only", "also", "just", "very", "really",
    "someone", "something", "unknown", "anything", "nothing",
}

_TURKISH_STOPWORDS = {
    "ve", "bir", "bu", "için", "ile", "olan", "olarak", "gibi",
    "daha", "ancak", "ama", "değil", "var", "çok", "sonra",
    "kadar", "bütün", "nasıl", "her", "bana", "onun", "benim",
}

_VAGUE_OPTIONS = {
    "unknown", "someone", "something", "someone else",
    "all kinds", "none", "all of the above", "none of the above",
    "bilinmiyor", "diğer", "hiçbiri", "hepsi",
}


def _extract_json(text: str) -> dict | None:
    """LLM çıktısından JSON bloğunu çıkarır. Markdown fence ve kirli karakterleri temizler."""
    cleaned = text.strip()

    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    brace_depth = 0
    start = None
    for i, c in enumerate(cleaned):
        if c == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                candidate = cleaned[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                    if "questions" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    start = None
                    continue

    return None


def _has_language_mixing(text: str, expected_lang: str) -> bool:
    """
    Metnin dil karisimi icerip icermedigini kontrol eder.
    Ornegin Turkce metinde Ingilizce stopword varsa -> karisik.
    """
    words = set(re.findall(r'[a-zA-ZçğıöşüÇĞİÖŞÜ]+', text.lower()))

    if expected_lang == "Turkish":
        english_hits = words & _ENGLISH_STOPWORDS
        turkish_hits = words & _TURKISH_STOPWORDS
        if len(english_hits) >= 3 and len(english_hits) > len(turkish_hits):
            return True
    elif expected_lang == "English":
        turkish_hits = words & _TURKISH_STOPWORDS
        if len(turkish_hits) >= 3:
            return True

    return False


def _has_vague_options(options: list[str]) -> bool:
    """Seceneklerin belirsiz/tembel olup olmadigini kontrol eder."""
    for opt in options:
        opt_clean = re.sub(r'^[A-D]\)\s*', '', opt).strip().lower()
        if opt_clean in _VAGUE_OPTIONS:
            return True
        if len(opt_clean) < 2:
            return True
    return False


def _validate_question(q: dict, expected_lang: str) -> bool:
    """Bir sorunun yapisal, dilsel ve kalite kontrolünü yapar."""
    required = ["question", "options", "correct_answer", "explanation"]
    if not all(key in q for key in required):
        logger.debug(f"Eksik alan: {q.get('question', '?')[:50]}")
        return False

    if not isinstance(q["options"], list) or len(q["options"]) != 4:
        logger.debug(f"Sik sayisi hatali: {q.get('question', '?')[:50]}")
        return False

    if q["correct_answer"] not in ("A", "B", "C", "D"):
        logger.debug(f"Gecersiz correct_answer: {q.get('correct_answer')}")
        return False

    # Soru cok kisa mi?
    if len(q["question"].strip()) < 10:
        logger.debug(f"Soru cok kisa: '{q['question']}'")
        return False

    # Dil karisimi kontrolu
    full_text = q["question"] + " " + " ".join(q["options"]) + " " + q["explanation"]
    if _has_language_mixing(full_text, expected_lang):
        logger.warning(f"Dil karisimi tespit edildi, soru atlanıyor: '{q['question'][:60]}'")
        return False

    # Belirsiz secenekler
    if _has_vague_options(q["options"]):
        logger.warning(f"Belirsiz secenek tespit edildi, soru atlanıyor: '{q['question'][:60]}'")
        return False

    return True


def _get_chunk_text(chunk) -> str:
    """Chunk'tan metin çıkarır. str veya dict olabilir."""
    if isinstance(chunk, dict):
        return chunk["text"]
    return chunk


def _get_chunk_pages(chunk) -> list[int]:
    """Chunk'tan sayfa bilgisi çıkarır."""
    if isinstance(chunk, dict):
        return chunk.get("source_pages", [])
    return []


def _collect_source_pages(chunks: list) -> list[int]:
    """Bir chunk grubundaki tüm kaynak sayfaları toplar ve sıralar."""
    pages = set()
    for chunk in chunks:
        pages.update(_get_chunk_pages(chunk))
    return sorted(pages)


def _build_context(chunks: list) -> str:
    """Chunk listesini max karakter sınırına göre birleştirir."""
    context_parts = []
    total_len = 0

    for chunk in chunks:
        text = _get_chunk_text(chunk)
        added_len = len(text) + (len("\n\n---\n\n") if context_parts else 0)
        if total_len + added_len > MAX_CHARS_PER_CALL:
            break
        context_parts.append(text)
        total_len += added_len

    return "\n\n---\n\n".join(context_parts)


def _select_chunk_groups(chunks: list, num_questions: int) -> list[list]:
    """
    Tüm chunk'ları kapsayacak şekilde gruplar oluşturur.
    Her grup bir LLM çağrısı için kullanılır.
    """
    if len(chunks) <= MAX_CHUNKS_PER_CALL:
        return [chunks]

    groups = []
    for i in range(0, len(chunks), MAX_CHUNKS_PER_CALL):
        group = chunks[i:i + MAX_CHUNKS_PER_CALL]
        groups.append(group)

    num_calls = math.ceil(num_questions / max(1, num_questions // len(groups) if len(groups) > 0 else 1))
    num_calls = min(num_calls, len(groups))

    if num_calls < len(groups):
        step = len(groups) / num_calls
        selected = [groups[int(i * step)] for i in range(num_calls)]
        return selected

    return groups


def _call_llm(context: str, count: int, language: str) -> list[dict]:
    """Tek bir LLM çağrısı yapıp geçerli soruları döner."""
    prompt = build_quiz_prompt(context=context, count=count, language=language)
    system = build_system_prompt(language)

    # Kalite filtresi bozuk soruları atacağı için fazladan soru iste
    request_count = min(count + 3, count * 2)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(
                f"LLM çağrısı deneme {attempt + 1}/{max_retries} "
                f"(model: {LLM_MODEL}, soru: {request_count}, dil: {language})"
            )

            response = ollama_client.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.4, "top_p": 0.9},
            )

            raw_text = response["message"]["content"]
            logger.debug(f"LLM ham çıktısı: {raw_text[:500]}...")

            parsed = _extract_json(raw_text)
            if parsed is None:
                logger.warning(f"Deneme {attempt + 1}: JSON parse edilemedi")
                continue

            questions = parsed.get("questions", [])
            valid = [q for q in questions if _validate_question(q, language)]

            if valid:
                logger.info(
                    f"  {len(valid)}/{len(questions)} soru kalite kontrolünden geçti"
                )
                return valid[:count]

            logger.warning(
                f"Deneme {attempt + 1}: {len(questions)} soru üretildi "
                f"ama hiçbiri kalite kontrolünden geçemedi"
            )

        except Exception as e:
            logger.error(f"Deneme {attempt + 1} hatası: {e}")

    return []


def generate_quiz(chunks: list, num_questions: int = 5) -> list[dict]:
    """
    Chunk'lardan çoktan seçmeli quiz soruları üretir.
    Tüm PDF'i kapsayacak şekilde chunk gruplarına böler ve
    her gruptan orantılı soru üretir.

    Args:
        chunks: Metin chunk'ları listesi (str veya dict with "text" and "source_pages")
        num_questions: Üretilecek toplam soru sayısı

    Returns:
        [{"question": "...", "options": [...], "correct_answer": "A",
          "explanation": "...", "source_pages": [3, 4]}, ...]

    Raises:
        RuntimeError: LLM'den hiç geçerli quiz alınamazsa
    """
    if not chunks:
        raise RuntimeError("Quiz üretilecek chunk bulunamadı")

    sample_texts = [_get_chunk_text(c) for c in chunks[:3]]
    full_sample = " ".join(sample_texts)
    detected_lang = detect_language(full_sample)
    logger.info(f"Algılanan dil: {detected_lang}")
    logger.info(f"Toplam chunk sayısı: {len(chunks)}, hedef soru: {num_questions}")

    groups = _select_chunk_groups(chunks, num_questions)
    logger.info(f"{len(groups)} chunk grubu oluşturuldu")

    questions_per_group = max(1, math.ceil(num_questions / len(groups)))
    all_questions = []

    for i, group in enumerate(groups):
        remaining = num_questions - len(all_questions)
        if remaining <= 0:
            break

        count = min(questions_per_group, remaining)
        context = _build_context(group)
        source_pages = _collect_source_pages(group)

        logger.info(
            f"Grup {i + 1}/{len(groups)}: {len(group)} chunk, "
            f"{len(context)} karakter, {count} soru hedefi, "
            f"sayfalar: {source_pages}"
        )

        questions = _call_llm(context, count, detected_lang)
        for q in questions:
            q["source_pages"] = source_pages
        all_questions.extend(questions)

    if not all_questions:
        raise RuntimeError(
            "Quiz üretimi başarısız: hiçbir gruptan geçerli soru üretilemedi. "
            "Daha güçlü bir model deneyin (ör: llama3.1:8b veya mistral)."
        )

    random.shuffle(all_questions)
    result = all_questions[:num_questions]

    logger.info(f"Toplam {len(result)} soru üretildi (hedef: {num_questions})")
    return result
