import json
import re
import math
import random
import logging
import numpy as np
import ollama as ollama_client

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEDUP_QUESTION_THRESHOLD, DEDUP_SIMILARITY_THRESHOLD, LLM_MODEL, VERIFY_ANSWERS, VERIFY_MODEL, VERIFY_THINK
from src.difficulty import normalize_difficulty
from src.embedder import _get_embeddings_batch
from src.prompts import PROMPT_VERSION, VERIFY_PROMPT, build_quiz_prompt, build_system_prompt, detect_language

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


_OPTION_PREFIX = re.compile(r'^\s*[A-Da-d]\s*[\)\.:]\s*')
_LETTERS = ("A", "B", "C", "D")


def _strip_option_prefix(opt: str) -> str:
    return _OPTION_PREFIX.sub('', str(opt)).strip()


def _has_duplicate_options(options: list[str]) -> bool:
    normalized = [_strip_option_prefix(o).lower() for o in options]
    return len(set(normalized)) != len(normalized)


def _shuffle_options(q: dict, rng: random.Random | None = None) -> dict:
    """
    Şıkları karıştırır ve A)-D) etiketlerini yeniden yazar.
    LLM doğru cevabı çoğunlukla A/B'ye koyuyor; bu konum yanlılığını kırar.
    """
    rng = rng or random
    texts = [_strip_option_prefix(o) for o in q["options"]]
    correct_text = texts[_LETTERS.index(q["correct_answer"])]

    order = list(range(len(texts)))
    rng.shuffle(order)
    shuffled = [texts[i] for i in order]

    q["options"] = [f"{_LETTERS[i]}) {t}" for i, t in enumerate(shuffled)]
    q["correct_answer"] = _LETTERS[shuffled.index(correct_text)]
    return q


def _validate_question(q: dict, expected_lang: str) -> bool:
    """Bir sorunun yapisal, dilsel ve kalite kontrolünü yapar."""
    required = ["question", "options", "correct_answer", "explanation"]
    if not all(key in q for key in required):
        logger.debug(f"Eksik alan: {q.get('question', '?')[:50]}")
        return False

    if not isinstance(q["options"], list) or len(q["options"]) != 4:
        logger.debug(f"Sik sayisi hatali: {q.get('question', '?')[:50]}")
        return False

    if q["correct_answer"] not in _LETTERS:
        logger.debug(f"Gecersiz correct_answer: {q.get('correct_answer')}")
        return False

    if _has_duplicate_options(q["options"]):
        logger.debug(f"Tekrarlanan sik: {q.get('question', '?')[:50]}")
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


def _strip_overlap(prev: str, curr: str, max_overlap: int = 400, min_overlap: int = 20) -> str:
    """Ardışık chunk'lar arasındaki splitter overlap'ini atar; aynı metin LLM'e iki kez gitmesin."""
    if not prev:
        return curr
    for k in range(min(len(prev), len(curr), max_overlap), min_overlap - 1, -1):
        if prev.endswith(curr[:k]):
            return curr[k:].lstrip()
    return curr


def _build_context(chunks: list) -> tuple[str, list]:
    """
    Chunk listesini max karakter sınırına göre birleştirir.
    Kaynak sayfa/indeks bilgisi doğru olsun diye LLM'e gerçekten giden chunk'ları da döner.
    """
    context_parts = []
    included = []
    total_len = 0

    prev_text = ""
    for chunk in chunks:
        raw = _get_chunk_text(chunk)
        text = _strip_overlap(prev_text, raw)
        prev_text = raw
        added_len = len(text) + (len("\n\n---\n\n") if context_parts else 0)
        if total_len + added_len > MAX_CHARS_PER_CALL and context_parts:
            break
        context_parts.append(text[:MAX_CHARS_PER_CALL])
        included.append(chunk)
        total_len += added_len

    return "\n\n---\n\n".join(context_parts), included


def _collect_chunk_indices(chunks: list) -> list[int]:
    return [c["index"] for c in chunks if isinstance(c, dict) and c.get("index") is not None]


_BOILERPLATE_PATTERNS = [
    r"yayın\s*no\b",
    r"isbn\b",
    r"sertifika\s*no\b",
    r"baskı\s*:\s*[0-9]+",
    r"matbaa\b",
    r"kapak\s*tasarımı\b",
    r"iç\s*düzen\b",
    r"mizanpaj\b",
    r"düzelti\b",
    r"all\s+rights\s+reserved\b",
    r"bütün\s+hakları\s+saklıdır\b",
    r"her\s+hakkı\s+saklıdır\b",
    r"printed\s+in\b",
    r"published\s+by\b",
    r"table\s+of\s+contents\b",
    r"içindekiler\b",
    r"kronik\s+kitap\b",
    r"yayıncılık\s+a\.ş\.",
]


def _is_boilerplate_chunk(text: str) -> bool:
    """Yayın evi künyesi, telif, ISBN veya içindekiler tablosu gibi içerik dışı chunk'ları tespit eder."""
    t_lower = text.lower()
    matches = sum(1 for pat in _BOILERPLATE_PATTERNS if re.search(pat, t_lower))
    if matches >= 2:
        return True
    if "isbn" in t_lower and any(w in t_lower for w in ("baskı", "yayın", "matbaa", "copyright")):
        return True
    return False


def _contiguous_runs(chunks: list) -> list[list]:
    """
    Chunk listesini metinde birbirini takip eden parçalara böler.
    İndeksi olan iki chunk arasında boşluk varsa (örn. kitabın farklı yerlerinden örneklenmişse)
    yeni bir parça başlar; böylece birbirinden kopuk metinler aynı LLM bağlamına yapıştırılmaz.
    """
    runs: list[list] = []
    prev_index = None
    for chunk in chunks:
        index = chunk.get("index") if isinstance(chunk, dict) else None
        adjacent = index is None or prev_index is None or index == prev_index + 1
        if not runs or not adjacent:
            runs.append([])
        runs[-1].append(chunk)
        prev_index = index
    return runs


def _select_chunk_groups(
    chunks: list,
    num_questions: int,
    randomize: bool = True,
) -> list[list]:
    """
    Tüm chunk'ları kapsayacak şekilde gruplar oluşturur.
    Her grup metinde ardışık chunk'lardan oluşur (en fazla MAX_CHUNKS_PER_CALL).
    Yayıncı künyesi ve telif gibi boilerplate chunk'ları eler.
    Rastgele katmanlı örnekleme (stratified random sampling) ile her çağrıda
    kitabın farklı bölümlerinden zengin ve taze soru üretilmesini sağlar.
    """
    # 1. Künye ve telif chunk'larını ayıkla (eğer yeterli chunk varsa)
    substantive_chunks = [c for c in chunks if not _is_boilerplate_chunk(_get_chunk_text(c))]
    if len(substantive_chunks) >= num_questions * 2:
        chunks_to_use = substantive_chunks
    else:
        chunks_to_use = chunks

    # 2. Ardışık parçaları MAX_CHUNKS_PER_CALL boyutunda gruplara böl
    groups = []
    for run in _contiguous_runs(chunks_to_use):
        for i in range(0, len(run), MAX_CHUNKS_PER_CALL):
            groups.append(run[i : i + MAX_CHUNKS_PER_CALL])
    if len(groups) <= 1:
        return groups

    num_calls = math.ceil(num_questions / max(1, num_questions // len(groups) if len(groups) > 0 else 1))
    num_calls = min(num_calls, len(groups))

    if num_calls >= len(groups):
        return groups

    # 3. Katmanlı seçim (Stratified Sampling)
    selected = []
    seg_size = len(groups) / num_calls

    for i in range(num_calls):
        start_idx = int(i * seg_size)
        end_idx = int((i + 1) * seg_size) - 1
        end_idx = max(start_idx, min(end_idx, len(groups) - 1))

        if randomize and end_idx > start_idx:
            chosen_idx = random.randint(start_idx, end_idx)
        else:
            chosen_idx = start_idx

        selected.append(groups[chosen_idx])

    return selected


def _call_llm(
    context: str,
    count: int,
    language: str,
    difficulty: str = "medium",
) -> list[dict]:
    """Tek bir LLM çağrısı yapıp geçerli soruları döner."""
    prompt = build_quiz_prompt(
        context=context,
        count=count,
        language=language,
        difficulty=difficulty,
    )
    system = build_system_prompt(language)

    # Kalite filtresi bozuk soruları atacağı için fazladan soru iste
    request_count = min(count + 3, count * 2)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(
                f"LLM çağrısı deneme {attempt + 1}/{max_retries} "
                f"(model: {LLM_MODEL}, soru: {request_count}, dil: {language}, zorluk: {difficulty})"
            )

            response = ollama_client.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.7, "top_p": 0.9},
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
                # Fazlası da dönülür; doğrulama bazılarını eleyebilir, kesme generate_quiz'de yapılır
                return valid

            logger.warning(
                f"Deneme {attempt + 1}: {len(questions)} soru üretildi "
                f"ama hiçbiri kalite kontrolünden geçemedi"
            )

        except Exception as e:
            logger.error(f"Deneme {attempt + 1} hatası: {e}")

    return []


_QUOTE_CHARS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-"})
MIN_EVIDENCE_CHARS = 15
MAX_GROUP_ATTEMPTS = 2


def _normalize_for_match(text: str) -> str:
    text = text.translate(_QUOTE_CHARS).lower()
    return re.sub(r"\s+", " ", text).strip()


def _evidence_in_context(evidence: str, context: str) -> bool:
    """Doğrulayıcının alıntısı metinde gerçekten geçiyor mu? ('...' ile bölünmüş alıntıları da kabul eder.)"""
    norm_ctx = _normalize_for_match(context)
    parts = [
        p.strip(" .,;:!?\"'")
        for p in re.split(r"\.{3}|…", _normalize_for_match(evidence))
    ]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < MIN_EVIDENCE_CHARS:
        return False
    return all(p in norm_ctx for p in parts)


def _parse_json_object(text: str) -> dict | None:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _verify_questions(questions: list[dict], context: str) -> list[dict]:
    """
    Grounding kontrolü: ikinci bir LLM çağrısı soruları yalnızca metne bakarak bağımsız cevaplar.
    Bir soru, doğrulayıcı aynı şıkkı seçer VE metinde birebir geçen bir kanıt alıntısı verirse kalır.
    Doğrulama yapılamazsa (LLM hatası/parse hatası) sorular güvenli tarafta kalmak için atılır.
    """
    if not questions:
        return []

    listing = "\n\n".join(
        f"{i}. {q['question']}\n" + "\n".join(q["options"])
        for i, q in enumerate(questions, 1)
    )
    prompt = VERIFY_PROMPT.format(context=context, questions=listing)

    chat_kwargs = {}
    if VERIFY_THINK is not None:
        chat_kwargs["think"] = VERIFY_THINK

    parsed = None
    for attempt in range(2):
        try:
            response = ollama_client.chat(
                model=VERIFY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.0},
                **chat_kwargs,
            )
            parsed = _parse_json_object(response["message"]["content"])
            if parsed and isinstance(parsed.get("answers"), list):
                break
            logger.warning(f"Doğrulama denemesi {attempt + 1}: JSON okunamadı")
        except Exception as e:
            logger.error(f"Doğrulama denemesi {attempt + 1} hatası: {e}")
        parsed = None

    if not parsed:
        logger.warning(f"Doğrulama yapılamadı, {len(questions)} soru atlandı")
        return []

    by_id = {}
    for item in parsed["answers"]:
        if isinstance(item, dict):
            try:
                by_id[int(item.get("id"))] = item
            except (TypeError, ValueError):
                continue

    verified = []
    for i, q in enumerate(questions, 1):
        item = by_id.get(i, {})
        answer = str(item.get("answer", "")).strip().upper()[:4]
        evidence = str(item.get("evidence", "") or "")

        if answer != q["correct_answer"]:
            logger.warning(
                f"Doğrulama reddetti (beklenen {q['correct_answer']}, doğrulayıcı {answer or '-'}): "
                f"'{q['question'][:70]}'"
            )
            continue
        if not _evidence_in_context(evidence, context):
            logger.warning(f"Kanıt metinde bulunamadı, soru atlandı: '{q['question'][:70]}'")
            continue

        q["evidence"] = evidence.strip()
        verified.append(q)

    logger.info(f"  {len(verified)}/{len(questions)} soru doğrulamadan geçti")
    return verified


def _dedup_text(q: dict) -> str:
    """Tekrar kontrolünde kullanılan metin: soru + doğru şıkkın metni (aynı soru kökü, farklı cevap ayrılsın)."""
    options = q.get("options") or []
    letter = q.get("correct_answer")
    answer = ""
    if letter in _LETTERS and len(options) == 4:
        answer = _strip_option_prefix(options[_LETTERS.index(letter)])
    return f"{q['question']} {answer}".strip()


def _deduplicate(candidates: list[dict], existing: list[dict] | None = None) -> list[dict]:
    """
    Tekrar eden soruları eler: önce birebir aynı soru metni, sonra bge-m3 embedding benzerliği.
    Soru+cevap benzerliği >= DEDUP_SIMILARITY_THRESHOLD (aynı soru, aynı cevap) veya yalnızca soru
    benzerliği >= DEDUP_QUESTION_THRESHOLD (aynı soru, çelişen cevap) ise tekrar sayılır. Hem aday listesinin kendi içinde hem de
    `existing` (örn. havuzdaki sorular) ile karşılaştırır; sıra korunur, ilk gelen kalır.
    Embedding alınamazsa yalnızca birebir kontrol yapılır (soru kaybetmemek için).
    """
    existing = existing or []
    seen = {_normalize_for_match(e["question"]) for e in existing}
    unique = []
    for q in candidates:
        key = _normalize_for_match(q["question"])
        if key in seen:
            logger.info(f"Birebir tekrar atlandı: '{q['question'][:70]}'")
            continue
        seen.add(key)
        unique.append(q)

    if not unique or (DEDUP_SIMILARITY_THRESHOLD >= 1.0 and DEDUP_QUESTION_THRESHOLD >= 1.0):
        return unique

    items = existing + unique
    try:
        texts = [_dedup_text(q) for q in items] + [q["question"] for q in items]
        vecs = np.asarray(_get_embeddings_batch(texts), dtype=float)
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    except Exception as e:
        logger.warning(f"Embedding alınamadı, yalnızca birebir tekrar kontrolü yapıldı: {e}")
        return unique
    qa_vecs, q_vecs = vecs[: len(items)], vecs[len(items):]

    kept_idx = list(range(len(existing)))
    kept = []
    for offset, q in enumerate(unique):
        i = len(existing) + offset
        if kept_idx:
            sim_qa = float(np.max(qa_vecs[kept_idx] @ qa_vecs[i]))
            sim_q = float(np.max(q_vecs[kept_idx] @ q_vecs[i]))
            if sim_qa >= DEDUP_SIMILARITY_THRESHOLD or sim_q >= DEDUP_QUESTION_THRESHOLD:
                logger.info(
                    f"Benzer soru atlandı (soru+cevap {sim_qa:.2f}, soru {sim_q:.2f}): '{q['question'][:70]}'"
                )
                continue
        kept_idx.append(i)
        kept.append(q)
    return kept


def generate_quiz(
    chunks: list,
    num_questions: int = 5,
    difficulty: str = "medium",
    question_type: str = "multiple_choice",
    randomize: bool = True,
    exclude_questions: list[dict] | None = None,
) -> list[dict]:
    """
    Chunk'lardan çoktan seçmeli quiz soruları üretir.
    Metni ardışık chunk gruplarına böler ve her gruptan orantılı soru üretir.

    Her tur üç aşamadan oluşur; aşamalar gruplar arasında toplu çalışır, böylece Ollama
    üretici ve doğrulayıcı modeller arasında grup başına değil, tur başına bir kez geçiş yapar:
      1. Üretim: eksik soru kalan her grup için üretici LLM çağrısı + şık karıştırma
      2. Doğrulama: her grubun adayları kendi metnine karşı doğrulanır (VERIFY_ANSWERS)
      3. Tekrar eleme: tüm adaylar birbirine, kabul edilmişlere ve exclude_questions'a karşı
    Doğrulama açıksa eksik kalan gruplar için bir tur daha yapılır.

    Args:
        chunks: Metin chunk'ları listesi (str veya dict with "text" and "source_pages")
        num_questions: Üretilecek toplam soru sayısı
        difficulty: Zorluk derecesi ("easy", "medium", "hard"; eski "kolay/orta/zor" da kabul edilir)
        question_type: Soru türü ("multiple_choice")
        randomize: True ise her seferinde farklı chunk gruplarından rastgele örnekleme yapar
        exclude_questions: Tekrarı istenmeyen mevcut sorular (örn. havuzdakiler)

    Returns:
        [{"question": "...", "options": [...], "correct_answer": "A",
          "explanation": "...", "source_pages": [3, 4]}, ...]

    Raises:
        RuntimeError: LLM'den hiç geçerli quiz alınamazsa
    """
    if not chunks:
        raise RuntimeError("Quiz üretilecek chunk bulunamadı")
    difficulty = normalize_difficulty(difficulty)

    sample_texts = [_get_chunk_text(c) for c in chunks[:3]]
    detected_lang = detect_language(" ".join(sample_texts))
    logger.info(f"Algılanan dil: {detected_lang}, zorluk: {difficulty}")
    logger.info(f"Toplam chunk sayısı: {len(chunks)}, hedef soru: {num_questions}")

    groups = _select_chunk_groups(chunks, num_questions, randomize=randomize)
    logger.info(f"{len(groups)} chunk grubu oluşturuldu")

    questions_per_group = max(1, math.ceil(num_questions / len(groups)))
    plan = []
    remaining = num_questions
    for i, group in enumerate(groups):
        if remaining <= 0:
            break
        target = min(questions_per_group, remaining)
        remaining -= target
        context, included = _build_context(group)
        plan.append({"context": context, "included": included, "target": target,
                     "accepted": [], "candidates": []})
        logger.info(
            f"Grup {i + 1}/{len(groups)}: {len(included)} chunk, {len(context)} karakter, "
            f"{target} soru hedefi, sayfalar: {_collect_source_pages(included)}"
        )

    exclude = list(exclude_questions or [])
    rounds = MAX_GROUP_ATTEMPTS if VERIFY_ANSWERS else 1
    for round_no in range(1, rounds + 1):
        pending = [p for p in plan if len(p["accepted"]) < p["target"]]
        if not pending:
            break
        logger.info(f"Tur {round_no}/{rounds}: {len(pending)} grup")

        # Aşama 1: üretim (üretici model)
        for p in pending:
            need = p["target"] - len(p["accepted"])
            p["candidates"] = [
                _shuffle_options(q)
                for q in _call_llm(p["context"], need, detected_lang, difficulty=difficulty)
            ]

        # Aşama 2: doğrulama (doğrulayıcı model)
        if VERIFY_ANSWERS:
            for p in pending:
                p["candidates"] = _verify_questions(p["candidates"], p["context"])

        # Aşama 3: tekrar eleme (tek embedding çağrısı)
        already = exclude + [q for p in plan for q in p["accepted"]]
        unique_ids = {id(q) for q in _deduplicate([q for p in pending for q in p["candidates"]], already)}
        for p in pending:
            need = p["target"] - len(p["accepted"])
            p["accepted"].extend([q for q in p["candidates"] if id(q) in unique_ids][:need])
            p["candidates"] = []

    all_questions = []
    for p in plan:
        included = p["included"]
        source_pages = _collect_source_pages(included)
        chunk_indices = _collect_chunk_indices(included)
        source_chapter = next((c.get("source_chapter") for c in included if isinstance(c, dict) and c.get("source_chapter")), None)
        progress_percent = next((c.get("progress_percent") for c in included if isinstance(c, dict) and c.get("progress_percent") is not None), None)
        for q in p["accepted"]:
            q["source_pages"] = source_pages
            q["source_chunk_indices"] = chunk_indices
            q["model"] = LLM_MODEL
            q["prompt_version"] = PROMPT_VERSION
            if source_chapter:
                q["source_chapter"] = source_chapter
            if progress_percent is not None:
                q["progress_percent"] = progress_percent
            all_questions.append(q)

    if not all_questions:
        raise RuntimeError(
            "Quiz üretimi başarısız: hiçbir gruptan geçerli soru üretilemedi. "
            "Daha güçlü bir model deneyin (ör: llama3.1:8b veya mistral)."
        )

    random.shuffle(all_questions)
    result = all_questions[:num_questions]

    logger.info(f"Toplam {len(result)} soru üretildi (hedef: {num_questions})")
    return result

