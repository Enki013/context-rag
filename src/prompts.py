SYSTEM_PROMPT = """You are a strict quiz generator. You ONLY output valid JSON. \
Never include markdown, code fences, or any text outside JSON. \
Every single word in your output must be in {language}. \
Never mix languages. Never use a word from another language."""

MULTIPLE_CHOICE_PROMPT = """Create {count} multiple-choice questions based on the text below.

LANGUAGE RULE (MOST IMPORTANT):
- Every word in questions, options, and explanations MUST be in {language}.
- Do NOT use any word from another language. Not even one word.
- If the source text has names (like character names, places), keep them as-is, but everything else must be {language}.

QUALITY RULES:
- Each question must have exactly 4 options labeled A), B), C), D)
- Exactly 1 option must be correct
- Options must be specific and meaningful — never use vague options like "unknown", "someone", "all of the above", "none"
- Questions must be clear and self-contained — a reader should understand what is being asked
- Explanations must be 1-2 sentences explaining WHY the answer is correct
- Base everything on the text — do not make up information

TEXT:
{context}

Respond ONLY with this JSON structure:
{{
  "questions": [
    {{
      "question": "Full clear question in {language}?",
      "options": ["A) Specific answer", "B) Specific answer", "C) Specific answer", "D) Specific answer"],
      "correct_answer": "A",
      "explanation": "Clear explanation in {language}."
    }}
  ]
}}"""


def detect_language(text: str) -> str:
    """
    Metnin dilini basit heuristik ile algılar.
    İlk 2000 karakterdeki karakter dağılımına ve yaygın kelimelere bakar.

    Args:
        text: Dili algılanacak metin

    Returns:
        Algılanan dil adı (İngilizce olarak, ör: "English", "Turkish", "German")
    """
    sample = text[:2000].lower()

    language_indicators = {
        "Turkish": {
            "chars": set("çğıöşüÇĞİÖŞÜ"),
            "words": ["ve", "bir", "bu", "için", "ile", "olan", "olarak", "gibi", "daha", "ancak",
                       "ama", "değil", "var", "çok", "sonra", "kadar", "bütün", "nasıl", "her", "bana"],
        },
        "English": {
            "chars": set(),
            "words": ["the", "and", "is", "in", "to", "of", "that", "for", "with", "this",
                       "was", "had", "but", "not", "from", "are", "were", "been", "have", "which"],
        },
        "German": {
            "chars": set("äöüßÄÖÜ"),
            "words": ["und", "der", "die", "das", "ist", "ein", "eine", "nicht", "sich", "mit"],
        },
        "French": {
            "chars": set("àâæçéèêëîïôœùûüÿ"),
            "words": ["le", "la", "les", "de", "des", "est", "un", "une", "dans", "pour"],
        },
        "Spanish": {
            "chars": set("áéíóúñ¿¡"),
            "words": ["el", "la", "de", "en", "los", "las", "del", "una", "por", "con"],
        },
        "Italian": {
            "chars": set("àèéìíîòóùú"),
            "words": ["il", "la", "di", "che", "non", "una", "per", "sono", "della", "anche"],
        },
        "Portuguese": {
            "chars": set("ãõáàâéêíóôúç"),
            "words": ["de", "que", "não", "uma", "para", "com", "por", "mais", "como", "seu"],
        },
        "Dutch": {
            "chars": set(),
            "words": ["de", "het", "een", "van", "en", "dat", "niet", "zijn", "maar", "voor"],
        },
        "Russian": {
            "chars": set("абвгдежзийклмнопрстуфхцчшщъыьэюя"),
            "words": ["и", "в", "не", "на", "что", "он", "как", "это", "было", "но"],
        },
        "Arabic": {
            "chars": set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي"),
            "words": ["في", "من", "على", "إلى", "أن", "هذا", "التي", "هو", "كان", "عن"],
        },
        "Chinese": {
            "chars": set(),
            "words": ["的", "是", "在", "了", "不", "和", "有", "这", "为", "我"],
        },
        "Japanese": {
            "chars": set("のにはをたがでてとしれさいうもなかっ"),
            "words": [],
        },
        "Korean": {
            "chars": set(),
            "words": ["의", "에", "를", "이", "는", "한", "하", "로", "와", "그"],
        },
    }

    scores = {}

    for lang, indicators in language_indicators.items():
        score = 0

        if indicators["chars"]:
            char_hits = sum(1 for c in sample if c in indicators["chars"])
            score += char_hits * 3

        words = sample.split()
        for word in indicators["words"]:
            count = words.count(word)
            score += count * 2

        scores[lang] = score

    if scores:
        best_lang = max(scores, key=scores.get)
        if scores[best_lang] > 0:
            return best_lang

    return "English"


def build_system_prompt(language: str) -> str:
    """Dil bilgisiyle system prompt oluşturur."""
    return SYSTEM_PROMPT.format(language=language)


def build_quiz_prompt(context: str, count: int = 5, language: str = None) -> str:
    """
    Quiz üretimi için prompt oluşturur.
    Dil otomatik algılanır veya manuel verilebilir.

    Args:
        context: LLM'e verilecek metin bağlamı
        count: Üretilecek soru sayısı
        language: Soru dili. None ise context'ten otomatik algılanır.

    Returns:
        Formatlanmış prompt string
    """
    if language is None:
        language = detect_language(context)

    return MULTIPLE_CHOICE_PROMPT.format(
        context=context,
        count=count,
        language=language,
    )
