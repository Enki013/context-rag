"""
api/schemas.py
--------------
Mobile ve web API istek ve yanıtları için Pydantic şemaları.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Health & Status ────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    ollama_connected: bool = Field(..., example=True)
    llm_model: str = Field(..., example="qwen2.5:14b")
    embedding_model: str = Field(..., example="bge-m3")
    catalog_installed: bool = Field(..., example=True)
    total_books: int = Field(..., example=79576)
    verify_answers: bool = Field(True, description="Grounding doğrulaması açık mı")
    verify_model: Optional[str] = Field(None, example="gemma4:31b")
    verify_model_installed: Optional[bool] = Field(
        None, description="Doğrulayıcı model Ollama'da kurulu mu (Ollama'ya ulaşılamazsa null)"
    )


# ─── Books & Search ─────────────────────────────────────────────────────────


class BookItem(BaseModel):
    book_id: str = Field(..., example="1661")
    title: str = Field(..., example="The Adventures of Sherlock Holmes")
    author: str = Field(..., example="Arthur Conan Doyle")
    subjects: str = Field("", example="Detective and mystery stories")
    bookshelves: str = Field("", example="Detective Fiction")
    language: str = Field("en", example="en")
    is_cached_locally: bool = Field(False, example=True)


class BookSearchResponse(BaseModel):
    query: str = Field(..., example="Sherlock Holmes")
    total: int = Field(..., example=5)
    books: List[BookItem]


class BookDetailResponse(BaseModel):
    book_id: str = Field(..., example="1661")
    title: str = Field(..., example="The Adventures of Sherlock Holmes")
    author: str = Field(..., example="Arthur Conan Doyle")
    subjects: str = Field("", example="Detective and mystery stories")
    bookshelves: str = Field("", example="Detective Fiction")
    language: str = Field("en", example="en")
    is_cached_locally: bool = Field(..., example=True)
    text_preview: Optional[str] = Field(None, example="To Sherlock Holmes she is always THE woman...")
    total_chapters: Optional[int] = Field(None, example=12, description="Kitaptaki toplam bölüm sayısı")


class ChapterItem(BaseModel):
    chapter_index: int = Field(..., example=1, description="1'den başlayan bölüm numarası")
    title: str = Field(..., example="I. A SCANDAL IN BOHEMIA", description="Bölüm başlığı")
    progress_percent: float = Field(..., example=0.1, description="Kitap içindeki başlangıç ilerleme yüzdesi (%0 - %100)")
    char_count: int = Field(..., example=46523, description="Bölümün karakter uzunluğu")


class BookChaptersResponse(BaseModel):
    book_id: str = Field(..., example="1661")
    title: str = Field(..., example="The Adventures of Sherlock Holmes")
    author: Optional[str] = Field(None, example="Arthur Conan Doyle")
    total_chapters: int = Field(..., example=12)
    chapters: List[ChapterItem]


class BookChapterCountResponse(BaseModel):
    book_id: str = Field(..., example="1661")
    title: str = Field(..., example="The Adventures of Sherlock Holmes")
    total_chapters: int = Field(..., example=12, description="Kitaptaki toplam bölüm sayısı")
    chapter_indices: List[int] = Field(..., example=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], description="Geçerli bölüm numaraları listesi")


# ─── Quiz Generation ────────────────────────────────────────────────────────


class QuizQuestion(BaseModel):
    question: str = Field(..., example="Sherlock Holmes hangi sokakta ikamet etmektedir?")
    options: List[str] = Field(
        ...,
        example=[
            "A) Baker Street 221B",
            "B) Oxford Street 12",
            "C) Abbey Road 4",
            "D) Fleet Street 10",
        ],
    )
    correct_answer: str = Field(..., example="A", description="Doğru şıkkın harfi (A, B, C veya D)")
    explanation: str = Field(..., example="Holmes ve Watson, Baker Street 221B adresinde ikamet etmektedir.")
    source_pages: List[int] = Field(default_factory=list, example=[1, 2])
    source_chapter: Optional[str] = Field(None, example="I. A SCANDAL IN BOHEMIA", description="Sorunun alındığı kitap bölümü")
    progress_percent: Optional[float] = Field(None, example=8.4, description="Kitap içindeki ilerleme yüzdesi")


class BookQuizRequest(BaseModel):
    book_id: str = Field(..., example="1661", description="Gutenberg kitap kimlik numarası (Kitabı tanımlamak için tek başına yeterlidir)")
    title: Optional[str] = Field(default=None, example="The Adventures of Sherlock Holmes", description="İsteğe bağlı: Boş bırakılırsa katalogdan otomatik doldurulur")
    author: Optional[str] = Field(default=None, example="Arthur Conan Doyle", description="İsteğe bağlı: Boş bırakılırsa katalogdan otomatik doldurulur")
    num_questions: int = Field(default=5, ge=1, le=10, description="Üretilecek soru sayısı (1-10)")
    difficulty: str = Field(default="orta", example="orta", description="kolay | orta | zor")
    question_type: str = Field(default="multiple_choice", example="multiple_choice")
    chapter_index: Optional[int] = Field(
        default=None,
        ge=1,
        example=1,
        description="İsteğe bağlı: Yalnızca belirli bir bölümden soru üretmek için bölüm numarası (1-based). Boş bırakılırsa tüm kitaptan dengeli seçilir.",
    )


class QuizResponse(BaseModel):
    source_type: str = Field(..., example="book", description="'book' veya 'pdf'")
    identifier: str = Field(..., example="1661", description="Kitap ID'si veya PDF dosya adı")
    title: str = Field(..., example="The Adventures of Sherlock Holmes")
    author: Optional[str] = Field(None, example="Arthur Conan Doyle")
    total_questions: int = Field(..., example=5)
    questions: List[QuizQuestion]


# ─── RESTful Quiz Resource (POST /api/v1/quizzes) ───────────────────────────


class QuizSource(BaseModel):
    type: Literal["book"] = Field(..., example="book", description="Quiz kaynağı türü: şimdilik yalnızca 'book'")
    book_id: str = Field(..., example="1661", description="Gutenberg kitap kimlik numarası")
    chapter_index: Optional[int] = Field(
        default=None,
        ge=1,
        example=1,
        description="İsteğe bağlı bölüm numarası (1-tabanlı). Boş bırakılırsa tüm kitaptan soru üretilir.",
    )


class CreateQuizRequest(BaseModel):
    source: QuizSource
    question_count: int = Field(default=5, ge=1, le=10, example=5, description="Üretilecek soru sayısı (1-10)")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        example="medium",
        description="Zorluk seviyesi: easy | medium | hard",
    )


class QuizResourceResponse(BaseModel):
    id: str = Field(..., example="quiz_8f31a2b3", description="Oluşturulan quiz için benzersiz kimlik")
    source_type: str = Field(..., example="book")
    book_id: str = Field(..., example="1661")
    chapter_index: Optional[int] = Field(None, example=1)
    title: str = Field(..., example="The Adventures of Sherlock Holmes")
    author: Optional[str] = Field(None, example="Arthur Conan Doyle")
    question_count: int = Field(..., example=5)
    difficulty: str = Field(..., example="medium", description="API difficulty değeri: easy | medium | hard")
    status: str = Field(default="completed", example="completed")
    questions: List[QuizQuestion]
    created_at: str = Field(..., example="2026-01-01T12:00:00Z", description="ISO 8601 formatında oluşturulma zamanı")




# ─── Quiz Evaluation (Cevap Kontrolü) ────────────────────────────────────────


class UserAnswerItem(BaseModel):
    question_index: int = Field(..., ge=0, example=0)
    selected_option: str = Field(..., example="A) Baker Street 221B", description="Seçilen şıkkın tamamı veya harfi")


class QuizEvaluationRequest(BaseModel):
    user_answers: List[UserAnswerItem]
    quiz: List[QuizQuestion]


class QuestionEvaluationResult(BaseModel):
    question_index: int = Field(..., example=0)
    question: str = Field(..., example="Sherlock Holmes nerede yaşar?")
    user_selected: str = Field(..., example="A) Baker Street 221B")
    correct_answer: str = Field(..., example="A")
    correct_option: str = Field(..., example="A) Baker Street 221B")
    is_correct: bool = Field(..., example=True)
    explanation: str = Field(..., example="Holmes ve Watson Baker Street 221B'de ikamet etmektedir.")


class QuizEvaluationResponse(BaseModel):
    total_questions: int = Field(..., example=5)
    correct_count: int = Field(..., example=4)
    score_percentage: float = Field(..., example=80.0)
    result_badge: str = Field(..., example="Harika! 🏆")
    details: List[QuestionEvaluationResult]


# ─── Mobile Headless Quiz Engine Schemas ───────────────────────────────────


class EngineChapterItem(BaseModel):
    chapter_index: int = Field(..., example=1)
    title: str = Field(..., example="Bölüm 1: Tanışma")
    start_page: Optional[int] = Field(None, example=5)
    end_page: Optional[int] = Field(None, example=24)
    progress_percent: float = Field(0.0, example=8.5)
    char_count: int = Field(0, example=12400)
    pool_count: int = Field(0, example=15, description="Bu bölüm için havuzda hazır bekleyen soru sayısı")


class BookImportResponse(BaseModel):
    book_id: str = Field(..., example="book_kurk_mantolu_madonna")
    title: str = Field(..., example="Kürk Mantolu Madonna")
    author: Optional[str] = Field(None, example="Sabahattin Ali")
    source_format: str = Field(..., example="epub")
    total_chapters: int = Field(..., example=12)
    total_pages: int = Field(0, example=160)
    chapters: List[EngineChapterItem]


class EngineServedQuestion(BaseModel):
    id: int = Field(..., example=101, description="Soru bankasındaki benzersiz soru ID'si")
    question: str = Field(..., example="Raif Efendi Maria Puder ile nerede karşılaştı?")
    options: List[str] = Field(..., example=["A) Resim galerisinde", "B) Parkta", "C) Kütüphanede", "D) Trende"])
    source_pages: Optional[List[int]] = Field(default=[], example=[12, 13])
    source_chapter: Optional[str] = Field(None, example="Bölüm 1")
    progress_percent: Optional[float] = Field(None, example=15.0)
    correct_answer: Optional[str] = Field(None, description="Yalnızca hide_answers=false ise döner")
    explanation: Optional[str] = Field(None, description="Yalnızca hide_answers=false ise döner")


class EngineServeQuizResponse(BaseModel):
    book_id: str = Field(..., example="book_kurk_mantolu_madonna")
    title: str = Field(..., example="Kürk Mantolu Madonna")
    author: Optional[str] = Field(None, example="Sabahattin Ali")
    chapter_index: int = Field(..., example=1)
    chapter_title: str = Field(..., example="Bölüm 1: Giriş")
    difficulty: str = Field("orta", example="orta")
    cache_hit: bool = Field(..., example=True, description="Sorular doğrudan havuzdan mı geldi (0ms) yoksa LLM ile yeni mi üretildi?")
    total_pool_size: int = Field(..., example=18, description="Bu bölüm için havuzdaki toplam soru adedi")
    question_count: int = Field(..., example=5)
    questions: List[EngineServedQuestion]


class BankAnswerItem(BaseModel):
    question_id: int = Field(..., example=101, description="Cevaplanan sorunun ID'si")
    selected_option: str = Field(..., example="A) Resim galerisinde", description="Seçilen şıkkın tamamı veya harfi (A)")


class BankEvaluationRequest(BaseModel):
    answers: List[BankAnswerItem]


class BankQuestionResult(BaseModel):
    question_id: int = Field(..., example=101)
    question_index: int = Field(..., example=0)
    question: str = Field(..., example="Raif Efendi Maria Puder ile nerede karşılaştı?")
    user_selected: str = Field(..., example="A) Resim galerisinde")
    correct_answer: str = Field(..., example="A")
    correct_option: str = Field(..., example="A) Resim galerisinde")
    is_correct: bool = Field(..., example=True)
    explanation: str = Field(..., example="Raif Efendi sergide Kürk Mantolu Madonna tablosunu incelerken karşılaşmıştır.")


class BankEvaluationResponse(BaseModel):
    total_questions: int = Field(..., example=5)
    correct_count: int = Field(..., example=4)
    score_percentage: float = Field(..., example=80.0)
    result_badge: str = Field(..., example="Harika! 🌟")
    details: List[BankQuestionResult]

