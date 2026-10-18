package com.contextrag.quizapp.data

import com.google.gson.annotations.SerializedName

data class HealthResponse(
    val status: String,
    @SerializedName("ollama_connected") val ollamaConnected: Boolean,
    @SerializedName("llm_model") val llmModel: String,
    @SerializedName("embedding_model") val embeddingModel: String,
    @SerializedName("catalog_installed") val catalogInstalled: Boolean,
    @SerializedName("total_books") val totalBooks: Int
)

data class BookItem(
    @SerializedName("book_id") val bookId: String,
    val title: String,
    val author: String? = null,
    val subjects: String? = null,
    val bookshelves: String? = null,
    val language: String? = "en",
    @SerializedName("is_cached_locally") val isCachedLocally: Boolean = false
)

data class BookSearchResponse(
    val query: String,
    val total: Int,
    val books: List<BookItem> = emptyList()
)

data class ChapterItem(
    @SerializedName("chapter_index") val chapterIndex: Int,
    val title: String,
    @SerializedName("start_page") val startPage: Int? = null,
    @SerializedName("end_page") val endPage: Int? = null,
    @SerializedName("progress_percent") val progressPercent: Float = 0f,
    @SerializedName("char_count") val charCount: Int = 0,
    @SerializedName("pool_count") val poolCount: Int = 0
)

data class BookChaptersResponse(
    @SerializedName("book_id") val bookId: String,
    val title: String,
    val author: String? = null,
    @SerializedName("total_chapters") val totalChapters: Int = 0,
    val chapters: List<ChapterItem> = emptyList()
)

data class ServedQuestion(
    val id: Int,
    val question: String,
    val options: List<String> = emptyList(),
    @SerializedName("source_pages") val sourcePages: List<Int> = emptyList(),
    @SerializedName("source_chapter") val sourceChapter: String? = null,
    @SerializedName("progress_percent") val progressPercent: Float? = null,
    @SerializedName("correct_answer") val correctAnswer: String? = null,
    val explanation: String? = null
)

data class ServeQuizResponse(
    @SerializedName("book_id") val bookId: String,
    val title: String,
    val author: String? = null,
    @SerializedName("chapter_index") val chapterIndex: Int,
    @SerializedName("chapter_title") val chapterTitle: String,
    val difficulty: String = "orta",
    @SerializedName("cache_hit") val cacheHit: Boolean = false,
    @SerializedName("total_pool_size") val totalPoolSize: Int = 0,
    @SerializedName("question_count") val questionCount: Int = 0,
    val questions: List<ServedQuestion> = emptyList()
)

data class AnswerItem(
    @SerializedName("question_id") val questionId: Int,
    @SerializedName("selected_option") val selectedOption: String
)

data class EvaluationRequest(
    val answers: List<AnswerItem>
)

data class QuestionEvaluationDetail(
    @SerializedName("question_id") val questionId: Int,
    @SerializedName("question_index") val questionIndex: Int,
    val question: String,
    @SerializedName("user_selected") val userSelected: String,
    @SerializedName("correct_answer") val correctAnswer: String,
    @SerializedName("correct_option") val correctOption: String,
    @SerializedName("is_correct") val isCorrect: Boolean,
    val explanation: String
)

data class EvaluationResponse(
    @SerializedName("total_questions") val totalQuestions: Int,
    @SerializedName("correct_count") val correctCount: Int,
    @SerializedName("score_percentage") val scorePercentage: Float,
    @SerializedName("result_badge") val resultBadge: String,
    val details: List<QuestionEvaluationDetail> = emptyList()
)
