package com.contextrag.quizapp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.contextrag.quizapp.data.AnswerItem
import com.contextrag.quizapp.data.ApiClient
import com.contextrag.quizapp.data.BookChaptersResponse
import com.contextrag.quizapp.data.BookItem
import com.contextrag.quizapp.data.ChapterItem
import com.contextrag.quizapp.data.EvaluationResponse
import com.contextrag.quizapp.data.HealthResponse
import com.contextrag.quizapp.data.ServeQuizResponse
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

enum class AppScreen {
    LIBRARY,        // Kitap Arama & Katalog
    BOOK_DETAIL,    // Seçili Kitap, Bölümler & Quiz Ayarları
    QUIZ,           // Aktif Çoktan Seçmeli Test
    RESULT,         // Sınav Sonucu & Doğru/Yanlış Analizi
    SETTINGS        // Sunucu Bağlantısı & Sağlık Durumu
}

data class UiState(
    val currentScreen: AppScreen = AppScreen.LIBRARY,
    val baseUrl: String = "http://10.0.2.2:8000",
    val health: HealthResponse? = null,
    val isCheckingHealth: Boolean = false,

    // Search & Books
    val searchQuery: String = "Sherlock Holmes",
    val searchResults: List<BookItem> = emptyList(),
    val isSearching: Boolean = false,
    val selectedBook: BookItem? = null,

    // Chapters
    val chaptersData: BookChaptersResponse? = null,
    val selectedChapter: ChapterItem? = null,
    val isLoadingChapters: Boolean = false,

    // Quiz Serving
    val quizDifficulty: String = "orta",
    val quizCount: Int = 5,
    val currentQuiz: ServeQuizResponse? = null,
    val isLoadingQuiz: Boolean = false,
    val selectedAnswers: Map<Int, String> = emptyMap(), // question_id -> option
    val currentQuestionIndex: Int = 0,

    // Evaluation
    val isEvaluating: Boolean = false,
    val evaluationResult: EvaluationResponse? = null
)

class QuizViewModel : ViewModel() {

    private val _uiState = MutableStateFlow(UiState())
    val uiState = _uiState.asStateFlow()

    private val _snackbarEvent = MutableSharedFlow<String>()
    val snackbarEvent = _snackbarEvent.asSharedFlow()

    private var apiClient = ApiClient(_uiState.value.baseUrl)

    init {
        checkHealth()
        searchBooks(_uiState.value.searchQuery)
    }

    // ─── Navigasyon Yönetimi ──────────────────────────────────────────────────

    fun navigateTo(screen: AppScreen) {
        _uiState.value = _uiState.value.copy(currentScreen = screen)
    }

    fun navigateBack() {
        val curr = _uiState.value.currentScreen
        when (curr) {
            AppScreen.BOOK_DETAIL -> {
                _uiState.value = _uiState.value.copy(
                    currentScreen = AppScreen.LIBRARY,
                    selectedBook = null,
                    chaptersData = null,
                    selectedChapter = null
                )
            }
            AppScreen.QUIZ -> {
                // Sınavdan çıkıp bölümlere dön
                _uiState.value = _uiState.value.copy(
                    currentScreen = AppScreen.BOOK_DETAIL,
                    currentQuiz = null,
                    selectedAnswers = emptyMap()
                )
            }
            AppScreen.RESULT -> {
                // Sonuç ekranından bölümlere dön
                _uiState.value = _uiState.value.copy(
                    currentScreen = AppScreen.BOOK_DETAIL,
                    currentQuiz = null,
                    evaluationResult = null,
                    selectedAnswers = emptyMap()
                )
            }
            AppScreen.SETTINGS -> {
                _uiState.value = _uiState.value.copy(currentScreen = AppScreen.LIBRARY)
            }
            AppScreen.LIBRARY -> {
                // Kök ekranda bir şey yapma
            }
        }
    }

    fun returnToLibrary() {
        _uiState.value = _uiState.value.copy(
            currentScreen = AppScreen.LIBRARY,
            selectedBook = null,
            chaptersData = null,
            selectedChapter = null,
            currentQuiz = null,
            evaluationResult = null,
            selectedAnswers = emptyMap()
        )
    }

    fun returnToChapters() {
        _uiState.value = _uiState.value.copy(
            currentScreen = AppScreen.BOOK_DETAIL,
            currentQuiz = null,
            evaluationResult = null,
            selectedAnswers = emptyMap()
        )
    }

    // ─── API İşlemleri ────────────────────────────────────────────────────────

    fun updateBaseUrl(newUrl: String) {
        val trimmed = newUrl.trimEnd('/')
        _uiState.value = _uiState.value.copy(baseUrl = trimmed)
        apiClient.baseUrl = trimmed
        checkHealth()
    }

    fun checkHealth() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isCheckingHealth = true)
            apiClient.checkHealth().fold(
                onSuccess = { health ->
                    _uiState.value = _uiState.value.copy(
                        health = health,
                        isCheckingHealth = false
                    )
                },
                onFailure = { err ->
                    _uiState.value = _uiState.value.copy(
                        health = null,
                        isCheckingHealth = false
                    )
                    _snackbarEvent.emit("Sunucuya bağlanılamadı: ${err.localizedMessage}")
                }
            )
        }
    }

    fun onSearchQueryChanged(q: String) {
        _uiState.value = _uiState.value.copy(searchQuery = q)
    }

    fun searchBooks(query: String = _uiState.value.searchQuery) {
        if (query.isBlank()) return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isSearching = true)
            apiClient.searchBooks(query).fold(
                onSuccess = { res ->
                    _uiState.value = _uiState.value.copy(
                        searchResults = res.books,
                        isSearching = false
                    )
                },
                onFailure = { err ->
                    _uiState.value = _uiState.value.copy(isSearching = false)
                    _snackbarEvent.emit("Arama hatası: ${err.localizedMessage}")
                }
            )
        }
    }

    fun selectBook(book: BookItem) {
        _uiState.value = _uiState.value.copy(
            selectedBook = book,
            chaptersData = null,
            selectedChapter = null,
            currentQuiz = null,
            evaluationResult = null,
            selectedAnswers = emptyMap(),
            currentScreen = AppScreen.BOOK_DETAIL
        )
        fetchChapters(book.bookId)
    }

    fun refreshChapters() {
        val book = _uiState.value.selectedBook ?: return
        fetchChapters(book.bookId)
    }

    private fun fetchChapters(bookId: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoadingChapters = true)
            apiClient.getBookChapters(bookId).fold(
                onSuccess = { data ->
                    val firstChapter = data.chapters.firstOrNull()
                    _uiState.value = _uiState.value.copy(
                        chaptersData = data,
                        selectedChapter = firstChapter,
                        isLoadingChapters = false
                    )
                },
                onFailure = { err ->
                    _uiState.value = _uiState.value.copy(isLoadingChapters = false)
                    _snackbarEvent.emit("Bölümler alınamadı: ${err.localizedMessage}")
                }
            )
        }
    }

    fun selectChapter(chapter: ChapterItem) {
        _uiState.value = _uiState.value.copy(
            selectedChapter = chapter,
            currentQuiz = null,
            evaluationResult = null,
            selectedAnswers = emptyMap()
        )
    }

    fun setDifficulty(difficulty: String) {
        _uiState.value = _uiState.value.copy(quizDifficulty = difficulty)
    }

    fun setQuizCount(count: Int) {
        _uiState.value = _uiState.value.copy(quizCount = count)
    }

    fun startQuiz() {
        val book = _uiState.value.selectedBook ?: return
        val chapter = _uiState.value.selectedChapter ?: return
        val count = _uiState.value.quizCount
        val difficulty = _uiState.value.quizDifficulty

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                isLoadingQuiz = true,
                currentQuiz = null,
                evaluationResult = null,
                selectedAnswers = emptyMap(),
                currentQuestionIndex = 0
            )

            apiClient.serveQuiz(
                bookId = book.bookId,
                chapterIndex = chapter.chapterIndex,
                count = count,
                difficulty = difficulty,
                hideAnswers = true
            ).fold(
                onSuccess = { quizResp ->
                    _uiState.value = _uiState.value.copy(
                        currentQuiz = quizResp,
                        isLoadingQuiz = false,
                        currentScreen = AppScreen.QUIZ
                    )
                    if (quizResp.cacheHit) {
                        _snackbarEvent.emit("⚡ 0ms Soru Bankası Havuzundan Getirildi (Cache Hit)!")
                    } else {
                        _snackbarEvent.emit("🤖 LLM ile Yeni Sorular Üretildi ve Havuzlandı.")
                    }
                },
                onFailure = { err ->
                    _uiState.value = _uiState.value.copy(isLoadingQuiz = false)
                    _snackbarEvent.emit("Quiz yükleme hatası: ${err.localizedMessage}")
                }
            )
        }
    }

    fun onOptionSelected(questionId: Int, option: String) {
        val updated = _uiState.value.selectedAnswers.toMutableMap()
        updated[questionId] = option
        _uiState.value = _uiState.value.copy(selectedAnswers = updated)
    }

    fun nextQuestion() {
        val quiz = _uiState.value.currentQuiz ?: return
        if (_uiState.value.currentQuestionIndex < quiz.questions.size - 1) {
            _uiState.value = _uiState.value.copy(
                currentQuestionIndex = _uiState.value.currentQuestionIndex + 1
            )
        }
    }

    fun prevQuestion() {
        if (_uiState.value.currentQuestionIndex > 0) {
            _uiState.value = _uiState.value.copy(
                currentQuestionIndex = _uiState.value.currentQuestionIndex - 1
            )
        }
    }

    fun goToQuestion(idx: Int) {
        _uiState.value = _uiState.value.copy(currentQuestionIndex = idx)
    }

    fun submitQuiz() {
        val quiz = _uiState.value.currentQuiz ?: return
        val answers = quiz.questions.map { q ->
            val sel = _uiState.value.selectedAnswers[q.id] ?: ""
            AnswerItem(questionId = q.id, selectedOption = sel)
        }

        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isEvaluating = true)
            apiClient.evaluateAnswers(answers).fold(
                onSuccess = { res ->
                    _uiState.value = _uiState.value.copy(
                        evaluationResult = res,
                        isEvaluating = false,
                        currentScreen = AppScreen.RESULT
                    )
                },
                onFailure = { err ->
                    _uiState.value = _uiState.value.copy(isEvaluating = false)
                    _snackbarEvent.emit("Değerlendirme hatası: ${err.localizedMessage}")
                }
            )
        }
    }

    fun resetQuiz() {
        _uiState.value = _uiState.value.copy(
            currentQuiz = null,
            evaluationResult = null,
            selectedAnswers = emptyMap(),
            currentQuestionIndex = 0,
            currentScreen = AppScreen.BOOK_DETAIL
        )
    }
}
