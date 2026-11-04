package com.contextrag.quizapp.viewmodel

import com.contextrag.quizapp.data.ApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class QuizViewModelTest {

    private val testDispatcher = StandardTestDispatcher()
    private lateinit var mockWebServer: MockWebServer
    private lateinit var apiClient: ApiClient

    private val healthJson = """
        {
            "status": "ok",
            "ollama_connected": true,
            "llm_model": "qwen2.5:14b",
            "embedding_model": "bge-m3",
            "catalog_installed": true,
            "total_books": 79576
        }
    """.trimIndent()

    private val searchJson = """
        {
            "query": "Sherlock Holmes",
            "total": 1,
            "books": [
                {
                    "book_id": "1661",
                    "title": "The Adventures of Sherlock Holmes",
                    "author": "Arthur Conan Doyle",
                    "subjects": "Detective stories",
                    "bookshelves": "Detective Fiction",
                    "language": "en",
                    "is_cached_locally": true
                }
            ]
        }
    """.trimIndent()

    private val chaptersJson = """
        {
            "book_id": "1661",
            "title": "The Adventures of Sherlock Holmes",
            "author": "Arthur Conan Doyle",
            "total_chapters": 2,
            "chapters": [
                {
                    "chapter_index": 1,
                    "title": "I. A SCANDAL IN BOHEMIA",
                    "start_page": 1,
                    "end_page": 20,
                    "progress_percent": 0.0,
                    "char_count": 45000,
                    "pool_count": 5
                },
                {
                    "chapter_index": 2,
                    "title": "II. THE RED-HEADED LEAGUE",
                    "start_page": 21,
                    "end_page": 40,
                    "progress_percent": 8.3,
                    "char_count": 42000,
                    "pool_count": 8
                }
            ]
        }
    """.trimIndent()

    private val quizJson = """
        {
            "book_id": "1661",
            "title": "The Adventures of Sherlock Holmes",
            "author": "Arthur Conan Doyle",
            "chapter_index": 1,
            "chapter_title": "I. A SCANDAL IN BOHEMIA",
            "difficulty": "medium",
            "cache_hit": true,
            "total_pool_size": 15,
            "question_count": 2,
            "questions": [
                {
                    "id": 101,
                    "question": "Soru 1?",
                    "options": ["A) Şık 1", "B) Şık 2"],
                    "source_pages": [1],
                    "source_chapter": "I. A SCANDAL IN BOHEMIA"
                },
                {
                    "id": 102,
                    "question": "Soru 2?",
                    "options": ["A) Şık A", "B) Şık B"],
                    "source_pages": [2],
                    "source_chapter": "I. A SCANDAL IN BOHEMIA"
                }
            ]
        }
    """.trimIndent()

    private val evalJson = """
        {
            "total_questions": 2,
            "correct_count": 2,
            "score_percentage": 100.0,
            "result_badge": "Kusursuz!",
            "details": [
                {
                    "question_id": 101,
                    "question_index": 0,
                    "question": "Soru 1?",
                    "user_selected": "A) Şık 1",
                    "correct_answer": "A",
                    "correct_option": "A) Şık 1",
                    "is_correct": true,
                    "explanation": "Açıklama 1"
                },
                {
                    "question_id": 102,
                    "question_index": 1,
                    "question": "Soru 2?",
                    "user_selected": "A) Şık A",
                    "correct_answer": "A",
                    "correct_option": "A) Şık A",
                    "is_correct": true,
                    "explanation": "Açıklama 2"
                }
            ]
        }
    """.trimIndent()

    @Before
    fun setUp() {
        Dispatchers.setMain(testDispatcher)
        mockWebServer = MockWebServer()
        mockWebServer.start()
        apiClient = ApiClient(
            baseUrl = mockWebServer.url("/").toString().trimEnd('/'),
            ioDispatcher = testDispatcher
        )
    }

    @After
    fun tearDown() {
        mockWebServer.shutdown()
        Dispatchers.resetMain()
    }

    @Test
    fun initialState_loadsHealthAndDefaultBooks() = runTest {
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))

        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(AppScreen.LIBRARY, state.currentScreen)
        assertNotNull(state.health)
        assertEquals(79576, state.health?.totalBooks)
        assertEquals(1, state.searchResults.size)
        assertEquals("1661", state.searchResults.first().bookId)
        assertFalse(state.isSearching)
        assertFalse(state.isCheckingHealth)
    }

    private suspend fun kotlinx.coroutines.test.TestScope.viewModelWithSelectedChapter(): QuizViewModel {
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))
        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(chaptersJson))
        viewModel.selectBook(viewModel.uiState.value.searchResults.first())
        advanceUntilIdle()
        return viewModel
    }

    private val preparingJson = """
        {"status": "preparing", "job_id": "job1", "poll_url": "/api/v1/jobs/job1",
         "retry_after_seconds": 5, "message": "Bu bölümün soruları hazırlanıyor."}
    """.trimIndent()

    @Test
    fun startQuiz_whenPoolIsPreparing_pollsJobThenOpensQuiz() = runTest {
        val viewModel = viewModelWithSelectedChapter()

        mockWebServer.enqueue(MockResponse().setResponseCode(202).setBody(preparingJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(
            """{"id": "job1", "kind": "populate", "status": "running"}"""))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(
            """{"id": "job1", "kind": "populate", "status": "completed"}"""))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(quizJson))

        viewModel.startQuiz()
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(AppScreen.QUIZ, state.currentScreen)
        assertEquals(2, state.currentQuiz?.questions?.size)
        assertFalse(state.isLoadingQuiz)
        assertNull(state.quizPreparingMessage)
    }

    @Test
    fun startQuiz_whenPreparationFails_staysOnBookDetail() = runTest {
        val viewModel = viewModelWithSelectedChapter()

        mockWebServer.enqueue(MockResponse().setResponseCode(202).setBody(preparingJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(
            """{"id": "job1", "kind": "populate", "status": "failed", "error": "Bölüm metni bulunamadı"}"""))

        viewModel.startQuiz()
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(AppScreen.BOOK_DETAIL, state.currentScreen)
        assertNull(state.currentQuiz)
        assertFalse(state.isLoadingQuiz)
        assertNull(state.quizPreparingMessage)
    }

    @Test
    fun fullAppFlow_selectBook_selectChapter_startQuiz_answerAndEvaluate() = runTest {
        // Init responses
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))

        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()

        val book = viewModel.uiState.value.searchResults.first()

        // 1. Kitap Seçimi -> Chapters yüklenir ve BookDetailScreen'e geçer
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(chaptersJson))
        viewModel.selectBook(book)
        advanceUntilIdle()

        var state = viewModel.uiState.value
        assertEquals(AppScreen.BOOK_DETAIL, state.currentScreen)
        assertEquals("1661", state.selectedBook?.bookId)
        assertNotNull(state.chaptersData)
        assertEquals(2, state.chaptersData?.chapters?.size)
        // İlk bölüm otomatik seçilmiş olmalı
        assertEquals(1, state.selectedChapter?.chapterIndex)

        // 2. İkinci bölümü seçme
        val chapter2 = state.chaptersData!!.chapters[1]
        viewModel.selectChapter(chapter2)
        state = viewModel.uiState.value
        assertEquals(2, state.selectedChapter?.chapterIndex)

        // 3. Ayarları değiştirme (Zorluk & Soru sayısı)
        viewModel.setDifficulty("hard")
        viewModel.setQuizCount(10)
        assertEquals("hard", viewModel.uiState.value.quizDifficulty)
        assertEquals(10, viewModel.uiState.value.quizCount)

        // 4. Quizi Başlatma -> QuizScreen'e geçiş
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(quizJson))
        viewModel.startQuiz()
        advanceUntilIdle()

        state = viewModel.uiState.value
        assertEquals(AppScreen.QUIZ, state.currentScreen)
        assertNotNull(state.currentQuiz)
        assertEquals(2, state.currentQuiz?.questions?.size)
        assertEquals(0, state.currentQuestionIndex)

        // 5. Soruları Cevaplama & İleri-Geri Navigasyon
        viewModel.onOptionSelected(101, "A) Şık 1")
        assertEquals("A) Şık 1", viewModel.uiState.value.selectedAnswers[101])

        viewModel.nextQuestion()
        assertEquals(1, viewModel.uiState.value.currentQuestionIndex)

        viewModel.onOptionSelected(102, "A) Şık A")
        assertEquals("A) Şık A", viewModel.uiState.value.selectedAnswers[102])

        // Sınır testleri (Next boundary)
        viewModel.nextQuestion()
        assertEquals(1, viewModel.uiState.value.currentQuestionIndex)

        viewModel.prevQuestion()
        assertEquals(0, viewModel.uiState.value.currentQuestionIndex)

        viewModel.prevQuestion()
        assertEquals(0, viewModel.uiState.value.currentQuestionIndex)

        viewModel.goToQuestion(1)
        assertEquals(1, viewModel.uiState.value.currentQuestionIndex)

        // 6. Quizi Gönderme ve Değerlendirme -> ResultScreen
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(evalJson))
        viewModel.submitQuiz()
        advanceUntilIdle()

        state = viewModel.uiState.value
        assertEquals(AppScreen.RESULT, state.currentScreen)
        assertNotNull(state.evaluationResult)
        assertEquals(2, state.evaluationResult?.correctCount)
        assertEquals(100.0f, state.evaluationResult?.scorePercentage)
        assertEquals("Kusursuz!", state.evaluationResult?.resultBadge)

        // 7. Geri Dönüş Fonksiyonları
        viewModel.returnToChapters()
        assertEquals(AppScreen.BOOK_DETAIL, viewModel.uiState.value.currentScreen)
        assertNull(viewModel.uiState.value.currentQuiz)
        assertNull(viewModel.uiState.value.evaluationResult)

        viewModel.returnToLibrary()
        assertEquals(AppScreen.LIBRARY, viewModel.uiState.value.currentScreen)
        assertNull(viewModel.uiState.value.selectedBook)
    }

    @Test
    fun navigationBack_handlesAllScreensProperly() = runTest {
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))

        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()

        // SETTINGS -> Back -> LIBRARY
        viewModel.navigateTo(AppScreen.SETTINGS)
        assertEquals(AppScreen.SETTINGS, viewModel.uiState.value.currentScreen)
        viewModel.navigateBack()
        assertEquals(AppScreen.LIBRARY, viewModel.uiState.value.currentScreen)

        // BOOK_DETAIL -> Back -> LIBRARY
        viewModel.navigateTo(AppScreen.BOOK_DETAIL)
        viewModel.navigateBack()
        assertEquals(AppScreen.LIBRARY, viewModel.uiState.value.currentScreen)

        // QUIZ -> Back -> BOOK_DETAIL
        viewModel.navigateTo(AppScreen.QUIZ)
        viewModel.navigateBack()
        assertEquals(AppScreen.BOOK_DETAIL, viewModel.uiState.value.currentScreen)

        // RESULT -> Back -> BOOK_DETAIL
        viewModel.navigateTo(AppScreen.RESULT)
        viewModel.navigateBack()
        assertEquals(AppScreen.BOOK_DETAIL, viewModel.uiState.value.currentScreen)
    }

    @Test
    fun updateBaseUrl_trimsTrailingSlashAndTriggersHealthCheck() = runTest {
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))

        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()

        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        val testUrl = mockWebServer.url("/test/").toString()
        viewModel.updateBaseUrl(testUrl)
        advanceUntilIdle()

        val expected = testUrl.trimEnd('/')
        assertEquals(expected, viewModel.uiState.value.baseUrl)
        assertEquals(expected, apiClient.baseUrl)
    }

    @Test
    fun searchBooks_blankQuery_doesNotPerformNetworkCall() = runTest {
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))

        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()

        viewModel.searchBooks("   ")
        advanceUntilIdle()

        // Sunucuya yeni istek gitmemeli, toplam istek sayısı 2 kalmalı (health + default search)
        assertEquals(2, mockWebServer.requestCount)
    }

    @Test
    fun fetchChapters_networkError_setsChaptersErrorState() = runTest {
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))

        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()

        val book = viewModel.uiState.value.searchResults.first()

        // 404 dön
        mockWebServer.enqueue(MockResponse().setResponseCode(404).setBody("""{"detail": "Kitap bulunamadı"}"""))
        viewModel.selectBook(book)
        advanceUntilIdle()

        val state = viewModel.uiState.value
        assertEquals(AppScreen.BOOK_DETAIL, state.currentScreen)
        assertFalse(state.isLoadingChapters)
        assertNotNull(state.chaptersError)
        assertTrue(state.chaptersError?.contains("Kitap bulunamadı") == true)
        assertNull(state.chaptersData)
    }

    @Test
    fun resetQuiz_clearsCurrentQuizAndReturnsToBookDetail() = runTest {
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(healthJson))
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(searchJson))

        val viewModel = QuizViewModel(apiClient)
        advanceUntilIdle()

        val book = viewModel.uiState.value.searchResults.first()
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(chaptersJson))
        viewModel.selectBook(book)
        advanceUntilIdle()

        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(quizJson))
        viewModel.startQuiz()
        advanceUntilIdle()

        assertEquals(AppScreen.QUIZ, viewModel.uiState.value.currentScreen)
        viewModel.onOptionSelected(101, "A) Şık 1")

        // Reset
        viewModel.resetQuiz()
        val state = viewModel.uiState.value
        assertEquals(AppScreen.BOOK_DETAIL, state.currentScreen)
        assertNull(state.currentQuiz)
        assertNull(state.evaluationResult)
        assertTrue(state.selectedAnswers.isEmpty())
        assertEquals(0, state.currentQuestionIndex)
    }
}
