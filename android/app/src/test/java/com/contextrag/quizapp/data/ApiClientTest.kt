package com.contextrag.quizapp.data

import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class ApiClientTest {

    private lateinit var mockWebServer: MockWebServer
    private lateinit var apiClient: ApiClient

    @Before
    fun setUp() {
        mockWebServer = MockWebServer()
        mockWebServer.start()
        apiClient = ApiClient(baseUrl = mockWebServer.url("/").toString().trimEnd('/'))
    }

    @After
    fun tearDown() {
        mockWebServer.shutdown()
    }

    @Test
    fun checkHealth_success() = runBlocking {
        val json = """
            {
                "status": "ok",
                "ollama_connected": true,
                "llm_model": "qwen2.5:14b",
                "embedding_model": "bge-m3",
                "catalog_installed": true,
                "total_books": 79576
            }
        """.trimIndent()
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(json))

        val result = apiClient.checkHealth()
        assertTrue("Expected success", result.isSuccess)
        val health = result.getOrNull()
        assertNotNull(health)
        assertEquals("ok", health?.status)
        assertTrue(health?.ollamaConnected == true)
        assertEquals(79576, health?.totalBooks)
        assertEquals("qwen2.5:14b", health?.llmModel)
    }

    @Test
    fun checkHealth_serverError() = runBlocking {
        val json = """{"detail": "Ollama servisine erişilemedi"}"""
        mockWebServer.enqueue(MockResponse().setResponseCode(503).setBody(json))

        val result = apiClient.checkHealth()
        assertTrue("Expected failure", result.isFailure)
        val error = result.exceptionOrNull()
        assertNotNull(error)
        assertTrue(error?.message?.contains("Ollama servisine erişilemedi") == true)
    }

    @Test
    fun searchBooks_success() = runBlocking {
        val json = """
            {
                "query": "Sherlock",
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
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(json))

        val result = apiClient.searchBooks("Sherlock", limit = 10)
        assertTrue("Expected success", result.isSuccess)
        val search = result.getOrNull()
        assertNotNull(search)
        assertEquals(1, search?.total)
        assertEquals(1, search?.books?.size)
        assertEquals("1661", search?.books?.first()?.bookId)
        assertTrue(search?.books?.first()?.isCachedLocally == true)

        val recordedRequest = mockWebServer.takeRequest()
        assertTrue(recordedRequest.path?.contains("/api/v1/books/search?q=Sherlock&limit=10") == true)
    }

    @Test
    fun getBookChapters_success() = runBlocking {
        val json = """
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
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(json))

        val result = apiClient.getBookChapters("1661")
        assertTrue("Expected success", result.isSuccess)
        val data = result.getOrNull()
        assertNotNull(data)
        assertEquals("1661", data?.bookId)
        assertEquals(2, data?.totalChapters)
        assertEquals(2, data?.chapters?.size)
        assertEquals("I. A SCANDAL IN BOHEMIA", data?.chapters?.get(0)?.title)
        assertEquals(5, data?.chapters?.get(0)?.poolCount)
    }

    @Test
    fun serveQuiz_success() = runBlocking {
        val json = """
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
                        "question": "Irene Adler hangi unvanla bilinmektedir?",
                        "options": ["A) The Woman", "B) The Queen", "C) The Lady", "D) The Countess"],
                        "source_pages": [5],
                        "source_chapter": "I. A SCANDAL IN BOHEMIA",
                        "progress_percent": 2.5
                    },
                    {
                        "id": 102,
                        "question": "Bohemya Kralı Holmes'a ne getirmiştir?",
                        "options": ["A) Bir mektup", "B) Bir fotoğraf", "C) Bir yüzük", "D) Bir kutu"],
                        "source_pages": [7],
                        "source_chapter": "I. A SCANDAL IN BOHEMIA",
                        "progress_percent": 3.8
                    }
                ]
            }
        """.trimIndent()
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(json))

        val result = apiClient.serveQuiz("1661", chapterIndex = 1, count = 2, difficulty = "medium", hideAnswers = true)
        assertTrue(result.isSuccess)
        val quiz = result.getOrNull()
        assertNotNull(quiz)
        assertTrue(quiz?.cacheHit == true)
        assertEquals(2, quiz?.questions?.size)
        assertEquals(101, quiz?.questions?.get(0)?.id)
        assertEquals(4, quiz?.questions?.get(0)?.options?.size)
    }

    @Test
    fun evaluateAnswers_success() = runBlocking {
        val json = """
            {
                "total_questions": 2,
                "correct_count": 2,
                "score_percentage": 100.0,
                "result_badge": "Kusursuz! 🌟",
                "details": [
                    {
                        "question_id": 101,
                        "question_index": 0,
                        "question": "Irene Adler?",
                        "user_selected": "A) The Woman",
                        "correct_answer": "A",
                        "correct_option": "A) The Woman",
                        "is_correct": true,
                        "explanation": "To Sherlock Holmes she is always THE woman."
                    },
                    {
                        "question_id": 102,
                        "question_index": 1,
                        "question": "Bohemya Kralı?",
                        "user_selected": "B) Bir fotoğraf",
                        "correct_answer": "B",
                        "correct_option": "B) Bir fotoğraf",
                        "is_correct": true,
                        "explanation": "Kral, Adler ile çekilmiş fotoğrafın peşindedir."
                    }
                ]
            }
        """.trimIndent()
        mockWebServer.enqueue(MockResponse().setResponseCode(200).setBody(json))

        val answers = listOf(
            AnswerItem(101, "A) The Woman"),
            AnswerItem(102, "B) Bir fotoğraf")
        )
        val result = apiClient.evaluateAnswers(answers)
        assertTrue(result.isSuccess)
        val eval = result.getOrNull()
        assertNotNull(eval)
        assertEquals(2, eval?.totalQuestions)
        assertEquals(2, eval?.correctCount)
        assertEquals(100.0f, eval?.scorePercentage)
        assertEquals("Kusursuz! 🌟", eval?.resultBadge)
        assertTrue(eval?.details?.all { it.isCorrect } == true)

        val recordedRequest = mockWebServer.takeRequest()
        assertEquals("POST", recordedRequest.method)
        assertEquals("/api/v1/quiz/evaluate-answers", recordedRequest.path)
        val body = recordedRequest.body.readUtf8()
        assertTrue(body.contains("\"question_id\":101"))
        assertTrue(body.contains("\"selected_option\":\"A) The Woman\""))
    }
}
