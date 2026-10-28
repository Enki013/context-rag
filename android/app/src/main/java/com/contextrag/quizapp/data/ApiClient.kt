package com.contextrag.quizapp.data

import com.google.gson.Gson
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

class ApiClient(
    var baseUrl: String = "http://10.0.2.2:8000",
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    private fun extractErrorMessage(response: okhttp3.Response): String {
        return try {
            val errorBody = response.body?.string() ?: ""
            val json = gson.fromJson(errorBody, com.google.gson.JsonObject::class.java)
            json?.get("detail")?.asString ?: "HTTP ${response.code}: ${response.message}"
        } catch (_: Exception) {
            "HTTP ${response.code}: ${response.message}"
        }
    }

    suspend fun checkHealth(): Result<HealthResponse> = withContext(ioDispatcher) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/api/v1/health")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException(extractErrorMessage(response)))
                }
                val body = response.body?.string() ?: ""
                val data = gson.fromJson(body, HealthResponse::class.java)
                Result.success(data)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun searchBooks(query: String, limit: Int = 15): Result<BookSearchResponse> = withContext(ioDispatcher) {
        try {
            val url = "$baseUrl/api/v1/books/search?q=${java.net.URLEncoder.encode(query, "UTF-8")}&limit=$limit"
            val request = Request.Builder().url(url).get().build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException(extractErrorMessage(response)))
                }
                val body = response.body?.string() ?: ""
                val data = gson.fromJson(body, BookSearchResponse::class.java)
                Result.success(data)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun getBookChapters(bookId: String): Result<BookChaptersResponse> = withContext(ioDispatcher) {
        try {
            val url = "$baseUrl/api/v1/books/$bookId/chapters"
            val request = Request.Builder().url(url).get().build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException(extractErrorMessage(response)))
                }
                val body = response.body?.string() ?: ""
                val data = gson.fromJson(body, BookChaptersResponse::class.java)
                Result.success(data)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun serveQuiz(
        bookId: String,
        chapterIndex: Int = 1,
        count: Int = 5,
        difficulty: String = "orta",
        hideAnswers: Boolean = true
    ): Result<ServeQuizResponse> = withContext(ioDispatcher) {
        try {
            val url = "$baseUrl/api/v1/quiz/serve?book_id=${java.net.URLEncoder.encode(bookId, "UTF-8")}&chapter_index=$chapterIndex&count=$count&difficulty=$difficulty&hide_answers=$hideAnswers"
            val request = Request.Builder().url(url).get().build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException(extractErrorMessage(response)))
                }
                val body = response.body?.string() ?: ""
                val data = gson.fromJson(body, ServeQuizResponse::class.java)
                Result.success(data)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun evaluateAnswers(answers: List<AnswerItem>): Result<EvaluationResponse> = withContext(ioDispatcher) {
        try {
            val payload = gson.toJson(EvaluationRequest(answers))
            val body = payload.toRequestBody(jsonMediaType)
            val request = Request.Builder()
                .url("$baseUrl/api/v1/quiz/evaluate-answers")
                .post(body)
                .build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return@withContext Result.failure(IOException(extractErrorMessage(response)))
                }
                val respBody = response.body?.string() ?: ""
                val data = gson.fromJson(respBody, EvaluationResponse::class.java)
                Result.success(data)
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
