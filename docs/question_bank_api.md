# Headless Question Bank & Quiz Engine - Mobile Integration Guide

This guide explains how mobile apps (Flutter, React Native, Swift, Kotlin) integrate with the **Context-RAG headless engine**, which serves chapter quizzes alongside reading, with server-side grading and no spoilers from later chapters.

---

## Design Principles

1. **No user data (decoupled and headless)**
   - The service stores **no user data, sessions or score history**.
   - It only handles book texts, chapter boundaries, question pools and answer grading. User auth and leaderboards live in your own mobile backend.
2. **Persistent question pool, generation in the background**
   - Each chapter has a pool of verified questions in SQLite. `GET /api/v1/quiz/serve` only reads from this pool and never calls the LLM, so it answers in milliseconds.
   - If a chapter's pool is too small, the request does not wait for generation. A background job that fills the pool is queued and the response is `202 Accepted` with a job id. Generation runs on a single background worker (a local GPU processes one LLM request at a time) and can take a few minutes per chapter.
   - Concurrent requests for the same chapter share one job, and queued jobs survive server restarts.
3. **Only verified questions**
   - Every question is checked by a second LLM that must pick the same answer and quote verbatim evidence from the chapter. Questions without evidence are never served.
4. **Anti-cheat masking**
   - When a quiz is served (`GET /api/v1/quiz/serve?hide_answers=true`, the default), `correct_answer` and `explanation` are `null`. The app never knows the correct option.
   - When the user answers, the app sends only `question_id` and the selected option (`POST /api/v1/quiz/evaluate-answers`). The server grades them.
5. **Anti-spoiler**
   - Questions are locked to the `chapter_index` the user has read. Information from later chapters cannot leak into them.

---

## Endpoints

### 1. Import a book and its chapters
Stores the content and chapter boundaries when a new EPUB or PDF is added to your library.

- **URL:** `POST /api/v1/books/import`
- **Content-Type:** `multipart/form-data`
- **Header:** `X-Admin-Key: {ADMIN_API_KEY}` (admin endpoint; disabled if `ADMIN_API_KEY` is not set on the server)

#### Form fields
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | EPUB (`.epub`) or PDF (`.pdf`) file |
| `book_id` | String | No | Custom book ID (derived from the file name if omitted) |
| `title` | String | No | Book title (taken from the file or its metadata if omitted) |
| `author` | String | No | Author name |
| `prepopulate` | Bool | No | `true` queues a pool-filling job for every chapter, so quizzes are ready before users arrive |

#### Example request (cURL)
```bash
curl -X POST http://localhost:8000/api/v1/books/import \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -F "file=@nutuk.epub" \
  -F "book_id=nutuk" \
  -F "title=Nutuk" \
  -F "author=Mustafa Kemal Atatürk" \
  -F "prepopulate=true"
```

#### Example response (200 OK)
```json
{
  "book_id": "nutuk",
  "title": "Nutuk",
  "author": "Mustafa Kemal Atatürk",
  "source_format": "epub",
  "total_chapters": 12,
  "total_pages": 240,
  "chapters": [
    {
      "chapter_index": 1,
      "title": "1. Bölüm: Samsun'a Çıkış",
      "start_page": 1,
      "end_page": 18,
      "progress_percent": 8.3,
      "char_count": 34500,
      "pool_count": 0
    }
  ],
  "queued_jobs": 12
}
```

When the same book is imported again, the pool of every chapter whose text changed is deleted and chapters that no longer exist are removed, so questions never end up attached to the wrong chapter.

---

### 2. List a book's chapters and pool status
Returns the chapters and the number of servable (verified) questions in each chapter's pool.

- **URL:** `GET /api/v1/books/{book_id}/chapters`

```bash
curl http://localhost:8000/api/v1/books/nutuk/chapters
```

Gutenberg books do not need to be imported: their chapters are detected from the Gutenberg EPUB table of contents (or headings in the plain text) on first request.

---

### 3. Get a chapter quiz
Call this when the user finishes a chapter. This endpoint never calls the LLM; it only reads verified questions from the pool.

- **URL:** `GET /api/v1/quiz/serve`
- **Query parameters:**
  - `book_id` *(string, required)*: book ID (e.g. `nutuk` or `1661`)
  - `chapter_index` *(int, default 1)*: 1-based chapter number
  - `count` *(int, default 5)*: number of questions (1-20)
  - `difficulty` *(string, default `medium`)*: `easy` \| `medium` \| `hard`
  - `hide_answers` *(bool, default `true`)*: hides the correct answers. `false` is only accepted with `X-Admin-Key` (otherwise `403`).

```bash
curl "http://localhost:8000/api/v1/quiz/serve?book_id=nutuk&chapter_index=1&count=5&hide_answers=true"
```

#### 200 OK: the pool has enough questions
```json
{
  "book_id": "nutuk",
  "title": "Nutuk",
  "author": "Mustafa Kemal Atatürk",
  "chapter_index": 1,
  "chapter_title": "1. Bölüm: Samsun'a Çıkış",
  "difficulty": "medium",
  "cache_hit": true,
  "total_pool_size": 15,
  "question_count": 5,
  "questions": [
    {
      "id": 101,
      "question": "Mustafa Kemal Paşa Samsun'a hangi görevle ayak basmıştır?",
      "options": [
        "A) Sadrazam Vekili",
        "B) 9. Ordu Müfettişi",
        "C) Harbiye Nazırı",
        "D) Kolordu Komutanı"
      ],
      "source_pages": [1],
      "source_chapter": "1. Bölüm: Samsun'a Çıkış",
      "progress_percent": 8.3,
      "correct_answer": null,
      "explanation": null
    }
  ]
}
```

#### 202 Accepted: the pool is being prepared
If the pool is too small, a job that fills it is queued (or an existing one for the same chapter is reused) and the response carries a `Retry-After` header:
```json
{
  "status": "preparing",
  "job_id": "3f9a1c2b7d4e5f60",
  "poll_url": "/api/v1/jobs/3f9a1c2b7d4e5f60",
  "retry_after_seconds": 15,
  "message": "Bu bölümün soruları hazırlanıyor. Biraz sonra tekrar deneyin."
}
```

Client flow:
1. Show a "preparing questions" state.
2. Poll `GET /api/v1/jobs/{job_id}` every `retry_after_seconds`. The status moves `queued` → `running` → `completed` | `failed`; while `queued`, `queue_position` is the number of jobs ahead of it.
3. When the job is `completed`, repeat the same `serve` request. It now returns `200`.
4. If the job is `failed`, show an error and let the user retry later.

---

### 4. Grade answers by question ID (anti-cheat)
The app sends the options the user picked, together with the question IDs.

- **URL:** `POST /api/v1/quiz/evaluate-answers`
- **Content-Type:** `application/json`

#### Request body
`selected_option` can be the full option text or just the letter.
```json
{
  "answers": [
    {
      "question_id": 101,
      "selected_option": "B) 9. Ordu Müfettişi"
    },
    {
      "question_id": 102,
      "selected_option": "A"
    }
  ]
}
```

#### Example response (200 OK)
```json
{
  "total_questions": 2,
  "correct_count": 2,
  "score_percentage": 100.0,
  "result_badge": "Kusursuz! 🏆",
  "details": [
    {
      "question_id": 101,
      "question_index": 0,
      "question": "Mustafa Kemal Paşa Samsun'a hangi görevle ayak basmıştır?",
      "user_selected": "B) 9. Ordu Müfettişi",
      "correct_answer": "B",
      "correct_option": "B) 9. Ordu Müfettişi",
      "is_correct": true,
      "explanation": "Mustafa Kemal Paşa bölgedeki asayişi sağlamak üzere 9. Ordu Müfettişliği göreviyle Samsun'a çıkmıştır."
    }
  ]
}
```

`result_badge` is a ready-made Turkish label from the server, based on the score: 100% `Kusursuz! 🏆`, 80%+ `Harika! 🌟`, 60%+ `İyi İş! 👍`, otherwise `Tekrar Dene! 📚`.

---

### 5. Pre-populate a question pool
Fills a chapter's pool before users reach it, so they never see the preparing state.

- **URL:** `POST /api/v1/quiz/populate?book_id={book_id}&chapter_index={chapter_index}&target_count=15&difficulty=medium`
- **Header:** `X-Admin-Key: {ADMIN_API_KEY}`
- `target_count` (1-30, default 10) is the total number of verified questions the pool should have; if the pool is partially filled, only the missing ones are generated.
- The job is queued and the response is `202 Accepted`. Track it with `GET /api/v1/jobs/{id}`.
- Importing a book with `prepopulate=true` queues this job for every chapter.

#### Example response (202 Accepted)
```json
{
  "id": "3f9a1c2b7d4e5f60",
  "kind": "populate",
  "status": "queued",
  "queue_position": 0,
  "params": {"book_id": "nutuk", "chapter_index": 1, "difficulty": "medium", "target_count": 15},
  "poll_url": "/api/v1/jobs/3f9a1c2b7d4e5f60"
}
```
When the job is completed, `result` is `{"generated": 15, "total_pool_size": 15}`.

---

## Security and Mobile Tips
- **Anti-cheat:** always leave `hide_answers` as `true` in the mobile client (it is the default).
- **Preparing state:** for a chapter with an empty pool, `serve` returns `202` and preparation on a local GPU can take a few minutes. Show a "preparing questions" state and poll `poll_url` (the Android client does this). To avoid any waiting, import books with `prepopulate=true` or run `populate` ahead of time.
- **Admin key:** never ship `ADMIN_API_KEY` in the mobile app. Admin endpoints (`/books/import`, `/quiz/populate`, `/quiz/reverify`, `/admin/*`) belong in your backend or ops tooling.
