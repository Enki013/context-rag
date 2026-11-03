# 📱 Headless Question Bank & Quiz Engine - Mobil Entegrasyon Kılavuzu

Bu doküman, mobil uygulamaların (Flutter, React Native, Swift, Kotlin) kitap okuma sürecine paralel olarak hile korumalı, spoiler içermeyen ve milisaniyeler içinde quiz servis eden **Context-RAG Headless Engine** ile entegrasyonunu açıklar.

---

## 🎯 Temel Tasarım İlkeleri

1. **Kullanıcı Verisinden Bağımsız (Decoupled & Headless):**
   - Bu servis **hiçbir kullanıcı verisi, oturum veya skor geçmişi saklamaz**.
   - Yalnızca kitap metinleri, bölüm sınırları, soru havuzları ve cevap doğrulamasıyla ilgilenir. Kullanıcı auth ve liderlik tabloları ana mobil backend'inizde yönetilir.
2. **Kalıcı Soru Havuzu & Önbellek (<1ms Cache Hit):**
   - İlgili bölümün havuzunda soru varsa LLM çalıştırılmaz; SQLite üzerinden rastgele sorular anında döndürülür.
   - Havuzda soru yoksa (`Cache Miss`), LLM ihtiyaç anında çalışıp soruları üretir, havuza kaydeder ve döner.
3. **Hile Koruması (Anti-Cheat Masking):**
   - Quiz dağıtılırken (`GET /api/v1/quiz/serve?hide_answers=true`), `correct_answer` ve `explanation` alanları `null` döner. Mobil uygulama doğru cevabı bilmez.
   - Kullanıcı soruları yanıtladığında yalnızca `question_id` ve seçtiği şıkkı (`POST /api/v1/quiz/evaluate-answers`) gönderir. Sunucu doğrular.
4. **Spoiler Koruması (Anti-Spoiler):**
   - Sorular kullanıcının okuduğu `chapter_index` sınırlarına kilitlidir. Kitabın sonraki bölümlerinden bilgi sızdırılamaz.

---

## 📡 Uç Noktalar (Endpoints)

### 1. Kitap ve Bölümleri İçe Aktarma
Mobil kütüphanenize yeni bir EPUB veya PDF eklendiğinde içeriği ve bölüm sınırlarını veritabanına kaydeder.

- **URL:** `POST /api/v1/books/import`
- **Content-Type:** `multipart/form-data`
- **Header:** `X-Admin-Key: {ADMIN_API_KEY}` (admin uç; sunucuda `ADMIN_API_KEY` tanımlı değilse kapalıdır)

#### Form Parametreleri:
| Alan | Tip | Zorunlu | Açıklama |
|---|---|---|---|
| `file` | File | Evet | EPUB (`.epub`) veya PDF (`.pdf`) ikili dosyası |
| `book_id` | String | Hayır | Özel kitap ID (verilmezse dosya adından üretilir) |
| `title` | String | Hayır | Kitap adı (belirtilmezse dosya/metadata'dan alınır) |
| `author` | String | Hayır | Yazar adı |

#### Örnek İstek (cURL):
```bash
curl -X POST http://localhost:8000/api/v1/books/import \
  -H "X-Admin-Key: $ADMIN_API_KEY" \
  -F "file=@nutuk.epub" \
  -F "book_id=nutuk" \
  -F "title=Nutuk" \
  -F "author=Mustafa Kemal Atatürk"
```

#### Örnek Yanıt (200 OK):
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
      "pool_count": 15
    }
  ]
}
```

---

### 2. Kitabın Bölümlerini ve Havuz Durumunu Sorgulama
Kullanıcının okuma durumuna göre bölümleri ve her bölümün soru havuzu sayısını döner.

- **URL:** `GET /api/v1/books/{book_id}/chapters`

#### Örnek İstek:
```bash
curl http://localhost:8000/api/v1/books/nutuk/chapters
```

---

### 3. Bölüm Quizi Getir (Cache Hit / On-Demand Serving)
Kullanıcı bir bölümü bitirdiğinde quizi çeker.

- **URL:** `GET /api/v1/quiz/serve`
- **Query Parametreleri:**
  - `book_id` *(string, zorunlu)*: Kitap ID (örn: `nutuk` veya `1661`)
  - `chapter_index` *(int, varsayılan 1)*: 1-tabanlı bölüm numarası
  - `count` *(int, varsayılan 5)*: İstenen soru adedi (1-20)
  - `difficulty` *(string, varsayılan "medium")*: `easy` \| `medium` \| `hard`
  - `hide_answers` *(bool, varsayılan true)*: `true` ise doğru cevaplar gizlenir. `false` yalnızca `X-Admin-Key` ile kabul edilir (aksi halde `403`).

#### Örnek İstek:
```bash
curl "http://localhost:8000/api/v1/quiz/serve?book_id=nutuk&chapter_index=1&count=5&hide_answers=true"
```

#### Örnek Yanıt (Hile Korumalı, <1ms Cache Hit):
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

---

### 4. Hile Korumalı Cevap Değerlendirme (Soru ID Bazlı)
Mobil istemci kullanıcının seçtiği şıkları soru ID'leri ile gönderir.

- **URL:** `POST /api/v1/quiz/evaluate-answers`
- **Content-Type:** `application/json`

#### İstek Gövdesi:
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

#### Örnek Yanıt (200 OK):
```json
{
  "total_questions": 2,
  "correct_count": 2,
  "score_percentage": 100.0,
  "result_badge": "Harika! 🏆",
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

---

### 5. Arka Plan Soru Havuzunu Önceden Doldurma (Pre-populate Pool)
Kullanıcılar okumaya başlamadan önce veya bir kitap sisteme yüklendiğinde arka plan worker'ı tarafından havuzu şişirmek için çağrılır.

- **URL:** `POST /api/v1/quiz/populate?book_id={book_id}&chapter_index={chapter_index}&target_count=15&difficulty=medium`
- **Header:** `X-Admin-Key: {ADMIN_API_KEY}`
- `target_count` havuzda olması hedeflenen toplam soru sayısıdır; havuz kısmen doluysa yalnızca eksik kadar üretilir.

#### Örnek Yanıt:
```json
{
  "status": "ok",
  "book_id": "nutuk",
  "chapter_index": 1,
  "difficulty": "medium",
  "generated": 15,
  "total_pool_size": 15,
  "message": "Bölüm 1 havuzu hazırlandı. Toplam 15 soru mevcut."
}
```

---

## 🔒 Güvenlik ve Mobil İpuçları
- **Hile Koruması:** `GET /api/v1/quiz/serve` çağrısında `hide_answers` parametresini mobil istemcide daima `true` bırakın (varsayılan zaten `true`).
- **Ağ Dayanıklılığı:** İlk kez açılan ve havuzu henüz boş olan bir bölüm için `cache_miss` olacağı için ilk çağrı 5-10 saniye sürebilir. Bu nedenle mobil backend üzerinden `populate` job'ı çalıştırmak veya mobil uygulamada ilk çağrıda şık bir yükleme animasyonu göstermek önerilir.
