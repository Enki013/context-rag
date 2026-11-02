# 📚 Context RAG - Mobile & Web Quiz API

Yapay zeka ve RAG (Retrieval-Augmented Generation) mimarisiyle güçlendirilmiş, **Project Gutenberg (79,000+ kitap)** ve **kullanıcı PDF belgeleri** üzerinden çoktan seçmeli quiz üreten ve değerlendiren backend servisi.

---

## 🚀 Hızlı Başlangıç

### 1. API Sunucusunu Başlatma
```bash
# Sanal ortamı etkinleştirin
source venv/bin/activate

# Admin uçları (/books/import, /quiz/populate) için anahtar (tanımlı değilse bu uçlar kapalıdır)
export ADMIN_API_KEY="uzun-rastgele-bir-deger"

# FastAPI sunucusunu başlatın (Port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Swagger UI (İnteraktif Test Arayüzü):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Dokümantasyonu:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Postman Koleksiyonu:** Proje kök dizinindeki `Context_RAG_Quiz_API.postman_collection.json` dosyasını Postman'e aktarabilirsiniz.

### 2. Streamlit Web Panelini Başlatma (İsteğe bağlı)
```bash
streamlit run ui/app.py --server.port 8501
```
Web Arayüzü: [http://localhost:8501](http://localhost:8501)

---

## 📱 Mobil Uygulama Geliştiricileri İçin Notlar

Mobil emülatör veya gerçek cihazdan yerel makinenize bağlanırken:
- **Android Emülatör:** `http://10.0.2.2:8000`
- **iOS Simülatör:** `http://localhost:8000` veya `http://127.0.0.1:8000`
- **Gerçek Cihaz (Wi-Fi):** Bilgisayarınızın yerel IP adresi (örn. `http://192.168.1.XX:8000`)
- **CORS:** Tüm `origin`, `method` ve `header`'lar için varsayılan olarak açıktır.

---

## 📡 API Endpoint Referansı

### 1. Sistem Durumu
```http
GET /api/v1/health
```
Ollama bağlantısını, aktif LLM (`qwen2.5:14b`) ve embedding (`bge-m3`) modellerini ve kayıtlı kitap sayısını döner.

---

### 2. Kitap Kataloğu & Arama

#### 🔍 Kitap Ara
```http
GET /api/v1/books/search?q={query}&limit={limit}
```
*Örnek:* `GET /api/v1/books/search?q=Sherlock Holmes&limit=10`
*Yanıt:*
```json
{
  "query": "Sherlock Holmes",
  "total": 5,
  "books": [
    {
      "book_id": "1661",
      "title": "The Adventures of Sherlock Holmes",
      "author": "Arthur Conan Doyle",
      "subjects": "Detective and mystery stories",
      "bookshelves": "Detective Fiction",
      "language": "en",
      "is_cached_locally": true
    }
  ]
}
```

#### 📖 Kitap Detayı
```http
GET /api/v1/books/{book_id}
```
*Örnek:* `GET /api/v1/books/1661`

---

### 3. Quiz Üretimi

#### 📚 Kitaptan Quiz Üret
```http
POST /api/v1/quiz/generate-from-book
Content-Type: application/json
```
*Gövde:*
```json
{
  "book_id": "1661",
  "title": "The Adventures of Sherlock Holmes",
  "author": "Arthur Conan Doyle",
  "num_questions": 5,
  "difficulty": "orta",
  "question_type": "multiple_choice"
}
```
*Yanıt:*
```json
{
  "source_type": "book",
  "identifier": "1661",
  "title": "The Adventures of Sherlock Holmes",
  "author": "Arthur Conan Doyle",
  "total_questions": 5,
  "questions": [
    {
      "question": "Sherlock Holmes hangi sokakta yaşamaktadır?",
      "options": [
        "A) Baker Street 221B",
        "B) Oxford Street 12",
        "C) Fleet Street 4",
        "D) Abbey Road 10"
      ],
      "correct_answer": "A",
      "explanation": "Holmes ve Watson, Baker Street 221B adresinde ikamet etmektedir.",
      "source_pages": []
    }
  ]
}
```

#### 📄 PDF'ten Quiz Üret
```http
POST /api/v1/quiz/generate-from-pdf
Content-Type: multipart/form-data
```
*Form Alanları:*
- `file`: PDF dosyası (Binary, maks. 50 MB)
- `num_questions`: int (varsayılan 5, 1-10 arası)
- `difficulty`: string ("kolay" | "orta" | "zor")
- `question_type`: string ("multiple_choice")

#### 📱 EPUB'tan Quiz Üret
```http
POST /api/v1/quiz/generate-from-epub
Content-Type: multipart/form-data
```
*Form Alanları:*
- `file`: EPUB dosyası (Binary, maks. 50 MB)
- `num_questions`: int (varsayılan 5, 1-10 arası)
- `difficulty`: string ("kolay" | "orta" | "zor")
- `question_type`: string ("multiple_choice")

---

### 4. Headless Soru Bankası & Mobil Quiz Servisi (0ms Cache & Anti-Cheat)

Mobil uygulamalar için kullanıcı verisinden bağımsız, kalıcı soru havuzlu (<1ms Cache Hit) ve hile korumalı quiz motoru.

> Detaylı mobil entegrasyon kılavuzu için [docs/question_bank_api.md](docs/question_bank_api.md) belgesini inceleyin.

#### 📥 Kitap & Bölüm İçe Aktar
```http
POST /api/v1/books/import
Content-Type: multipart/form-data
X-Admin-Key: {ADMIN_API_KEY}
```
EPUB veya PDF dosyasını okur, gömülü içindekiler (TOC) ve sayfalarla otomatik bölümler ve sisteme kaydeder. **Admin uç:** `X-Admin-Key` gerekir.
Aynı kitap yeniden içe aktarıldığında metni değişen bölümlerin soru havuzu silinir, artık olmayan bölümler kaldırılır (yanlış bölüme düşen soru/spoiler olmasın diye).

#### 🎯 Bölüm Quizi Getir (Cache Hit / On-Demand)
```http
GET /api/v1/quiz/serve?book_id={book_id}&chapter_index={chapter_index}&count=5&difficulty=orta&hide_answers=true
```
- Havuzda soru varsa: **0 milisaniyede** havuzdan rastgele soru döner (`cache_hit: true`).
- Havuz boşsa: LLM ile ilgili bölüm metninden sorular üretip havuza kaydeder (`cache_hit: false`).
- `hide_answers=true`: Doğru şıklar ve açıklamalar istemciye sızdırılmaz (`correct_answer: null`).
- `hide_answers=false` yalnızca geçerli `X-Admin-Key` header'ı ile kabul edilir; aksi halde `403`.

#### 🛡️ Hile Korumalı Cevap Değerlendirme (Soru ID Bazlı)
```http
POST /api/v1/quiz/evaluate-answers
Content-Type: application/json
```
```json
{
  "answers": [
    {
      "question_id": 101,
      "selected_option": "A) Baker Street 221B"
    }
  ]
}
```
Mobil istemci doğru şıkkı bilmeden yalnızca soru ID'si ve seçtiği şıkkı gönderir; sunucu veritabanında doğrular ve puanlama yapar.

#### ⚡ Soru Havuzunu Önceden Doldur (Pre-populate Pool)
```http
POST /api/v1/quiz/populate?book_id={book_id}&chapter_index={chapter_index}&target_count=15&difficulty=orta
X-Admin-Key: {ADMIN_API_KEY}
```
Arka planda (Admin / Cron) bir bölümün soru havuzunu `target_count` soruya kadar doldurur; havuz kısmen doluysa yalnızca eksik kadar üretir. **Admin uç:** `X-Admin-Key` gerekir.

---

### 5. Geriye Dönük Uyumluluk: Quiz Cevaplarını Değerlendir

```http
POST /api/v1/quiz/evaluate
Content-Type: application/json
```
Kullanıcının gönderdiği test şıklarını orijinal quiz listesiyle karşılaştırarak puanlar.

---

## 📬 Postman İle Test Etme

1. Postman uygulamasını açın.
2. Sol üstteki **Import** butonuna tıklayın.
3. Proje dizinindeki `Context_RAG_Quiz_API.postman_collection.json` dosyasını seçin.
4. Koleksiyon değişkeni `baseUrl` varsayılan olarak `http://localhost:8000` ayarlıdır.
5. Koleksiyon içerisindeki 6 ana kategori altındaki tüm istekleri (Arama, EPUB/PDF Import, Bölüm Quizi, Hile Korumalı Değerlendirme vb.) hemen test edebilirsiniz!

