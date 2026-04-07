# Context RAG - PDF Quiz Generator
## Product Requirements Document (PRD)

---

## 1. Proje Özeti

**Context RAG**, kullanıcının yüklediği PDF dokümanlarını akıllı bir şekilde parçalara (chunk) ayırarak, bu parçalardan otomatik quiz soruları üreten bir RAG (Retrieval-Augmented Generation) sistemidir. Sistem tamamen lokal çalışır ve LLM altyapısı olarak **Ollama** kullanır.

---

## 2. Problem Tanımı

- Öğrenciler ve profesyoneller uzun PDF dokümanlarını okuyup kavramak için çok zaman harcıyor.
- Okunan materyalden kendini test etmek için manuel soru hazırlamak verimsiz.
- Mevcut çözümlerin çoğu bulut tabanlı, gizlilik endişesi yaratıyor ve internet bağlantısı gerektiriyor.

---

## 3. Çözüm

Tamamen lokal çalışan, PDF'den otomatik quiz üreten bir RAG pipeline'ı:

1. **PDF Yükleme** → Kullanıcı PDF dosyasını sisteme verir
2. **Metin Çıkarma** → PDF'den ham metin extract edilir
3. **Chunking (Overlap ile)** → Metin, bağlamı koruyacak şekilde örtüşmeli parçalara ayrılır
4. **Embedding & Vektör Depolama** → Chunk'lar vektör veritabanında saklanır
5. **Quiz Üretimi** → Ollama LLM kullanarak chunk'lardan quiz soruları generate edilir
6. **Sonuç Sunumu** → Sorular kullanıcıya sunulur

---

## 4. Teknik Mimari

### 4.1 Teknoloji Stack'i

| Bileşen | Teknoloji | Neden |
|---------|-----------|-------|
| **Dil** | Python 3.11+ | Zengin NLP/ML ekosistemi |
| **LLM** | Ollama (llama3.2 veya mistral) | Lokal, ücretsiz, hızlı |
| **PDF İşleme** | PyMuPDF (fitz) | Hızlı ve güvenilir PDF parsing |
| **Chunking** | LangChain Text Splitters | Overlap destekli, esnek chunking |
| **Embedding** | Ollama Embeddings (nomic-embed-text) | Lokal embedding, Ollama ile uyumlu |
| **Vektör DB** | ChromaDB | Lightweight, local-first, kolay kurulum |
| **API Framework** | FastAPI | Modern, async, otomatik docs |
| **Frontend** | Streamlit | Hızlı prototipleme, Python-native |

### 4.2 Sistem Diyagramı

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐
│  PDF Upload  │────▶│  PDF Parser   │────▶│  Text Extraction    │
└─────────────┘     └──────────────┘     └─────────┬───────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────────┐
                                          │  Chunking Engine     │
                                          │  (overlap=200 char)  │
                                          └─────────┬───────────┘
                                                    │
                                          ┌─────────▼───────────┐
                                          │  Embedding Layer     │
                                          │  (nomic-embed-text)  │
                                          └─────────┬───────────┘
                                                    │
                                          ┌─────────▼───────────┐
                                          │  ChromaDB            │
                                          │  (Vector Store)      │
                                          └─────────┬───────────┘
                                                    │
                                          ┌─────────▼───────────┐
                                          │  Quiz Generator      │
                                          │  (Ollama LLM)        │
                                          └─────────┬───────────┘
                                                    │
                                          ┌─────────▼───────────┐
                                          │  Streamlit UI        │
                                          └─────────────────────┘
```

---

## 5. Fonksiyonel Gereksinimler

### 5.1 PDF Yükleme ve İşleme

| ID | Gereksinim | Öncelik |
|----|-----------|---------|
| FR-01 | Kullanıcı tek veya birden fazla PDF yükleyebilmeli | P0 |
| FR-02 | PDF'den metin doğru şekilde extract edilmeli (Türkçe karakter desteği dahil) | P0 |
| FR-03 | Sayfa numaraları ve başlıklar metadata olarak saklanmalı | P1 |
| FR-04 | Desteklenen max dosya boyutu: 50MB | P1 |

### 5.2 Chunking

| ID | Gereksinim | Öncelik |
|----|-----------|---------|
| FR-05 | Metin chunk_size=1000 karakter ile parçalanmalı (konfigüre edilebilir) | P0 |
| FR-06 | Chunk'lar arası overlap=200 karakter olmalı (konfigüre edilebilir) | P0 |
| FR-07 | Chunk'lar cümle ortasından kesilmemeli (sentence-aware splitting) | P0 |
| FR-08 | Her chunk, kaynak sayfa numarası metadata'sını taşımalı | P1 |

### 5.3 Embedding & Vektör Depolama

| ID | Gereksinim | Öncelik |
|----|-----------|---------|
| FR-09 | Chunk'lar Ollama embedding modeli ile vektöre dönüştürülmeli | P0 |
| FR-10 | Vektörler ChromaDB'de persist edilmeli (yeniden başlatmada kaybolmamalı) | P0 |
| FR-11 | Aynı PDF tekrar yüklendiğinde duplikasyon önlenmeli | P1 |
| FR-12 | Collection bazlı organizasyon (her PDF = 1 collection) | P1 |

### 5.4 Quiz Üretimi

| ID | Gereksinim | Öncelik |
|----|-----------|---------|
| FR-13 | Çoktan seçmeli (4 şık, 1 doğru) quiz soruları üretilmeli | P0 |
| FR-14 | Doğru/Yanlış soruları üretilmeli | P1 |
| FR-15 | Açık uçlu sorular üretilmeli | P1 |
| FR-16 | Kullanıcı soru sayısını belirleyebilmeli (varsayılan: 5) | P0 |
| FR-17 | Kullanıcı zorluk seviyesi seçebilmeli (kolay / orta / zor) | P1 |
| FR-18 | Her sorunun hangi chunk'tan üretildiği referans gösterilmeli | P1 |
| FR-19 | Sorular JSON formatında structured output olarak dönmeli | P0 |

### 5.5 Kullanıcı Arayüzü (Streamlit)

| ID | Gereksinim | Öncelik |
|----|-----------|---------|
| FR-20 | PDF yükleme arayüzü (drag & drop) | P0 |
| FR-21 | İşleme durumu progress bar ile gösterilmeli | P1 |
| FR-22 | Quiz interaktif olarak cevaplanabilmeli | P0 |
| FR-23 | Quiz sonunda skor gösterilmeli | P0 |
| FR-24 | Yanlış cevapların doğru cevapları ve açıklamaları gösterilmeli | P0 |
| FR-25 | Yüklenen PDF'lerin listesi sidebar'da gösterilmeli | P1 |

---

## 6. Non-Fonksiyonel Gereksinimler

| ID | Gereksinim | Detay |
|----|-----------|-------|
| NFR-01 | **Gizlilik** | Tüm veri lokal kalmalı, dışarıya hiçbir veri gönderilmemeli |
| NFR-02 | **Performans** | 10 sayfalık PDF için quiz üretimi < 60 saniye |
| NFR-03 | **Modülerlik** | Her bileşen bağımsız modül olarak tasarlanmalı |
| NFR-04 | **Konfigürasyon** | chunk_size, overlap, model adı, soru sayısı vb. config dosyasından yönetilmeli |
| NFR-05 | **Hata Yönetimi** | Ollama bağlantısı yoksa anlamlı hata mesajı gösterilmeli |
| NFR-06 | **Loglama** | Tüm işlemler loglanmalı (processing, generation, errors) |

---

## 7. Proje Yapısı

```
context-rag/
├── main.py                  # Uygulama giriş noktası
├── config.py                # Konfigürasyon ayarları
├── requirements.txt         # Python bağımlılıkları
├── PRD.md                   # Bu doküman
├── README.md                # Kurulum ve kullanım rehberi
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py     # PDF yükleme ve metin çıkarma
│   ├── chunker.py           # Overlap ile chunking mantığı
│   ├── embedder.py          # Embedding ve ChromaDB işlemleri
│   ├── quiz_generator.py    # LLM ile quiz üretimi
│   └── prompts.py           # LLM prompt template'leri
│
├── ui/
│   ├── __init__.py
│   └── app.py               # Streamlit arayüzü
│
├── data/
│   ├── uploads/             # Yüklenen PDF'ler
│   └── chroma_db/           # ChromaDB persist dizini
│
└── tests/
    ├── __init__.py
    ├── test_pdf_processor.py
    ├── test_chunker.py
    └── test_quiz_generator.py
```

---

## 8. Chunking Stratejisi (Detay)

Overlap mekanizması, chunk'lar arası bağlamın korunması için kritiktir:

```
Orijinal Metin: [A B C D E F G H I J K L M N O P Q R S T]

chunk_size = 8, overlap = 3

Chunk 1: [A B C D E F G H]
Chunk 2:           [F G H I J K L M]      ← F,G,H overlap
Chunk 3:                     [L M N O P Q R S]  ← L,M overlap
Chunk 4:                               [R S T]  ← R,S overlap
```

**Neden overlap?**
- Bir cümle iki chunk'ın sınırına denk geldiğinde, overlap sayesinde her iki chunk'ta da tam cümle bulunur.
- LLM, soruyu üretirken yeterli bağlama sahip olur.
- Semantic search sırasında daha iyi eşleşme sağlanır.

---

## 9. Quiz Üretim Prompt Stratejisi

### Çoktan Seçmeli Soru Prompt Örneği

```
Aşağıdaki metne dayanarak {difficulty} seviyesinde {count} adet çoktan seçmeli 
soru oluştur. Her sorunun 4 şıkkı olsun ve sadece 1 tanesi doğru olsun.

Metin:
{context}

Yanıtı aşağıdaki JSON formatında ver:
{
  "questions": [
    {
      "question": "Soru metni",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "Doğru cevabın açıklaması"
    }
  ]
}
```

---

## 10. Konfigürasyon Parametreleri

```python
# config.py
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2"                    # veya "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000          # karakter
CHUNK_OVERLAP = 200        # karakter
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

CHROMA_PERSIST_DIR = "./data/chroma_db"
UPLOAD_DIR = "./data/uploads"

DEFAULT_QUIZ_COUNT = 5
DEFAULT_DIFFICULTY = "orta"   # kolay, orta, zor
DEFAULT_QUESTION_TYPE = "multiple_choice"  # multiple_choice, true_false, open_ended

MAX_FILE_SIZE_MB = 50
```

---

## 11. Geliştirme Fazları

### Faz 1 - MVP (Hafta 1-2)
- [x] PRD hazırlığı
- [ ] PDF metin çıkarma modülü
- [ ] Overlap ile chunking modülü
- [ ] ChromaDB entegrasyonu
- [ ] Ollama LLM ile basit quiz üretimi (çoktan seçmeli)
- [ ] Temel Streamlit arayüzü

### Faz 2 - İyileştirme (Hafta 3)
- [ ] Farklı soru tipleri (doğru/yanlış, açık uçlu)
- [ ] Zorluk seviyesi seçimi
- [ ] Skor sistemi ve sonuç ekranı
- [ ] PDF metadata desteği (sayfa no, başlık)
- [ ] Error handling ve loglama

### Faz 3 - Polish (Hafta 4)
- [ ] UI/UX iyileştirmeleri
- [ ] Quiz geçmişi
- [ ] PDF koleksiyon yönetimi
- [ ] Performans optimizasyonu
- [ ] Test yazımı

---

## 12. Ön Koşullar ve Kurulum

### Ollama Kurulumu
```bash
# macOS
brew install ollama

# Model indirme
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Python Ortamı
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 13. Başarı Kriterleri

| Metrik | Hedef |
|--------|-------|
| Quiz sorusu kalitesi | Üretilen soruların %80+'ı anlamlı ve cevaplanabilir |
| Chunk coverage | PDF içeriğinin %95+'ı quiz kapsamında |
| İşleme süresi (10 sayfa) | < 60 saniye |
| Kullanıcı memnuniyeti | Kullanıcı 5 dakikada ilk quiz'ini üretebilmeli |

---

## 14. Riskler ve Mitigasyon

| Risk | Etki | Mitigasyon |
|------|------|-----------|
| Ollama model kalitesi düşük olabilir | Quiz soruları anlamsız olabilir | Farklı modeller denenecek, prompt engineering yapılacak |
| Büyük PDF'lerde bellek sorunu | Uygulama crash olabilir | Streaming processing, chunk limiti |
| Türkçe dil desteği zayıf | Sorular İngilizce üretilebilir | Türkçe'ye güçlü modeller tercih edilecek, prompt'ta dil belirtilecek |
| ChromaDB ölçeklenme | Çok fazla PDF'de yavaşlama | Collection bazlı izolasyon, eski collection silme |

---

*Son güncelleme: 16 Şubat 2026*