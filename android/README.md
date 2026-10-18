# 📱 Context RAG - Android Test Uygulaması (Kotlin + Jetpack Compose Material 3)

Bu uygulama, **Context-RAG Headless Quiz & Soru Bankası** mikroservisini test etmek amacıyla saf **Kotlin** ve **%100 Material Design 3 (M3 / Material You)** bileşenleri kullanılarak geliştirilmiştir.

---

## 🎨 Kullanılan Material 3 (M3) Bileşenleri

- **Scaffold & App Bars:** `Scaffold`, `CenterAlignedTopAppBar`, `TopAppBarDefaults`
- **Navigasyon:** `NavigationBar`, `NavigationBarItem`, `BadgedBox`, `Badge`
- **Kartlar & Konteynerlar:** `ElevatedCard`, `OutlinedCard`, `Card`, `Surface`
- **Etkileşimli Butonlar:** `Button` (Filled), `FilledTonalButton`, `OutlinedButton`, `IconButton`, `FilledTonalIconButton`
- **Chips (Etiketler):** `FilterChip` (Zorluk ve soru sayısı filtreleri), `AssistChip` (Havuz sayaçları), `SuggestionChip` (0ms Cache Hit yeşil rozeti)
- **Giriş Bileşenleri:** `OutlinedTextField`, `RadioButton`
- **Geri Bildirim & İlerleme:** `LinearProgressIndicator` (Bölüm okuma yüzdesi ve quiz adımları), `CircularProgressIndicator` (Asenkron yüklemeler), `SnackbarHost`
- **Liste Yapısı:** `ListItem` (M3 standart iki ve üç satırlı liste elemanları)
- **Dinamik Renkler (Material You):** Android 12+ (API 31+) cihazlarda kullanıcının duvar kağıdına göre dinamik renk paleti (`dynamicLightColorScheme` / `dynamicDarkColorScheme`), eski cihazlar için özel M3 uyumlu mavi/yeşil palet fallback'i.

---

## 🚀 Çalıştırma ve Kurulum

### 1. Backend Servisini Başlatın
Android uygulamasının bağlanabilmesi için yerel FastAPI backend'inizi çalıştırın:
```bash
# Terminal 1: FastAPI sunucusu
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Android Studio ile Projeyi Açın
1. **Android Studio**'yu açın.
2. **Open** diyerek bu projenin altındaki `android/` dizinini seçin.
3. Gradle senkronizasyonunun tamamlanmasını bekleyin.
4. Bir Android Emülatörü (Pixel 8 / API 34 önerilir) veya USB ile bağlı gerçek bir cihaz seçin.
5. **Run ('app')** butonuna tıklayın (Shift + F10).

### 3. Ağ Yapılandırması (Sunucu Sekmesi)
- **Android Emülatör:** Varsayılan olarak `http://10.0.2.2:8000` adresine bağlanır (Android emülatörün bilgisayarınızın `localhost`una erişim adresidir).
- **Gerçek Cihaz (Wi-Fi):** Uygulamanın altındaki **Sunucu** sekmesine geçerek Base URL alanına bilgisayarınızın yerel Wi-Fi IP'sini girin (örn: `http://192.168.1.45:8000`) ve **"Bağlan ve Test Et"** butonuna dokunun.

---

## 📱 Ekranlar ve Test Akışı

1. **Kitaplar (BooksScreen):**
   - 79,000+ kitaplık Gutenberg kataloğunda arama yapın.
   - Bir kitaba dokunduğunuzda bölümleri, okuma ilerleme yüzdesini (%0 - %100) ve her bölümün soru havuzundaki mevcut soru sayısını görün.
   - Zorluk (Kolay/Orta/Zor) ve soru sayısını seçip **"Bölüm Quizini Başlat"** butonuna basın.

2. **Quiz (QuizScreen):**
   - Sorular havuzdan gelmişse üstte yeşil **`⚡ 0ms Soru Bankası (Cache Hit)`** rozeti görünür.
   - Hile koruması devrededir; doğru şıklar cihaza sızdırılmaz.
   - Çoktan seçmeli şıkları işaretleyin ve sonraki sorulara geçin.
   - Son soruda **"Testi Bitir"** butonuna dokunarak cevaplarınızı sunucuya gönderin.

3. **Sonuçlar (ResultScreen):**
   - Sunucudan dönen başarı rozeti (`Harika! 🏆`, `Tekrar Dene! 📚`) ve yüzdelik puanı inceleyin.
   - Soru bazlı analizde kullanıcının cevabı ile doğru cevap karşılaştırmasını ve sunucunun ürettiği ayrıntılı açıklamayı okuyun.

4. **Sunucu (SettingsScreen):**
   - API bağlantı adresini değiştirin, Ollama LLM (`qwen2.5:14b`) ve `bge-m3` model bağlantısını canlı olarak denetleyin.
