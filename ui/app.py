import streamlit as st
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_processor import extract_text_from_bytes, get_full_text
from src.chunker import chunk_text
from src.contextual_chunker import generate_summary, contextualize_chunks
from src.embedder import store_chunks, get_all_chunks, generate_collection_name
from src.quiz_generator import generate_quiz
from src.dataset_search import (
    search_books,
    get_chunks_for_book,
    get_book_chapters,
    get_cache_status,
    build_index,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Difficulty mapping ──────────────────────────────────────────────────────

DIFFICULTY_OPTIONS = {
    "🟢 Kolay": "kolay",
    "🟡 Orta": "orta",
    "🔴 Zor": "zor",
}

# ─── Session State ──────────────────────────────────────────────────────────


def init_session_state():
    defaults = {
        # Quiz
        "quiz_data": None,
        "quiz_meta": None,       # {id, title, author, chapter, difficulty, source_type}
        "user_answers": {},
        "submitted": False,
        # PDF flow
        "pdf_processed": False,
        "collection_name": None,
        "pdf_filename": None,
        "processing": False,
        # Dataset flow
        "dataset_search_results": [],
        "dataset_query": "",
        "dataset_selected_book": None,  # {book_id, title, author}
        "dataset_chapters": [],         # [{chapter_index, title, progress_percent, char_count}]
        "dataset_no_result": False,
        "active_tab": "dataset",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz():
    st.session_state.quiz_data = None
    st.session_state.quiz_meta = None
    st.session_state.user_answers = {}
    st.session_state.submitted = False


def reset_dataset_state():
    st.session_state.dataset_search_results = []
    st.session_state.dataset_selected_book = None
    st.session_state.dataset_chapters = []
    st.session_state.dataset_no_result = False
    reset_quiz()


# ─── Dataset arama akışı ────────────────────────────────────────────────────


def do_dataset_search(query: str):
    """Yerel SQLite FTS5 veritabanında arama yapar (0ms)."""
    reset_dataset_state()
    st.session_state.processing = True
    results = search_books(query, limit=20)
    if not results:
        st.session_state.dataset_no_result = True
    else:
        st.session_state.dataset_search_results = results
    st.session_state.processing = False


def do_load_chapters(book_id: str, title: str, author: str):
    """Seçilen kitabın bölümlerini yükler."""
    try:
        result = get_book_chapters(book_id)
        st.session_state.dataset_chapters = result.get("chapters", [])
    except Exception as e:
        logger.warning(f"Bölümler yüklenemedi: {e}")
        st.session_state.dataset_chapters = []


def do_generate_quiz_from_book(
    book_id: str,
    title: str,
    author: str,
    num_questions: int,
    difficulty: str,
    chapter_index: int | None = None,
):
    """Seçilen kitaptan (opsiyonel bölüm ile) quiz üretir."""
    reset_quiz()
    st.session_state.processing = True
    st.session_state.dataset_selected_book = {"book_id": book_id, "title": title, "author": author}

    chapter_label = None
    if chapter_index is not None and st.session_state.dataset_chapters:
        chap = next(
            (c for c in st.session_state.dataset_chapters if c["chapter_index"] == chapter_index),
            None,
        )
        if chap:
            chapter_label = chap["title"]

    status_text = f"📖 *{title}*"
    if chapter_label:
        status_text += f" — Bölüm {chapter_index}: {chapter_label}"

    with st.status(f"{status_text} işleniyor...", expanded=True) as status:
        st.write("Kitap metni hazırlanıyor...")
        chunks = get_chunks_for_book(book_id, title, author, chapter_index=chapter_index)
        st.write(f"✅ {len(chunks)} chunk hazırlandı.")
        st.write(f"🧠 {num_questions} soru üretiliyor... (zorluk: **{difficulty}**)")

        quiz_chunks = [
            {
                "text": c["text"],
                "source_pages": c.get("source_pages", []),
                "source_chapter": c.get("source_chapter"),
                "progress_percent": c.get("progress_percent"),
            }
            for c in chunks
        ]
        quiz_data = generate_quiz(quiz_chunks, num_questions=num_questions, difficulty=difficulty)

        st.session_state.quiz_data = quiz_data
        st.session_state.quiz_meta = {
            "title": title,
            "author": author,
            "chapter_index": chapter_index,
            "chapter_label": chapter_label,
            "difficulty": difficulty,
            "source_type": "book",
            "book_id": book_id,
        }
        st.session_state.user_answers = {}
        st.session_state.submitted = False
        status.update(label="Quiz hazır! 🎉", state="complete")

    st.session_state.processing = False


# ─── PDF akışı ─────────────────────────────────────────────────────────────


def process_pdf(uploaded_file, num_questions: int, difficulty: str):
    """PDF'i işleyip quiz üretir."""
    st.session_state.processing = True

    with st.status("📄 PDF işleniyor...", expanded=True) as status:
        st.write("Metin çıkarılıyor...")
        pdf_bytes = uploaded_file.getvalue()
        pages = extract_text_from_bytes(pdf_bytes, uploaded_file.name)
        full_text = get_full_text(pages)

        if not full_text.strip():
            st.error("PDF'den metin çıkarılamadı. Lütfen metin içeren bir PDF yükleyin.")
            st.session_state.processing = False
            return

        st.write(f"{len(pages)} sayfa, {len(full_text)} karakter metin çıkarıldı.")

        st.write("Metin parçalanıyor (overlap ile)...")
        chunks = chunk_text(full_text, pages=pages)
        st.write(f"{len(chunks)} chunk oluşturuldu.")

        st.write("Doküman özeti çıkarılıyor...")
        summary = generate_summary(full_text)
        if summary:
            st.write(f"Özet: _{summary[:200]}{'...' if len(summary) > 200 else ''}_")

        st.write("Chunk'lara bağlam ekleniyor (Contextual Retrieval)...")
        progress_bar = st.progress(0)

        def update_progress(current, total):
            progress_bar.progress(current / total)

        ctx_chunks = contextualize_chunks(chunks, summary, progress_callback=update_progress)
        progress_bar.empty()
        st.write(f"{len(ctx_chunks)} chunk bağlamlandırıldı.")

        st.write("Vektörler oluşturuluyor ve kaydediliyor...")
        collection_name = generate_collection_name(uploaded_file.name)
        store_chunks(ctx_chunks, collection_name)
        st.session_state.collection_name = collection_name
        st.session_state.pdf_filename = uploaded_file.name
        st.session_state.pdf_processed = True

        st.write(f"Quiz soruları üretiliyor... (zorluk: **{difficulty}**)")
        all_chunks = get_all_chunks(collection_name, with_metadata=True)
        quiz_data = generate_quiz(all_chunks, num_questions=num_questions, difficulty=difficulty)

        st.session_state.quiz_data = quiz_data
        st.session_state.quiz_meta = {
            "title": uploaded_file.name,
            "author": None,
            "chapter_index": None,
            "chapter_label": None,
            "difficulty": difficulty,
            "source_type": "pdf",
        }
        st.session_state.user_answers = {}
        st.session_state.submitted = False

        status.update(label="PDF işlendi, quiz hazır! ✅", state="complete")

    st.session_state.processing = False


# ─── Sidebar ───────────────────────────────────────────────────────────────


def render_sidebar() -> tuple[int, str]:
    """Sidebar: quiz ayarları (soru sayısı + zorluk). (num_questions, difficulty) döner."""
    with st.sidebar:
        st.markdown("## ⚙️ Quiz Ayarları")

        num_questions = st.slider(
            "Soru sayısı",
            min_value=1,
            max_value=10,
            value=5,
            help="Üretilecek çoktan seçmeli soru sayısı",
        )

        difficulty_label = st.radio(
            "Zorluk seviyesi",
            options=list(DIFFICULTY_OPTIONS.keys()),
            index=1,  # Orta varsayılan
            help="Soruların zorluğunu belirler",
        )
        difficulty = DIFFICULTY_OPTIONS[difficulty_label]

        st.divider()

        # Aktif quiz bilgisi
        meta = st.session_state.quiz_meta
        if meta and st.session_state.quiz_data:
            if meta["source_type"] == "book":
                st.success(f"📖 **{meta['title']}**\n\n_{meta.get('author', '')}_ ")
                if meta.get("chapter_label"):
                    st.caption(f"Bölüm {meta['chapter_index']}: {meta['chapter_label']}")
                diff_emoji = {"kolay": "🟢", "orta": "🟡", "zor": "🔴"}.get(meta["difficulty"], "")
                st.caption(f"Zorluk: {diff_emoji} {meta['difficulty'].capitalize()}")
            else:
                st.success(f"📄 PDF: **{meta['title']}**")

            if st.button("🔄 Yeni Quiz Üret", use_container_width=True):
                reset_quiz()
                st.rerun()

        st.divider()

        cache = get_cache_status()
        if cache["exists"]:
            st.caption(f"✅ Yerel katalog: {cache['num_books']:,} kitap")
        else:
            st.caption("⚠️ Kitap kataloğu henüz kurulmadı.")

    return num_questions, difficulty


# ─── Quiz render ────────────────────────────────────────────────────────────


def _format_page_ref(pages: list[int]) -> str:
    if not pages:
        return ""
    pages = sorted(set(pages))
    if len(pages) == 1:
        return f"Sayfa {pages[0]}"
    if pages[-1] - pages[0] == len(pages) - 1:
        return f"Sayfa {pages[0]}-{pages[-1]}"
    return f"Sayfa {', '.join(str(p) for p in pages)}"


def render_answer_feedback(question: dict, selected: str | None):
    correct_letter = question["correct_answer"]
    correct_option = next(
        (opt for opt in question["options"] if opt.startswith(f"{correct_letter})")),
        None,
    )
    if selected and selected.startswith(f"{correct_letter})"):
        st.success(f"✅ Doğru! {question['explanation']}")
    else:
        st.error(
            f"❌ Yanlış. Doğru cevap: **{correct_option}**\n\n"
            f"{question['explanation']}"
        )


def render_quiz():
    quiz = st.session_state.quiz_data
    if not quiz:
        return

    meta = st.session_state.quiz_meta or {}

    # ── Quiz başlığı ve meta bilgi ──────────────────────────────────────────
    header_parts = [f"📝 Quiz — {len(quiz)} Soru"]
    if meta.get("title"):
        header_parts = [f"📝 {meta['title']} — {len(quiz)} Soru"]
    st.subheader("".join(header_parts))

    info_cols = st.columns(3)
    with info_cols[0]:
        if meta.get("author"):
            st.caption(f"✍️ {meta['author']}")
    with info_cols[1]:
        if meta.get("chapter_label"):
            st.caption(f"📑 Bölüm {meta['chapter_index']}: {meta['chapter_label']}")
        elif meta.get("chapter_index"):
            st.caption(f"📑 Bölüm {meta['chapter_index']}")
        else:
            st.caption("📚 Tüm kitaptan")
    with info_cols[2]:
        diff_emoji = {"kolay": "🟢", "orta": "🟡", "zor": "🔴"}.get(meta.get("difficulty", ""), "")
        st.caption(f"{diff_emoji} {meta.get('difficulty', '').capitalize()}")

    st.caption("Her soru için bir şık seçin, ardından cevaplarınızı kontrol edin.")
    st.divider()

    # ── Sorular ─────────────────────────────────────────────────────────────
    for i, q in enumerate(quiz):
        source_chapter = q.get("source_chapter")
        progress_pct = q.get("progress_percent")
        source_pages = q.get("source_pages", [])

        # Soru başlığı + kaynak bilgisi
        col_q, col_src = st.columns([5, 1])
        with col_q:
            st.markdown(f"**Soru {i + 1}:** {q['question']}")
        with col_src:
            if progress_pct is not None:
                st.caption(f"📍 %{progress_pct:.0f}")
            elif source_pages:
                st.caption(_format_page_ref(source_pages))

        if source_chapter:
            st.caption(f"📑 _{source_chapter}_")

        selected = st.radio(
            f"Cevabınız (Soru {i + 1})",
            options=q["options"],
            key=f"q_{i}",
            index=None,
            label_visibility="collapsed",
            disabled=st.session_state.submitted,
        )

        if selected:
            st.session_state.user_answers[i] = selected

        if st.session_state.submitted:
            render_answer_feedback(q, selected)

        st.divider()

    # ── Gönder butonu ───────────────────────────────────────────────────────
    if not st.session_state.submitted:
        all_answered = len(st.session_state.user_answers) == len(quiz)
        if st.button(
            "✅ Cevapları Kontrol Et",
            type="primary",
            use_container_width=True,
            disabled=not all_answered,
        ):
            st.session_state.submitted = True
            st.rerun()
        if not all_answered:
            remaining = len(quiz) - len(st.session_state.user_answers)
            st.info(f"⏳ {remaining} soru daha cevaplanmalı.")


def render_score():
    if not st.session_state.submitted or not st.session_state.quiz_data:
        return

    quiz = st.session_state.quiz_data
    correct_count = sum(
        1
        for i, q in enumerate(quiz)
        if (ans := st.session_state.user_answers.get(i, ""))
        and ans.startswith(f"{q['correct_answer']})") 
    )
    total = len(quiz)
    pct = (correct_count / total * 100) if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Doğru", f"{correct_count}/{total}")
    with col2:
        st.metric("Başarı", f"%{pct:.0f}")
    with col3:
        label = "🏆 Harika!" if pct >= 80 else ("👍 İyi" if pct >= 50 else "📚 Tekrar Dene")
        st.metric("Sonuç", label)


# ─── Tab 1: Kitap Dataset'i ─────────────────────────────────────────────────


def render_dataset_tab(num_questions: int, difficulty: str):
    cache = get_cache_status()

    if not cache["exists"]:
        st.warning("⚠️ **Kitap kataloğu henüz yerel diske indirilmedi.**")
        st.markdown(
            "Aramanın **tamamen yerel, çevrimdışı ve anlık** çalışabilmesi için "
            "Project Gutenberg kataloğunun (~20 MB) önceden indirilip kaydedilmesi gerekir."
        )
        col_setup, _ = st.columns([2, 3])
        with col_setup:
            if st.button("📥 Kataloğu İndir ve Kur (~15 saniye)", type="primary", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def _on_progress(current, total, msg="İndiriliyor..."):
                    pct = min(1.0, current / total) if total > 0 else 0.0
                    mb_cur = current / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    progress_bar.progress(pct)
                    status_text.info(f"⏳ {msg} ({mb_cur:.1f} MB / {mb_tot:.1f} MB — %{int(pct*100)})")

                try:
                    total = build_index(progress_callback=_on_progress, force=True)
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ {total:,} kitap başarıyla kuruldu! Yenileniyor...")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    progress_bar.empty()
                    status_text.error(f"❌ Hata: {e}")
        st.caption("💡 Alternatif olarak terminalden `python scripts/download_dataset.py` çalıştırabilirsiniz.")
        return

    st.caption(f"📚 **{cache['num_books']:,} kitap** içeren yerel indeks hazır (anlık arama).")
    st.markdown(
        "> **Örnek aramalar:** `Sherlock Holmes` · `Don Quixote` · `Charles Dickens` · "
        "`Pride and Prejudice` · `Dracula` · `Philosophy`"
    )

    # ── Arama ───────────────────────────────────────────────────────────────
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        query = st.text_input(
            "Konu, kitap adı veya yazar",
            value=st.session_state.dataset_query,
            placeholder="Örn: Sherlock Holmes veya Jane Austen",
            label_visibility="collapsed",
            key="query_input",
        )
    with col_btn:
        search_btn = st.button(
            "🔍 Ara",
            type="primary",
            use_container_width=True,
            disabled=not query.strip() or st.session_state.processing,
        )

    if search_btn and query.strip():
        st.session_state.dataset_query = query.strip()
        try:
            do_dataset_search(query.strip())
            st.rerun()
        except Exception as e:
            st.error(f"Hata: {e}")
            logger.error(f"Dataset arama hatası: {e}", exc_info=True)

    if st.session_state.dataset_no_result:
        st.warning(
            f"⚠️ **'{st.session_state.dataset_query}'** için yerel katalogda sonuç bulunamadı. "
            "**PDF Yükle** sekmesine geçerek kendi dokümanınızla devam edebilirsiniz.",
            icon="📄",
        )

    # ── Kitap listesi ────────────────────────────────────────────────────────
    results = st.session_state.dataset_search_results
    selected_book = st.session_state.dataset_selected_book

    if results and not st.session_state.quiz_data:
        st.markdown(f"### 📖 {len(results)} Kitap Bulundu")

        # Kitap henüz seçilmemişse liste göster
        if not selected_book:
            st.caption("Bir kitap seçin, ardından bölüm ve zorluk ayarlayabilirsiniz.")
            for book in results:
                col_info, col_btn2 = st.columns([5, 1])
                with col_info:
                    subjects = book.get("subjects", "")
                    subject_str = (
                        f" — _{subjects[:60]}..._" if len(subjects) > 60
                        else (f" — _{subjects}_" if subjects else "")
                    )
                    st.markdown(f"**{book['title']}** &nbsp;·&nbsp; _{book['author']}_{subject_str}")
                with col_btn2:
                    if st.button(
                        "Seç",
                        key=f"book_{book['book_id']}",
                        use_container_width=True,
                        disabled=st.session_state.processing,
                    ):
                        st.session_state.dataset_selected_book = {
                            "book_id": book["book_id"],
                            "title": book["title"],
                            "author": book["author"],
                        }
                        do_load_chapters(book["book_id"], book["title"], book["author"])
                        st.rerun()

        # Kitap seçildi → bölüm seçici + quiz üret
        else:
            book = selected_book
            chapters = st.session_state.dataset_chapters

            # Geri butonu
            if st.button("← Başka kitap seç"):
                st.session_state.dataset_selected_book = None
                st.session_state.dataset_chapters = []
                reset_quiz()
                st.rerun()

            st.markdown(f"### 📖 {book['title']}")
            st.caption(f"✍️ {book['author']}  ·  🆔 {book['book_id']}")

            st.divider()

            # ── Bölüm seçici ────────────────────────────────────────────────
            chapter_index = None
            if chapters:
                st.markdown("**📑 Bölüm Seçimi** _(opsiyonel)_")
                chapter_options = {"📚 Tüm Kitap (Dengeli Seçim)": None}
                for ch in chapters:
                    pct_str = f"%{ch['progress_percent']:.0f}" if ch.get("progress_percent") else ""
                    label = f"Bölüm {ch['chapter_index']}: {ch['title']}"
                    if pct_str:
                        label += f"  [{pct_str}]"
                    chapter_options[label] = ch["chapter_index"]

                selected_chapter_label = st.selectbox(
                    "Hangi bölümden soru üretilsin?",
                    options=list(chapter_options.keys()),
                    index=0,
                    label_visibility="visible",
                )
                chapter_index = chapter_options[selected_chapter_label]
            else:
                st.info("Bu kitap için bölüm bilgisi bulunamadı, tüm kitaptan soru üretilecek.")

            st.divider()

            # ── Quiz üret butonu ─────────────────────────────────────────────
            if st.button(
                "🧠 Quiz Üret",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.processing,
            ):
                try:
                    do_generate_quiz_from_book(
                        book["book_id"],
                        book["title"],
                        book["author"],
                        num_questions,
                        difficulty,
                        chapter_index=chapter_index,
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Hata: {e}")
                    logger.error(f"Quiz üretim hatası: {e}", exc_info=True)

    # ── Sonuç ve Quiz ────────────────────────────────────────────────────────
    if st.session_state.submitted:
        render_score()
    if st.session_state.quiz_data and st.session_state.dataset_selected_book:
        render_quiz()
    elif (
        not st.session_state.processing
        and not st.session_state.dataset_no_result
        and not st.session_state.quiz_data
        and not results
    ):
        st.info("Yukarıya bir kitap adı veya yazar yazıp 'Ara' butonuna tıklayın.")


# ─── Tab 2: PDF Yükleme ─────────────────────────────────────────────────────


def render_pdf_tab(num_questions: int, difficulty: str):
    if st.session_state.dataset_no_result:
        st.info(
            f"📌 **'{st.session_state.dataset_query}'** dataset'te bulunamadı. "
            "Kendi PDF'inizi yükleyerek devam edebilirsiniz."
        )

    uploaded_file = st.file_uploader(
        "PDF dosyanızı sürükleyip bırakın",
        type=["pdf"],
        help="Maksimum 50MB boyutunda PDF dosyası yükleyebilirsiniz.",
    )

    generate_btn = st.button(
        "📄 Quiz Oluştur",
        type="primary",
        disabled=uploaded_file is None or st.session_state.processing,
    )

    if st.session_state.pdf_processed and st.session_state.pdf_filename:
        st.success(f"✅ Yüklü: {st.session_state.pdf_filename}")

    if generate_btn and uploaded_file:
        reset_quiz()
        st.session_state.dataset_selected_book = None
        try:
            process_pdf(uploaded_file, num_questions, difficulty)
            st.rerun()
        except Exception as e:
            st.error(f"Hata oluştu: {e}")
            logger.error(f"PDF işleme hatası: {e}", exc_info=True)

    if st.session_state.submitted:
        render_score()
    if st.session_state.quiz_data and not st.session_state.dataset_selected_book:
        render_quiz()
    elif (
        not st.session_state.processing
        and not st.session_state.quiz_data
        and not st.session_state.dataset_no_result
    ):
        st.info("Bir PDF yükleyin ve 'Quiz Oluştur' butonuna tıklayın.")


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Context RAG - Quiz Generator",
        page_icon="📚",
        layout="wide",
    )
    init_session_state()

    st.title("📚 Context RAG — Quiz Generator")
    st.markdown(
        "Kitap dataset'inden **arama yapın** ya da kendi **PDF'inizi yükleyin**, "
        "yapay zeka otomatik olarak çoktan seçmeli quiz soruları üretsin."
    )

    num_questions, difficulty = render_sidebar()

    tab_dataset, tab_pdf = st.tabs(
        ["🔍 Kitap Dataset'i Ara", "📄 PDF Yükle"]
    )

    with tab_dataset:
        render_dataset_tab(num_questions, difficulty)

    with tab_pdf:
        render_pdf_tab(num_questions, difficulty)


if __name__ == "__main__":
    main()
