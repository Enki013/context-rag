import streamlit as st
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_processor import extract_text_from_bytes, get_full_text
from src.chunker import chunk_text
from src.contextual_chunker import generate_summary, contextualize_chunks
from src.embedder import store_chunks, get_all_chunks, generate_collection_name
from src.quiz_generator import generate_quiz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state():
    """Session state varsayılan değerlerini ayarlar."""
    defaults = {
        "quiz_data": None,
        "user_answers": {},
        "submitted": False,
        "pdf_processed": False,
        "collection_name": None,
        "pdf_filename": None,
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_quiz():
    """Quiz state'ini sıfırlar."""
    st.session_state.quiz_data = None
    st.session_state.user_answers = {}
    st.session_state.submitted = False


def process_pdf(uploaded_file, num_questions: int):
    """PDF'i işleyip quiz üretir."""
    st.session_state.processing = True

    with st.status("PDF işleniyor...", expanded=True) as status:
        # 1. Metin çıkarma
        st.write("Metin çıkarılıyor...")
        pdf_bytes = uploaded_file.getvalue()
        pages = extract_text_from_bytes(pdf_bytes, uploaded_file.name)
        full_text = get_full_text(pages)

        if not full_text.strip():
            st.error("PDF'den metin çıkarılamadı. Lütfen metin içeren bir PDF yükleyin.")
            st.session_state.processing = False
            return

        st.write(f"{len(pages)} sayfa, {len(full_text)} karakter metin çıkarıldı.")

        # 2. Chunking
        st.write("Metin parçalanıyor (overlap ile)...")
        chunks = chunk_text(full_text, pages=pages)
        st.write(f"{len(chunks)} chunk oluşturuldu.")

        # 3. Contextual Retrieval -- doküman özeti ve chunk bağlamı
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

        # 4. Embedding
        st.write("Vektörler oluşturuluyor ve kaydediliyor...")
        collection_name = generate_collection_name(uploaded_file.name)
        store_chunks(ctx_chunks, collection_name)
        st.session_state.collection_name = collection_name
        st.session_state.pdf_filename = uploaded_file.name

        # 5. Quiz üretimi
        st.write("Quiz soruları üretiliyor (bu biraz sürebilir)...")
        all_chunks = get_all_chunks(collection_name, with_metadata=True)
        quiz_data = generate_quiz(all_chunks, num_questions=num_questions)

        st.session_state.quiz_data = quiz_data
        st.session_state.pdf_processed = True
        st.session_state.user_answers = {}
        st.session_state.submitted = False

        status.update(label="PDF işlendi, quiz hazır!", state="complete")

    st.session_state.processing = False


def render_sidebar():
    """Sidebar: PDF yükleme ve ayarlar."""
    with st.sidebar:
        st.header("PDF Yükle")

        uploaded_file = st.file_uploader(
            "PDF dosyanızı sürükleyip bırakın",
            type=["pdf"],
            help="Maksimum 50MB boyutunda PDF dosyası yükleyebilirsiniz.",
        )

        st.divider()
        st.header("Quiz Ayarları")

        num_questions = st.slider(
            "Soru sayısı",
            min_value=1,
            max_value=10,
            value=5,
            help="Üretilecek çoktan seçmeli soru sayısı",
        )

        generate_btn = st.button(
            "Quiz Oluştur",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None or st.session_state.processing,
        )

        if st.session_state.pdf_processed:
            st.divider()
            st.success(f"Yüklü: {st.session_state.pdf_filename}")
            if st.button("Yeni Quiz Üret", use_container_width=True):
                reset_quiz()
                st.rerun()

        return uploaded_file, num_questions, generate_btn


def _format_page_ref(pages: list[int]) -> str:
    """Sayfa numaralarını okunabilir formata çevirir. ör: [3,4,5] -> 'Sayfa 3-5'"""
    if not pages:
        return ""
    pages = sorted(set(pages))
    if len(pages) == 1:
        return f"Kaynak: Sayfa {pages[0]}"
    if pages[-1] - pages[0] == len(pages) - 1:
        return f"Kaynak: Sayfa {pages[0]}-{pages[-1]}"
    return f"Kaynak: Sayfa {', '.join(str(p) for p in pages)}"


def render_quiz():
    """Quiz sorularını gösterir."""
    quiz = st.session_state.quiz_data

    if not quiz:
        return

    st.subheader(f"Quiz ({len(quiz)} Soru)")
    st.caption("Her soru için bir şık seçin, ardından cevaplarınızı kontrol edin.")

    for i, q in enumerate(quiz):
        source_pages = q.get("source_pages", [])
        if source_pages:
            page_label = _format_page_ref(source_pages)
            st.markdown(f"**Soru {i + 1}:** {q['question']}  \n`{page_label}`")
        else:
            st.markdown(f"**Soru {i + 1}:** {q['question']}")

        options = q["options"]
        key = f"q_{i}"

        selected = st.radio(
            f"Cevabınız (Soru {i + 1})",
            options=options,
            key=key,
            index=None,
            label_visibility="collapsed",
            disabled=st.session_state.submitted,
        )

        if selected:
            st.session_state.user_answers[i] = selected

        if st.session_state.submitted:
            render_answer_feedback(i, q, selected)

        st.divider()

    if not st.session_state.submitted:
        all_answered = len(st.session_state.user_answers) == len(quiz)

        if st.button(
            "Cevapları Kontrol Et",
            type="primary",
            use_container_width=True,
            disabled=not all_answered,
        ):
            st.session_state.submitted = True
            st.rerun()

        if not all_answered:
            remaining = len(quiz) - len(st.session_state.user_answers)
            st.info(f"{remaining} soru daha cevaplanmalı.")


def render_answer_feedback(index: int, question: dict, selected: str | None):
    """Bir sorunun cevap geri bildirimini gösterir."""
    correct_letter = question["correct_answer"]
    correct_option = None
    for opt in question["options"]:
        if opt.startswith(f"{correct_letter})"):
            correct_option = opt
            break

    if selected and selected.startswith(f"{correct_letter})"):
        st.success(f"Doğru! {question['explanation']}")
    else:
        st.error(
            f"Yanlış. Doğru cevap: **{correct_option}**\n\n"
            f"{question['explanation']}"
        )


def render_score():
    """Quiz tamamlandığında skor gösterir."""
    if not st.session_state.submitted or not st.session_state.quiz_data:
        return

    quiz = st.session_state.quiz_data
    correct_count = 0

    for i, q in enumerate(quiz):
        selected = st.session_state.user_answers.get(i, "")
        correct_letter = q["correct_answer"]
        if selected and selected.startswith(f"{correct_letter})"):
            correct_count += 1

    total = len(quiz)
    percentage = (correct_count / total * 100) if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Doğru", f"{correct_count}/{total}")
    with col2:
        st.metric("Başarı", f"%{percentage:.0f}")
    with col3:
        if percentage >= 80:
            st.metric("Sonuç", "Harika!")
        elif percentage >= 50:
            st.metric("Sonuç", "İyi")
        else:
            st.metric("Sonuç", "Tekrar Dene")


def main():
    st.set_page_config(
        page_title="Context RAG - Quiz Generator",
        page_icon="📚",
        layout="wide",
    )

    init_session_state()

    st.title("Context RAG - PDF Quiz Generator")
    st.markdown(
        "PDF dokümanınızı yükleyin, yapay zeka otomatik olarak "
        "çoktan seçmeli quiz soruları üretsin."
    )

    uploaded_file, num_questions, generate_btn = render_sidebar()

    if generate_btn and uploaded_file:
        reset_quiz()
        try:
            process_pdf(uploaded_file, num_questions)
            st.rerun()
        except Exception as e:
            st.error(f"Hata oluştu: {e}")
            logger.error(f"İşleme hatası: {e}", exc_info=True)

    if st.session_state.submitted:
        render_score()

    if st.session_state.quiz_data:
        render_quiz()
    elif not st.session_state.processing:
        st.info(
            "Başlamak için sol panelden bir PDF yükleyin ve "
            "'Quiz Oluştur' butonuna tıklayın."
        )


if __name__ == "__main__":
    main()
