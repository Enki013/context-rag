import logging
import ollama as ollama_client

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, CONTEXTUAL_LLM_THRESHOLD

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """Summarize the following document in 3-4 sentences. \
Focus on the main topic, key concepts, and structure. \
Write the summary in the same language as the document.

Document:
{text}

Summary:"""

CONTEXT_PROMPT = """Here is a chunk from a document:
<chunk>
{chunk}
</chunk>

Document summary: {summary}
Chunk position: {position}

Provide a short (1-2 sentence) context that situates this chunk within the overall document. \
Write in the same language as the chunk. Respond ONLY with the context, nothing else."""


def generate_summary(full_text: str) -> str:
    """
    Dokümanın kısa bir özetini LLM ile üretir.
    Tek bir LLM çağrısı yapar.
    """
    max_chars = 6000
    text_for_summary = full_text[:max_chars]

    prompt = SUMMARY_PROMPT.format(text=text_for_summary)

    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        summary = response["message"]["content"].strip()
        logger.info(f"Doküman özeti üretildi ({len(summary)} karakter)")
        return summary
    except Exception as e:
        logger.error(f"Özet üretme hatası: {e}")
        return ""


def _get_neighbor_snippet(chunks: list[dict], index: int, direction: str, chars: int = 200) -> str:
    """Komşu chunk'tan kısa bir snippet alır."""
    if direction == "before" and index > 0:
        return chunks[index - 1]["text"][-chars:]
    elif direction == "after" and index < len(chunks) - 1:
        return chunks[index + 1]["text"][:chars]
    return ""


def _generate_llm_context(chunk_text: str, summary: str, position: str) -> str:
    """LLM ile chunk için bağlam üretir."""
    prompt = CONTEXT_PROMPT.format(
        chunk=chunk_text,
        summary=summary,
        position=position,
    )

    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"LLM bağlam üretme hatası: {e}")
        return ""


def _generate_deterministic_context(
    chunk: dict,
    chunks: list[dict],
    summary: str,
    total_chunks: int,
) -> str:
    """LLM çağrısı yapmadan, komşu chunk'lar ve özetten bağlam oluşturur."""
    idx = chunk["index"]
    position_pct = int((idx / max(total_chunks - 1, 1)) * 100)

    before = _get_neighbor_snippet(chunks, idx, "before")
    after = _get_neighbor_snippet(chunks, idx, "after")

    parts = [f"[Document: {summary}]"]
    parts.append(f"[Position: chunk {idx + 1}/{total_chunks}, {position_pct}% through document]")

    if before:
        parts.append(f"[Preceding text: ...{before}]")
    if after:
        parts.append(f"[Following text: {after}...]")

    return " ".join(parts)


def contextualize_chunks(
    chunks: list[dict],
    summary: str,
    progress_callback=None,
) -> list[dict]:
    """
    Her chunk'a bağlam bilgisi ekler.

    Küçük dokümanlar (< CONTEXTUAL_LLM_THRESHOLD chunk) için LLM ile bağlam üretir.
    Büyük dokümanlar için deterministic bağlam (özet + pozisyon + komşular) kullanır.

    Args:
        chunks: [{"index": 0, "text": "...", "length": 950}, ...]
        summary: Doküman özeti
        progress_callback: İlerleme için callable(current, total)

    Returns:
        [{"index": 0, "text": "raw", "contextualized_text": "context + raw", "length": ...}, ...]
    """
    total = len(chunks)
    use_llm = total <= CONTEXTUAL_LLM_THRESHOLD

    method = "LLM" if use_llm else "deterministic"
    logger.info(f"{total} chunk için bağlam ekleniyor (yöntem: {method})")

    contextualized = []

    for i, chunk in enumerate(chunks):
        position = f"Chunk {chunk['index'] + 1}/{total} ({int((chunk['index'] / max(total - 1, 1)) * 100)}%)"

        if use_llm:
            context = _generate_llm_context(chunk["text"], summary, position)
        else:
            context = _generate_deterministic_context(chunk, chunks, summary, total)

        if context:
            ctx_text = f"{context}\n\n{chunk['text']}"
        else:
            ctx_text = chunk["text"]

        ctx_entry = {
            "index": chunk["index"],
            "text": chunk["text"],
            "contextualized_text": ctx_text,
            "length": chunk["length"],
        }
        if "source_pages" in chunk:
            ctx_entry["source_pages"] = chunk["source_pages"]
        contextualized.append(ctx_entry)

        if progress_callback:
            progress_callback(i + 1, total)

    logger.info(f"{total} chunk bağlamlandırıldı")
    return contextualized
