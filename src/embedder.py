import json
import logging
import hashlib
import os
import numpy as np
import ollama as ollama_client

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def _get_embedding(text: str) -> list[float]:
    """Ollama embedding modeli ile tek bir metin için vektör üretir."""
    response = ollama_client.embed(model=EMBEDDING_MODEL, input=text)
    return response["embeddings"][0]


def _get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Ollama embedding modeli ile toplu vektör üretir."""
    response = ollama_client.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki vektör arasındaki cosine similarity hesaplar."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _get_store_path(collection_name: str) -> str:
    """Collection için dosya yolunu döner."""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return os.path.join(CHROMA_PERSIST_DIR, f"{collection_name}.json")


def generate_collection_name(filename: str) -> str:
    """Dosya adından güvenli bir collection adı üretir."""
    safe = filename.replace(" ", "_").replace(".", "_")
    safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-"))
    if not safe:
        safe = "collection"
    short_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"{safe[:50]}_{short_hash}"


def store_chunks(chunks: list[dict], collection_name: str) -> str:
    """
    Chunk'ları embedding ile vektörleştirip JSON dosyasına kaydeder.
    Contextual chunk'lar varsa contextualized_text ile embedding yapar,
    quiz üretimi için ham text'i ayrıca saklar.

    Args:
        chunks: [{"index": 0, "text": "...", "contextualized_text": "...", "length": 950}, ...]
               contextualized_text opsiyoneldir, yoksa text kullanılır.
        collection_name: Collection adı

    Returns:
        Kullanılan collection adı
    """
    embed_texts = []
    for chunk in chunks:
        embed_texts.append(chunk.get("contextualized_text", chunk["text"]))

    logger.info(f"{len(embed_texts)} chunk için embedding üretiliyor...")
    embeddings = _get_embeddings_batch(embed_texts)

    store_data = {
        "collection_name": collection_name,
        "chunks": [],
    }

    for chunk, embedding in zip(chunks, embeddings):
        entry = {
            "index": chunk["index"],
            "text": chunk["text"],
            "length": chunk["length"],
            "embedding": embedding,
        }
        if "source_pages" in chunk:
            entry["source_pages"] = chunk["source_pages"]
        store_data["chunks"].append(entry)

    store_path = _get_store_path(collection_name)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store_data, f, ensure_ascii=False)

    logger.info(f"{len(embed_texts)} chunk kaydedildi: {store_path}")
    return collection_name


def _load_store(collection_name: str) -> dict:
    """Collection verisini diskten yükler."""
    store_path = _get_store_path(collection_name)
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Collection bulunamadı: {collection_name}")

    with open(store_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_chunks(collection_name: str, with_metadata: bool = False) -> list:
    """
    Bir collection'daki tüm ham chunk metinlerini sıralı getirir.

    Args:
        collection_name: Collection adı
        with_metadata: True ise chunk dict'leri döner (text + source_pages)

    Returns:
        with_metadata=False: ["chunk text", ...]
        with_metadata=True: [{"text": "...", "source_pages": [3, 4]}, ...]
    """
    data = _load_store(collection_name)
    sorted_chunks = sorted(data["chunks"], key=lambda x: x["index"])

    if with_metadata:
        return [
            {"text": c["text"], "source_pages": c.get("source_pages", [])}
            for c in sorted_chunks
        ]

    return [c["text"] for c in sorted_chunks]


def retrieve_relevant_chunks(
    query: str,
    collection_name: str,
    n_results: int = 5
) -> list[str]:
    """
    Verilen sorguya en yakın chunk'ları semantic search ile getirir.

    Args:
        query: Arama sorgusu
        collection_name: Collection adı
        n_results: Döndürülecek chunk sayısı

    Returns:
        En ilgili ham chunk metinleri listesi
    """
    data = _load_store(collection_name)
    query_embedding = np.array(_get_embedding(query))

    scored = []
    for chunk in data["chunks"]:
        chunk_embedding = np.array(chunk["embedding"])
        similarity = _cosine_similarity(query_embedding, chunk_embedding)
        scored.append((similarity, chunk["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [text for _, text in scored[:n_results]]
