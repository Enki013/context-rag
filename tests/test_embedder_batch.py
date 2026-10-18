"""
tests/test_embedder_batch.py
----------------------------
Embedding batching mantığı ve Ollama tokenizer EOF korumasını test eder.
"""

from unittest.mock import MagicMock, patch
from src.embedder import _get_embeddings_batch, store_chunks


def test_get_embeddings_batch_empty():
    res = _get_embeddings_batch([])
    assert res == []


def test_get_embeddings_batch_partitioning():
    # 70 metin verildiğinde batch_size=32 ile 3 ayrı API çağrısı yapılmalı (32 + 32 + 6)
    texts = [f"Text {i}" for i in range(70)]

    mock_embeddings = [[0.1 * i, 0.2 * i] for i in range(70)]

    def fake_embed(model, input):
        # input listesine karşılık gelen alt dilimi dön
        return {"embeddings": [[0.5] * 1024 for _ in input]}

    with patch("src.embedder.ollama_client.embed", side_effect=fake_embed) as mock_client:
        progress_calls = []

        def on_progress(cur, total):
            progress_calls.append((cur, total))

        embeddings = _get_embeddings_batch(
            texts, batch_size=32, progress_callback=on_progress
        )

        assert len(embeddings) == 70
        assert mock_client.call_count == 3
        # batch boyutları: 32, 32, 6
        assert len(mock_client.call_args_list[0].kwargs["input"]) == 32
        assert len(mock_client.call_args_list[1].kwargs["input"]) == 32
        assert len(mock_client.call_args_list[2].kwargs["input"]) == 6

        # Progress çağrıları: 32/70, 64/70, 70/70
        assert progress_calls == [(32, 70), (64, 70), (70, 70)]


def test_store_chunks_with_batching(tmp_path):
    chunks = [
        {"index": i, "text": f"Raw {i}", "contextualized_text": f"Ctx {i}", "length": 50}
        for i in range(10)
    ]

    fake_embs = [[0.1] * 128 for _ in range(10)]

    with patch("src.embedder._get_embeddings_batch", return_value=fake_embs) as mock_batch, \
         patch("src.embedder.CHROMA_PERSIST_DIR", str(tmp_path)):
        
        coll = store_chunks(chunks, "test_collection", batch_size=4)
        assert coll == "test_collection"
        assert mock_batch.call_count == 1
        # batch_size=4 argümanı ile çağrıldı mı?
        assert mock_batch.call_args.kwargs["batch_size"] == 4
