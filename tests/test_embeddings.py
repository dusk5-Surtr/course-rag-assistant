from __future__ import annotations

import math

from src.embeddings import HashingEmbeddings


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    embedding = HashingEmbeddings(dimensions=256)
    first = embedding.embed_query("RAG 的索引阶段包括加载、拆分和存储")
    second = embedding.embed_query("RAG 的索引阶段包括加载、拆分和存储")

    assert first == second
    assert len(first) == 256
    assert math.isclose(math.sqrt(sum(value * value for value in first)), 1.0, rel_tol=1e-6)


def test_different_texts_get_different_vectors() -> None:
    embedding = HashingEmbeddings(dimensions=256)
    rag = embedding.embed_query("RAG 使用向量检索")
    agent = embedding.embed_query("Agent 使用工具调用")

    assert rag != agent
