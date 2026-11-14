from __future__ import annotations

from src.config import get_settings
from src.ingest import load_corpus
from src.rag import format_context


def test_corpus_loads_sample_materials() -> None:
    settings = get_settings()
    corpus = load_corpus(settings)

    assert corpus.file_count >= 5
    assert corpus.chunk_count >= 10
    assert corpus.raw_characters > 5000
    assert all(chunk.metadata.get("citation") for chunk in corpus.chunks)


def test_context_contains_stable_citation() -> None:
    settings = get_settings()
    corpus = load_corpus(settings)
    context = format_context(corpus.chunks[:2])

    assert "来源：" in context
    assert ".md#" in context
