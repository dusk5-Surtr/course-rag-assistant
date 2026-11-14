from __future__ import annotations

from dataclasses import replace

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config import get_settings
from src.ingest import load_corpus
from src.rag import RagService


def _demo_service() -> RagService:
    settings = replace(get_settings(), llm_provider="demo", top_k=5)
    corpus = load_corpus(settings)
    return RagService(settings, corpus.chunks)


def test_retrieval_returns_course_sources() -> None:
    service = _demo_service()
    documents = service.retrieve("RAG 的索引阶段包括哪些步骤？")

    assert documents
    assert any("RAG" in str(document.metadata.get("file_name")) for document in documents)


def test_demo_mode_answers_with_citations() -> None:
    service = _demo_service()
    result = service.ask("RAG 的索引阶段包括哪些步骤？")

    assert "索引" in result.answer
    assert "[来源：" in result.answer
    assert result.sources
    assert result.mode == "本地检索演示"


def test_demo_mode_refuses_uncovered_question() -> None:
    service = _demo_service()
    result = service.ask("火星探测器今天遇到了什么天气？")

    assert "未找到" in result.answer or "资料不足" in result.answer


def test_llm_chain_runs_with_fake_chat_model() -> None:
    settings = replace(get_settings(), llm_provider="demo", top_k=5)
    corpus = load_corpus(settings)
    llm = FakeListChatModel(responses=["这是测试回答。[来源：03_RAG原理与实现.md#001]"])
    service = RagService(settings, corpus.chunks, llm=llm)

    result = service.ask("RAG 的索引阶段包括哪些步骤？")

    assert result.answer == "这是测试回答。[来源：03_RAG原理与实现.md#001]"
    assert service.uses_llm is True
    assert result.sources
