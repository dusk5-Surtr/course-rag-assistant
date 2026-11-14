"""LangChain RAG 服务。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore

from src.config import Settings
from src.embeddings import create_embeddings
from src.prompts import RAG_PROMPT
from src.providers import create_chat_model
from src.text_utils import excerpt, lexical_terms, split_sentences


@dataclass(frozen=True)
class SourceItem:
    citation: str
    file_name: str
    page: int | None
    excerpt: str


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: tuple[SourceItem, ...]
    mode: str
    elapsed_ms: int


def _document_source(document: Document) -> SourceItem:
    metadata = document.metadata
    page = metadata.get("page")
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None

    return SourceItem(
        citation=str(metadata.get("citation") or metadata.get("file_name") or "未知来源"),
        file_name=str(metadata.get("file_name") or "未知文件"),
        page=page,
        excerpt=excerpt(document.page_content),
    )


def _source_items(documents: Iterable[Document]) -> tuple[SourceItem, ...]:
    items: list[SourceItem] = []
    seen: set[str] = set()
    for document in documents:
        item = _document_source(document)
        if item.citation not in seen:
            items.append(item)
            seen.add(item.citation)
    return tuple(items)


def format_context(documents: Sequence[Document]) -> str:
    """把检索片段格式化为带稳定来源标识的上下文。"""

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        citation = metadata.get("citation", "未知来源")
        title = metadata.get("title") or metadata.get("file_name") or ""
        page = metadata.get("page")
        page_text = f"｜第 {page} 页" if page is not None else ""
        blocks.append(
            f"【资料 {index}｜来源：{citation}{page_text}｜标题：{title}】\n"
            f"{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def _clean_markdown_line(text: str) -> str:
    cleaned = re.sub(r"\*\*|__", "", text)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" 。；;：:")
    return cleaned[:300].rstrip() + "…" if len(cleaned) > 300 else cleaned


def _section_heading(document: Document) -> str:
    for line in document.page_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


MIN_RETRIEVAL_COVERAGE = 0.15


def _retrieval_coverage(question: str, documents: Sequence[Document]) -> float:
    query_terms = lexical_terms(question)
    if not query_terms or not documents:
        return 0.0
    return max(
        len(query_terms & lexical_terms(document.page_content))
        / len(query_terms)
        for document in documents
    )


def _rerank_documents(
    question: str,
    documents: Sequence[Document],
    top_k: int,
) -> list[Document]:
    """对向量召回候选做轻量混合重排。

    默认 Embedding 零下载、可复现，但纯哈希向量对短标题不够敏感。
    这里叠加正文词项覆盖、章节标题匹配和向量召回顺序：
    既保留向量检索，又确保“ReAct 框架”这类标题能进入上下文。
    """

    query_terms = lexical_terms(question)
    if not query_terms or not documents:
        return list(documents[:top_k])

    total = max(1, len(documents))
    scored: list[tuple[float, Document]] = []
    for rank, document in enumerate(documents):
        content_terms = lexical_terms(document.page_content)
        heading_terms = lexical_terms(_section_heading(document))
        content_hits = query_terms & content_terms
        heading_hits = query_terms & heading_terms

        coverage = len(content_hits) / max(1, len(query_terms))
        vector_rank_score = 1.0 - rank / (total + 1)
        heading_boost = 0.0
        for term in heading_hits:
            if re.fullmatch(r"[a-z0-9_.+-]{2,}", term):
                heading_boost += 0.32
            else:
                heading_boost += 0.06

        score = 0.30 * vector_rank_score + 0.45 * coverage + heading_boost
        scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:top_k]]


def _following_list_items(
    document: Document,
    anchor: str,
    *,
    max_items: int = 4,
) -> list[str]:
    """当命中“包括/如下”等引导句时，补足同一段的列表项。"""

    content = document.page_content
    position = content.find(anchor)
    if position < 0:
        return []

    tail = content[position + len(anchor) :]
    items: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            break
        match = re.match(r"^(?:[-*+]|\d+\.)\s+(.+)$", stripped)
        if match:
            items.append(_clean_markdown_line(match.group(1)))
            if len(items) >= max_items:
                break

    return items


def _candidate_sentences(
    question: str,
    documents: Sequence[Document],
) -> list[tuple[float, int, str, str, Document]]:
    """对检索片段中的句子做轻量词面打分，供无模型演示使用。"""

    query_terms = lexical_terms(question)
    if not query_terms:
        return []

    candidates: list[tuple[float, int, str, str, Document]] = []
    seen_sentences: list[str] = []

    for document_rank, document in enumerate(documents):
        citation = str(document.metadata.get("citation") or "未知来源")
        for sentence in split_sentences(document.page_content):
            sentence_terms = lexical_terms(sentence)
            overlap = query_terms & sentence_terms
            if not overlap:
                continue

            score = len(overlap) / max(1, len(query_terms))
            score += 0.08 / (document_rank + 1)

            # 问题和原句存在较长的完全匹配时适当加权。
            normalized_question = question.strip("？?。.!！ ")
            if len(normalized_question) >= 5 and normalized_question in sentence:
                score += 0.25

            if any(sentence[:28] == seen[:28] for seen in seen_sentences):
                continue
            seen_sentences.append(sentence)
            candidates.append((score, len(overlap), sentence, citation, document))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates


def _demo_answer(question: str, documents: Sequence[Document]) -> tuple[str, str]:
    candidates = _candidate_sentences(question, documents)
    query_terms = lexical_terms(question)

    if not candidates or not query_terms:
        return (
            "知识库中未找到足够依据。请换一种问法，或补充相关课程资料后再试。",
            "insufficient",
        )

    best_score, best_overlap, _, _, _ = candidates[0]
    if best_overlap == 0 or (len(query_terms) >= 4 and best_score < 0.12):
        return (
            "知识库中未找到足够依据。当前资料没有覆盖这个问题，因此不做推测性回答。",
            "insufficient",
        )

    base_selected: list[tuple[float, int, str, str, Document]] = []
    for candidate in candidates:
        sentence = candidate[2]
        if any(
            sentence[:24] == chosen[2][:24]
            or sentence in chosen[2]
            or chosen[2] in sentence
            for chosen in base_selected
        ):
            continue
        base_selected.append(candidate)
        if len(base_selected) >= 3:
            break

    # 演示模式命中“包括/如下”等引导句时，补入紧随其后的列表项，
    # 避免只回答“包括以下步骤”却遗漏真正的步骤。
    selected: list[tuple[float, int, str, str, Document]] = []
    trigger_markers = ("包括", "如下", "以下", "分为", "模式有", "模块有")
    for candidate in base_selected:
        selected.append(candidate)
        _, _, sentence, citation, document = candidate
        if any(marker in sentence for marker in trigger_markers):
            for item in _following_list_items(document, sentence):
                selected.append((0.0, 0, item, citation, document))

    unique_selected: list[tuple[float, int, str, str, Document]] = []
    for candidate in selected:
        sentence = candidate[2]
        if any(
            sentence[:24] == chosen[2][:24]
            or sentence in chosen[2]
            or chosen[2] in sentence
            for chosen in unique_selected
        ):
            continue
        unique_selected.append(candidate)
        if len(unique_selected) >= 8:
            break

    lines = [
        f"- {sentence} [来源：{citation}]"
        for _, _, sentence, citation, _ in unique_selected
    ]
    answer = "根据知识库，可以确认以下信息：\n\n" + "\n".join(lines)
    return answer, "answered"


class RagService:
    """封装检索、提示词、模型调用与来源整理。"""

    def __init__(
        self,
        settings: Settings,
        chunks: Sequence[Document],
        *,
        llm: Any | None = None,
        mode: str | None = None,
    ) -> None:
        if not chunks:
            raise ValueError("知识库为空，无法创建问答服务")

        self.settings = settings
        self.chunks = tuple(chunks)
        self.embedding = create_embeddings(settings)
        self.vector_store = InMemoryVectorStore(embedding=self.embedding)
        self.vector_store.add_documents(list(self.chunks))
        self.candidate_k = min(
            len(self.chunks),
            max(settings.top_k * 4, 12),
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.candidate_k}
        )

        if llm is not None:
            self.llm = llm
            self.mode = mode or "LangChain 测试模型"
        else:
            self.llm = create_chat_model(settings)
            self.mode = settings.mode_label

        self.answer_chain = (
            RAG_PROMPT | self.llm | StrOutputParser()
            if self.llm is not None
            else None
        )

    @property
    def uses_llm(self) -> bool:
        return self.llm is not None

    def retrieve(self, question: str) -> list[Document]:
        candidates = list(self.retriever.invoke(question))
        return _rerank_documents(question, candidates, self.settings.top_k)

    def ask(self, question: str) -> RagAnswer:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("问题不能为空")

        started = time.perf_counter()
        documents = self.retrieve(normalized_question)
        coverage = _retrieval_coverage(normalized_question, documents)
        if not documents:
            answer = "知识库中未找到相关片段。请先补充课程资料或换一种问法。"
        elif coverage < MIN_RETRIEVAL_COVERAGE:
            answer = (
                "知识库中未找到足够依据。当前资料没有覆盖这个问题，"
                "因此不做推测性回答。"
            )
        elif self.answer_chain is None:
            answer, _status = _demo_answer(normalized_question, documents)
        else:
            answer = self.answer_chain.invoke(
                {
                    "context": format_context(documents),
                    "question": normalized_question,
                }
            )
            answer = str(answer).strip()
            # 小参数离线模型偶尔忽略引用格式；若答案没有引用，补充本次检索来源，
            # 保证 UI 与评测始终可以追溯，同时不改写模型正文。
            if (
                answer
                and "[来源：" not in answer
                and "[来源:" not in answer
                and not answer.startswith(("知识库中未找到", "资料不足", "无法"))
            ):
                citations = " ".join(
                    f"[来源：{document.metadata.get('citation', '未知来源')}]"
                    for document in documents
                )
                answer = f"{answer}\n\n参考来源：{citations}"

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RagAnswer(
            question=normalized_question,
            answer=answer,
            sources=_source_items(documents),
            mode=self.mode,
            elapsed_ms=elapsed_ms,
        )
