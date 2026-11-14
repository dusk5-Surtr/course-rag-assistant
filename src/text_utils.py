"""轻量文本处理工具。

中文使用字符二元组作为检索特征，英文和数字使用完整词项。
这样不依赖分词模型即可稳定运行，同时保留较好的中文检索效果。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


LATIN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+.-]*|\d+(?:\.\d+)?")
CHINESE_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "一个",
    "一些",
    "以及",
    "什么",
    "介绍",
    "可以",
    "哪些",
    "如何",
    "怎么",
    "是否",
    "有关",
    "本文",
    "本篇",
    "资料",
    "这个",
    "这些",
    "进行",
    "需要",
    "主要",
    "分别",
    "包括",
    "区别",
    "之间",
    "有哪",
    "中的",
    "是什",
    "么是",
}


def normalize_text(text: str) -> str:
    """统一大小写与空白，便于稳定哈希。"""

    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    """返回用于哈希向量和演示检索的词项列表。"""

    normalized = normalize_text(text)
    tokens: list[str] = LATIN_RE.findall(normalized)

    for run in CHINESE_RUN_RE.findall(normalized):
        # 单字只在文本没有可组合的二元组时保留。
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))

    return [token for token in tokens if token and token not in STOP_TERMS]


def term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def lexical_terms(text: str) -> set[str]:
    """查询与句子打分使用的去重词项。"""

    return set(tokenize(text))


def split_sentences(text: str) -> list[str]:
    """按中文标点和换行切分可引用的短句。"""

    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"[ \t]+", " ", text)

    raw_parts = re.split(r"(?<=[。！？；!?;])\s*|\n+", text)
    sentences: list[str] = []
    for part in raw_parts:
        cleaned = re.sub(r"^[，。；：、\s]+|[，；：、\s]+$", "", part).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        # 过滤“2.1 索引阶段”这类纯编号标题；它们不适合作为回答句。
        if re.match(r"^\d+(?:\.\d+)+\s+\S+$", cleaned):
            continue
        if 12 <= len(cleaned) <= 320:
            sentences.append(cleaned)
    return sentences


def excerpt(text: str, max_chars: int = 190) -> str:
    """生成来源卡片中的短摘录，并去除 Markdown 控制符号。"""

    compact = re.sub(r"^[ \t]*>[ \t]?", "", text, flags=re.MULTILINE)
    compact = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", compact, flags=re.MULTILINE)
    compact = re.sub(r"\*\*|__|`", "", compact)
    compact = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
