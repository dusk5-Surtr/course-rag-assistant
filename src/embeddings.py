"""Embedding 实现与工厂。

默认的 HashingEmbeddings 不需要下载模型，适合课堂演示、自动测试和离线环境；
配置 EMBEDDING_PROVIDER=huggingface 后可切换到句向量模型。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.text_utils import tokenize


class HashingEmbeddings(Embeddings):
    """基于带符号特征哈希的轻量文本向量。

    它不是大模型 Embedding，但具备确定性、零下载、可离线运行的特点，
    用于保证应用在没有任何 API Key 或网络时仍能完成检索闭环。
    """

    def __init__(self, dimensions: int = 1024) -> None:
        if dimensions < 64:
            raise ValueError("dimensions 不能小于 64")
        self.dimensions = dimensions

    def _vectorize(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dimensions

        for token, count in counts.items():
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=8
            ).digest()
            hashed = int.from_bytes(digest, "little")
            index = hashed % self.dimensions
            sign = 1.0 if (hashed >> 63) & 1 == 0 else -1.0
            weight = 1.0 + math.log(count)
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def create_embeddings(settings: Settings) -> Embeddings:
    """根据配置创建 Embedding 对象。"""

    if settings.embedding_provider == "hashing":
        return HashingEmbeddings(dimensions=settings.embedding_dimensions)

    if settings.embedding_provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise RuntimeError(
                "使用 HuggingFace Embedding 需要安装可选依赖："
                "pip install -r requirements-local-embedding.txt"
            ) from exc

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(f"未知的 Embedding 提供方：{settings.embedding_provider}")
