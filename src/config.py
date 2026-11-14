"""应用配置。

所有敏感配置都从环境变量读取；仓库只提交 ``.env.example``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 依赖缺失时仍允许读取系统环境变量
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(ValueError):
    """配置不合法时抛出。"""


def _env_text(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if value else ""


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = _env_text(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是整数，当前值为 {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} 不能小于 {minimum}，当前值为 {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = _env_text(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 必须是数字，当前值为 {raw!r}") from exc


def _env_path(name: str, default: str) -> Path:
    path = Path(_env_text(name, default)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """应用运行参数。"""

    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float
    llm_max_tokens: int

    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int

    data_dir: Path
    top_k: int
    chunk_size: int
    chunk_overlap: int
    app_title: str

    @property
    def uses_llm(self) -> bool:
        return self.llm_provider != "demo"

    @property
    def mode_label(self) -> str:
        labels = {
            "demo": "本地检索演示",
            "openai_compatible": "在线大模型",
            "ollama": "离线大模型",
        }
        return labels.get(self.llm_provider, self.llm_provider)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读取环境变量并返回不可变配置。"""

    provider = _env_text("LLM_PROVIDER", "demo").lower()
    if provider not in {"demo", "openai_compatible", "ollama"}:
        raise ConfigError(
            "LLM_PROVIDER 仅支持 demo、openai_compatible 或 ollama，"
            f"当前值为 {provider!r}"
        )

    embedding_provider = _env_text("EMBEDDING_PROVIDER", "hashing").lower()
    if embedding_provider not in {"hashing", "huggingface"}:
        raise ConfigError(
            "EMBEDDING_PROVIDER 仅支持 hashing 或 huggingface，"
            f"当前值为 {embedding_provider!r}"
        )

    chunk_size = _env_int("CHUNK_SIZE", 700, minimum=100)
    chunk_overlap = _env_int("CHUNK_OVERLAP", 120, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")

    return Settings(
        llm_provider=provider,
        llm_model=_env_text("LLM_MODEL", "deepseek-chat"),
        llm_base_url=_env_text("LLM_BASE_URL", "https://api.deepseek.com"),
        llm_api_key=_env_text("LLM_API_KEY"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 1024, minimum=64),
        embedding_provider=embedding_provider,
        embedding_model=_env_text("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 1024, minimum=64),
        data_dir=_env_path("DATA_DIR", "data/raw"),
        top_k=_env_int("TOP_K", 5),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        app_title=_env_text("APP_TITLE", "智学助手"),
    )
