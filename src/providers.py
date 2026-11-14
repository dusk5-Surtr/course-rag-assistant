"""大语言模型提供方工厂。

在线和离线集成都在函数内部延迟导入，因此演示模式不需要安装额外 SDK。
"""

from __future__ import annotations

from typing import Any

from src.config import Settings


class ProviderConfigError(RuntimeError):
    """模型提供方配置不完整。"""


def create_chat_model(settings: Settings) -> Any | None:
    """创建 LangChain ChatModel。

    ``demo`` 返回 ``None``，由应用切换到可解释的本地检索演示回复。
    """

    if settings.llm_provider == "demo":
        return None

    if settings.llm_provider == "openai_compatible":
        if not settings.llm_api_key:
            raise ProviderConfigError(
                "在线模式需要在 .env 中设置 LLM_API_KEY；"
                "你也可以把 LLM_PROVIDER 改为 demo 先体验应用。"
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise ProviderConfigError(
                "缺少 langchain-openai，请先执行 pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "api_key": settings.llm_api_key,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        return ChatOpenAI(**kwargs)

    if settings.llm_provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise ProviderConfigError(
                "缺少 langchain-ollama，请先执行 pip install -r requirements.txt"
            ) from exc

        kwargs = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "num_predict": settings.llm_max_tokens,
        }
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        return ChatOllama(**kwargs)

    raise ProviderConfigError(f"不支持的 LLM_PROVIDER：{settings.llm_provider}")
