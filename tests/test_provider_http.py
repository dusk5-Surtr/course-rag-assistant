from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.config import get_settings
from src.ingest import load_corpus
from src.providers import ProviderConfigError, create_chat_model
from src.rag import RagService


class _MockChatHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server 约定
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)
        response = {
            "id": "chatcmpl-local-test",
            "object": "chat.completion",
            "created": 1,
            "model": payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "本地兼容接口测试回答。[来源：03_RAG原理与实现.md#002]",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 12,
                "total_tokens": 32,
            },
        }
        data = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_openai_compatible_mode_calls_langchain_chat_model() -> None:
    _MockChatHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        settings = replace(
            get_settings(),
            llm_provider="openai_compatible",
            llm_model="mock-course-model",
            llm_base_url=f"http://127.0.0.1:{server.server_port}",
            llm_api_key="test-key-not-real",
            llm_temperature=0.0,
            llm_max_tokens=128,
        )
        corpus = load_corpus(settings)
        service = RagService(settings, corpus.chunks)
        result = service.ask("RAG 的索引阶段包括哪些步骤？")

        assert result.mode == "在线大模型"
        assert "本地兼容接口测试回答" in result.answer
        assert result.sources
        assert _MockChatHandler.requests
        assert _MockChatHandler.requests[0]["model"] == "mock-course-model"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _MockOllamaHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server 约定
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(payload)
        response = {
            "model": payload.get("model", "mock-ollama"),
            "created_at": "2026-11-14T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Ollama 本地兼容接口测试回答。[来源：03_RAG原理与实现.md#002]",
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 100,
            "load_duration": 10,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 20,
            "eval_count": 10,
            "eval_duration": 30,
        }
        data = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_ollama_mode_calls_langchain_chat_model() -> None:
    _MockOllamaHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        settings = replace(
            get_settings(),
            llm_provider="ollama",
            llm_model="mock-local-model",
            llm_base_url=f"http://127.0.0.1:{server.server_port}",
            llm_temperature=0.0,
            llm_max_tokens=128,
        )
        corpus = load_corpus(settings)
        service = RagService(settings, corpus.chunks)
        result = service.ask("RAG 的索引阶段包括哪些步骤？")

        assert result.mode == "离线大模型"
        assert "Ollama 本地兼容接口测试回答" in result.answer
        assert result.sources
        assert _MockOllamaHandler.requests
        assert _MockOllamaHandler.requests[0]["model"] == "mock-local-model"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_online_mode_without_key_raises_clear_error() -> None:
    settings = replace(
        get_settings(),
        llm_provider="openai_compatible",
        llm_api_key="",
    )

    with pytest.raises(ProviderConfigError, match="LLM_API_KEY"):
        create_chat_model(settings)
