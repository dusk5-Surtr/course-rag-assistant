"""智学助手 Streamlit 应用入口。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import streamlit as st

from src.config import ConfigError, Settings, get_settings
from src.ingest import Corpus, IngestError, load_corpus
from src.providers import ProviderConfigError
from src.rag import RagAnswer, RagService


st.set_page_config(
    page_title="智学助手 | LangChain RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXAMPLES = [
    "在线调用和离线调用分别适合什么场景？",
    "LangChain 的核心模块有哪些？",
    "RAG 的索引阶段包括哪些步骤？",
    "如何降低 RAG 回答中的幻觉？",
    "ReAct Agent 的工作闭环是什么？",
    "AutoGen 有哪几种典型协作模式？",
    "作业需要提交哪些内容？",
]


CSS = """
<style>
:root {
    --ink: #18302d;
    --muted: #657772;
    --pine: #0d4f46;
    --mint: #dff3e9;
    --paper: #f6f3ea;
    --card: rgba(255, 255, 255, 0.78);
    --line: rgba(24, 48, 45, 0.12);
    --amber: #b76e2b;
}
.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(196, 228, 205, 0.75), transparent 34rem),
        radial-gradient(circle at 0% 82%, rgba(238, 211, 168, 0.4), transparent 30rem),
        var(--paper);
    color: var(--ink);
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stAppDeployButton"] { display: none !important; }
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b3c36 0%, #102c2a 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #eef9f3; }
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small { color: #b6cec7 !important; }
section[data-testid="stSidebar"] .mode-pill { color: var(--pine) !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.16); }
section[data-testid="stSidebar"] button {
    border-color: rgba(255,255,255,0.24) !important;
    background: rgba(255,255,255,0.06) !important;
    color: #f4fbf7 !important;
}
section[data-testid="stSidebar"] button:hover {
    border-color: #a8d8c8 !important;
    background: rgba(255,255,255,0.11) !important;
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.3rem 2.5rem 2.1rem;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 26px;
    background:
        linear-gradient(122deg, rgba(8, 67, 59, 0.98), rgba(18, 91, 75, 0.93)),
        #0d4f46;
    box-shadow: 0 22px 60px rgba(30, 70, 59, 0.2);
}
.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -70px;
    top: -100px;
    border-radius: 50%;
    background: rgba(213, 240, 221, 0.16);
    box-shadow: 0 0 0 30px rgba(213, 240, 221, 0.06);
}
.eyebrow {
    position: relative;
    z-index: 1;
    font-size: 0.76rem;
    letter-spacing: 0.24em;
    font-weight: 700;
    color: #b8e2d1;
    text-transform: uppercase;
}
.hero h1 {
    position: relative;
    z-index: 1;
    margin: 0.36rem 0 0.4rem;
    color: #fffdf5;
    font-size: clamp(2.2rem, 5vw, 4.1rem);
    line-height: 1.05;
    letter-spacing: -0.04em;
    font-weight: 800;
}
.hero p {
    position: relative;
    z-index: 1;
    max-width: 720px;
    margin: 0;
    color: #dcefe6;
    font-size: 1rem;
    line-height: 1.8;
}
.hero-tags { position: relative; z-index: 1; margin-top: 1.15rem; }
.tag {
    display: inline-block;
    margin: 0.2rem 0.4rem 0.2rem 0;
    padding: 0.32rem 0.68rem;
    border: 1px solid rgba(226, 248, 237, 0.24);
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    color: #e7f8f0;
    font-size: 0.76rem;
}
.stat-card {
    margin-top: 0.9rem;
    padding: 1rem 1.1rem;
    border: 1px solid var(--line);
    border-radius: 17px;
    background: var(--card);
    box-shadow: 0 10px 30px rgba(44, 67, 59, 0.05);
}
.stat-value { color: var(--pine); font-size: 1.55rem; font-weight: 800; line-height: 1.2; }
.stat-label { color: var(--muted); font-size: 0.78rem; margin-top: 0.25rem; }
.section-kicker {
    margin: 1.8rem 0 0.15rem;
    color: var(--amber);
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.section-title {
    margin: 0 0 0.7rem;
    color: var(--ink);
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.mode-pill {
    display: inline-block;
    padding: 0.28rem 0.66rem;
    border-radius: 999px;
    background: var(--mint);
    color: var(--pine);
    font-size: 0.78rem;
    font-weight: 700;
}
.answer-meta {
    margin-top: 0.2rem;
    color: var(--muted);
    font-size: 0.78rem;
}
.source-card {
    margin: 0.55rem 0;
    padding: 0.8rem 0.9rem;
    border-left: 3px solid #8ecab5;
    border-radius: 0 10px 10px 0;
    background: rgba(230, 244, 237, 0.72);
}
.source-title { color: var(--pine); font-weight: 700; font-size: 0.86rem; }
.source-excerpt { color: #52645f; font-size: 0.84rem; line-height: 1.7; margin-top: 0.28rem; }
div[data-testid="stForm"] {
    border: 1px solid var(--line);
    border-radius: 18px;
    background: var(--card);
    padding: 1rem 1rem 0.55rem;
    box-shadow: 0 10px 30px rgba(44, 67, 59, 0.05);
}
div[data-testid="stFormSubmitButton"] button, .stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(13, 79, 70, 0.2);
    background: rgba(255,255,255,0.64);
    color: var(--pine);
    font-weight: 650;
}
div[data-testid="stFormSubmitButton"] button {
    background: var(--pine);
    color: #fffdf6;
    border-color: var(--pine);
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: #0a6b5b;
    color: #ffffff;
    border-color: #0a6b5b;
}
[data-testid="stChatMessage"] {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: rgba(255,255,255,0.62);
}
.footer {
    margin-top: 2.4rem;
    padding-top: 1.1rem;
    border-top: 1px solid var(--line);
    color: #778781;
    font-size: 0.76rem;
    line-height: 1.7;
}
@media (max-width: 760px) {
    .hero { padding: 1.5rem 1.3rem; border-radius: 20px; }
    .hero h1 { font-size: 2.3rem; }
    .block-container { padding-top: 1.2rem; }
}
</style>
"""


@st.cache_resource(show_spinner=False)
def load_runtime(cache_token: str) -> tuple[Settings, Corpus, RagService, str]:
    """加载语料和 RAG 服务；配置不完整时回退到本地演示模式。"""

    del cache_token
    settings = get_settings()
    corpus = load_corpus(settings)

    warning = ""
    try:
        service = RagService(settings, corpus.chunks)
    except ProviderConfigError as exc:
        warning = str(exc)
        settings = replace(settings, llm_provider="demo")
        service = RagService(settings, corpus.chunks)

    return settings, corpus, service, warning


def _query_param(name: str) -> str:
    value: Any = st.query_params.get(name, "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def _reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.question_input = ""


def _render_sources(answer: RagAnswer) -> None:
    if not answer.sources:
        return
    with st.expander(f"查看引用来源 · {len(answer.sources)} 个片段", expanded=False):
        for index, source in enumerate(answer.sources, start=1):
            page_text = f" · 第 {source.page} 页" if source.page else ""
            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-title">{index}. {source.citation}{page_text}</div>'
                f'<div class="source-excerpt">{source.excerpt}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "📚"):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.markdown(
                    f'<div class="answer-meta">{message["mode"]} · {message["elapsed_ms"]} ms</div>',
                    unsafe_allow_html=True,
                )
                _render_sources(message["answer"])


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        with st.spinner("正在加载课程知识库…"):
            settings, corpus, service, warning = load_runtime("v1")
    except (ConfigError, IngestError) as exc:
        st.error(f"应用启动失败：{exc}")
        st.info("请检查 .env 配置和 data/raw 目录后刷新页面。")
        st.stop()
    except Exception as exc:  # pragma: no cover - 顶层兜底提示
        st.error(f"应用加载时出现异常：{exc}")
        st.info("可先执行 pytest -q 排查环境问题。")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    with st.sidebar:
        st.markdown("## 智学助手")
        st.caption("COURSE KNOWLEDGE · LANGCHAIN RAG")
        st.markdown("---")
        st.markdown("**运行模式**")
        st.markdown(f'<span class="mode-pill">{service.mode}</span>', unsafe_allow_html=True)
        if warning:
            st.warning(warning)

        st.markdown("---")
        st.markdown("**知识库概览**")
        st.metric("课程文档", f"{corpus.file_count} 份")
        st.metric("可检索片段", f"{corpus.chunk_count} 段")
        st.caption(" · ".join(corpus.source_names))

        st.markdown("---")
        st.markdown("**检索参数**")
        st.caption(
            f"Embedding：{settings.embedding_provider}\n\n"
            f"Top-K：{settings.top_k}\n\n"
            f"Chunk：{settings.chunk_size} / overlap {settings.chunk_overlap}"
        )
        if st.button("重建知识库", use_container_width=True):
            load_runtime.clear()
            st.rerun()
        if st.button("清空对话", use_container_width=True):
            _reset_chat()
            st.rerun()

        st.markdown("---")
        st.caption(
            "资料范围：开源大模型调用、LangChain、RAG、Agent 与作业规范。"
            "演示模式不会调用外部大模型。"
        )

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">LANGCHAIN · RAG · OPEN SOURCE</div>
            <h1>智学助手</h1>
            <p>把第 7 章课程资料变成可追问、可核验的知识库。每个回答都带来源，资料不足时明确说“不知道”。</p>
            <div class="hero-tags">
                <span class="tag">文档检索</span>
                <span class="tag">来源引用</span>
                <span class="tag">在线 / 离线模型</span>
                <span class="tag">可复现实验</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    stat_columns = st.columns(3)
    stats = [
        (f"{corpus.file_count} 份", "课程资料来源"),
        (f"{corpus.chunk_count} 段", "已索引文本片段"),
        (f"{corpus.raw_characters:,}", "知识库字符数"),
    ]
    for column, (value, label) in zip(stat_columns, stats):
        with column:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{value}</div>'
                f'<div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    if warning:
        st.info(f"{warning} 当前已自动切换到本地检索演示。")

    st.markdown('<div class="section-kicker">START HERE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">从一个好问题开始</div>', unsafe_allow_html=True)
    example_columns = st.columns(3)
    for index, example in enumerate(EXAMPLES):
        with example_columns[index % 3]:
            if st.button(example, key=f"example_{index}", use_container_width=True):
                st.session_state.question_input = example
                st.session_state.pending_submit = True

    auto_question = _query_param("q")
    if auto_question and not st.session_state.get("auto_question_handled"):
        st.session_state.question_input = auto_question
        st.session_state.pending_submit = True
        st.session_state.auto_question_handled = True

    _render_history()

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "向课程知识库提问",
            key="question_input",
            height=104,
            placeholder="例如：RAG 的索引阶段包括哪些步骤？",
        )
        submitted = st.form_submit_button(
            "检索并回答", type="primary", use_container_width=True
        )

    pending_submit = bool(st.session_state.pop("pending_submit", False))
    should_ask = bool(submitted or pending_submit)

    if should_ask and question.strip():
        with st.spinner("正在检索资料并组织答案…"):
            try:
                answer = service.ask(question)
            except Exception as exc:  # pragma: no cover - 依赖外部服务
                st.error(f"回答失败：{exc}")
                st.info("请检查模型配置、网络或 API Key；演示模式可继续验证检索流程。")
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": question.strip()}
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer.answer,
                        "answer": answer,
                        "mode": answer.mode,
                        "elapsed_ms": answer.elapsed_ms,
                    }
                )
                st.rerun()
    elif should_ask:
        st.warning("请先输入一个问题。")

    with st.expander("这个应用如何工作？"):
        st.markdown(
            """
            1. 文档先经过 **加载与切分**，每个片段保留文件名和片段编号；
            2. 片段经过 **Embedding** 后写入 LangChain `InMemoryVectorStore`；
            3. 提问时由 **Retriever** 召回最相关的 `Top-K` 片段；
            4. 有模型配置时，通过 **LangChain Prompt → ChatModel → OutputParser** 生成答案；
            5. 未配置模型时使用明确标注的 **本地检索演示**，仍会展示检索来源并支持拒答；
            6. 在线模式会把检索片段发送给模型服务商；离线 Ollama 模式在本机完成生成。
            """
        )

    privacy_copy = {
        "demo": "当前为本地检索演示：不调用外部大模型。",
        "ollama": "当前为离线模型模式：生成在本机完成。",
        "openai_compatible": "当前为在线模型模式：问题与检索片段会发送给模型服务商。",
    }.get(settings.llm_provider, "")
    st.markdown(
        f'<div class="footer">智学助手 · 基于 LangChain 的课程知识库 RAG 应用<br>'
        f'{privacy_copy} 答案仅供课程复习参考，请以引用原文为准。</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
