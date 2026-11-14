"""运行可复现的课程知识库评测，并生成 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.ingest import load_corpus
from src.providers import ProviderConfigError
from src.rag import RagService
from src.text_utils import excerpt


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class EvaluationCase:
    case_id: int
    question: str
    must_include: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()
    expected_source: str | None = None
    expect_refusal: bool = False


CASES = (
    EvaluationCase(
        1,
        "在线调用和离线调用分别适合什么场景？",
        must_include=("在线", "离线"),
        expected_source="01_开源大模型与调用方式.md",
    ),
    EvaluationCase(
        2,
        "LangChain 的核心模块有哪些？",
        must_include=("核心模块",),
        any_of=("Model I/O", "检索", "链", "记忆", "智能体"),
        expected_source="02_LangChain框架基础.md",
    ),
    EvaluationCase(
        3,
        "RAG 的索引阶段包括哪些步骤？",
        must_include=("索引", "加载", "拆分", "存储"),
        expected_source="03_RAG原理与实现.md",
    ),
    EvaluationCase(
        4,
        "如何降低 RAG 回答中的幻觉？",
        any_of=("系统提示词", "来源", "拒答", "资料不足"),
        expected_source="03_RAG原理与实现.md",
    ),
    EvaluationCase(
        5,
        "ReAct Agent 的工作闭环是什么？",
        any_of=("观察", "思考", "行动", "推理"),
        expected_source="04_Agent与AutoGen.md",
    ),
    EvaluationCase(
        6,
        "AutoGen 有哪几种典型协作模式？",
        must_include=("对等式", "轮询式", "层级式"),
        expected_source="04_Agent与AutoGen.md",
    ),
    EvaluationCase(
        7,
        "作业需要提交哪些内容？",
        must_include=("README", "仓库"),
        expected_source="05_第七章作业说明.md",
    ),
    EvaluationCase(
        8,
        "MCP 协议是什么？",
        expect_refusal=True,
    ),
)

REFUSAL_MARKERS = ("未找到", "资料不足", "无法", "不做推测")


def _content_ok(case: EvaluationCase, answer: str) -> bool:
    if case.expect_refusal:
        return any(marker in answer for marker in REFUSAL_MARKERS)
    # 非拒答题如果以“未找到”开头，说明模型没有回答问题；
    # 注意不能因为答案中引用了“资料不足时应拒答”这句话而误判。
    if answer.strip().startswith(REFUSAL_MARKERS):
        return False
    if any(phrase not in answer for phrase in case.must_include):
        return False
    if case.any_of and not any(phrase in answer for phrase in case.any_of):
        return False
    return True


def _source_ok(case: EvaluationCase, citations: str) -> bool:
    if case.expect_refusal or not case.expected_source:
        return True
    return case.expected_source in citations


def main() -> int:
    parser = argparse.ArgumentParser(description="智学助手 RAG 评测")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="强制使用本地检索演示，不调用任何大模型",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/latest.md"),
        help="Markdown 报告路径",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("evaluation/latest.json"),
        help="JSON 结果路径",
    )
    args = parser.parse_args()

    settings = get_settings()
    fallback_note = ""
    if args.demo:
        settings = replace(settings, llm_provider="demo")
    else:
        try:
            corpus = load_corpus(settings)
            probe_service = RagService(settings, corpus.chunks)
            del probe_service
        except ProviderConfigError as exc:
            fallback_note = str(exc)
            settings = replace(settings, llm_provider="demo")

    corpus = load_corpus(settings)
    try:
        service = RagService(settings, corpus.chunks)
    except ProviderConfigError as exc:
        fallback_note = str(exc)
        settings = replace(settings, llm_provider="demo")
        service = RagService(settings, corpus.chunks)

    records: list[dict[str, object]] = []
    for case in CASES:
        result = service.ask(case.question)
        citations = " ".join(source.citation for source in result.sources)
        content_ok = _content_ok(case, result.answer)
        source_ok = _source_ok(case, citations)
        passed = content_ok and source_ok
        records.append(
            {
                **asdict(case),
                "answer": result.answer,
                "sources": [asdict(source) for source in result.sources],
                "citations_text": citations,
                "content_ok": content_ok,
                "source_ok": source_ok,
                "passed": passed,
                "elapsed_ms": result.elapsed_ms,
            }
        )
        print(f"[{'PASS' if passed else 'FAIL'}] {case.case_id}. {case.question}")

    passed_count = sum(1 for record in records if record["passed"])
    total = len(records)
    accuracy = passed_count / total if total else 0.0
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": created_at,
        "mode": service.mode,
        "fallback_note": fallback_note,
        "passed": passed_count,
        "total": total,
        "accuracy": round(accuracy, 4),
        "records": records,
    }
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 智学助手 RAG 评测报告",
        "",
        f"- 运行时间：{created_at}",
        f"- 运行模式：{service.mode}",
        f"- 评测结果：**{passed_count}/{total}**（{accuracy:.1%}）",
    ]
    if fallback_note:
        lines.append(f"- 配置回退说明：{fallback_note}")

    is_demo = service.mode == "本地检索演示"
    report_command = (
        "python scripts/evaluate.py --demo"
        if is_demo
        else "python scripts/evaluate.py"
    )
    report_note = (
        "演示模式使用本地检索与抽取式摘要，不等同于真实大模型回答。"
        if is_demo
        else f"本报告使用{service.mode}生成，回答质量依赖所配置的模型与网络服务。"
    )
    lines.extend(
        [
            "",
            f"> 说明：本报告由 `{report_command}` 生成。{report_note}",
            "",
            "| # | 问题 | 结果 | 命中来源 | 回答摘要 |",
            "|---:|---|:---:|---|---|",
        ]
    )
    for record in records:
        mark = "✅" if record["passed"] else "❌"
        citations = str(record["citations_text"]).replace("|", "\\|")
        answer_excerpt = excerpt(str(record["answer"]), 120).replace("|", "\\|")
        question = str(record["question"]).replace("|", "\\|")
        lines.append(
            f"| {record['case_id']} | {question} | {mark} | {citations or '—'} | {answer_excerpt} |"
        )
    lines.extend(
        [
            "",
            "## 评测口径",
            "",
            "- **内容通过**：回答包含必需关键词；拒答题必须出现明确拒答表达。",
            "- **来源通过**：引用来源中出现预期文件；拒答题不强制来源。",
            "- **综合通过**：内容与来源两项同时通过。",
            "",
            "## 复现方式",
            "",
            "```bash",
            report_command,
            "```",
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nMode: {service.mode}")
    print(f"Result: {passed_count}/{total} ({accuracy:.1%})")
    print(f"Report: {args.output.resolve()}")
    print(f"JSON:   {args.json.resolve()}")
    return 0 if accuracy >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
