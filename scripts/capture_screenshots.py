"""用本机 Edge 和 Playwright 截取真实运行界面，供 README 与交付验收使用。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


EDGE_PATHS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)
DEFAULT_QUESTION = "RAG 的索引阶段包括哪些步骤？"


def _edge_path() -> Path:
    for path in EDGE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("未找到 Microsoft Edge，无法执行浏览器截图")


def _assert_page_ok(page: Page) -> None:
    exceptions = page.locator('[data-testid="stException"]')
    if exceptions.count():
        raise RuntimeError(f"页面出现异常：{exceptions.first.inner_text()[:500]}")


def capture(base_url: str, output_dir: Path, prefix: str, question_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(_edge_path()),
            headless=True,
            args=["--disable-gpu", "--no-first-run", "--no-default-browser-check"],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1100},
            device_scale_factor=1,
            color_scheme="light",
            locale="zh-CN",
        )
        page = context.new_page()

        page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
        page.locator(".hero h1").wait_for(state="visible", timeout=60_000)
        page.locator(".stat-card").first.wait_for(state="visible", timeout=60_000)
        question = page.get_by_role("button", name=question_text)
        question.wait_for(state="visible", timeout=60_000)
        page.locator('[data-testid="stForm"]').wait_for(state="visible", timeout=60_000)
        page.wait_for_timeout(1_500)
        _assert_page_ok(page)
        page.screenshot(path=output_dir / f"{prefix}-home.png", full_page=False)
        print(f"saved {prefix}-home.png")

        question.click()
        page.get_by_text("查看引用来源", exact=False).first.wait_for(
            state="visible", timeout=60_000
        )
        page.wait_for_timeout(800)
        _assert_page_ok(page)

        # 展开来源，展示可核验的引用片段。
        page.get_by_text("查看引用来源", exact=False).first.click()
        page.wait_for_timeout(500)

        chat_messages = page.locator('[data-testid="stChatMessage"]')
        chat_messages.last.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=output_dir / f"{prefix}-answer.png", full_page=False)
        print(f"saved {prefix}-answer.png")

        body_text = page.locator("body").inner_text()
        print(f"page_chars={len(body_text)}")
        print(f"has_citation={'来源：' in body_text}")
        print(f"has_demo={'本地检索演示' in body_text}")
        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="截图智学助手真实运行界面")
    parser.add_argument("--base-url", default="http://127.0.0.1:8501")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    parser.add_argument("--prefix", default="app")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        capture(args.base_url, args.output_dir, args.prefix, args.question)
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
