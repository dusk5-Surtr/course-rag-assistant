"""将 README.md 渲染为可提交、可预览的 README.pdf。"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import markdown
from pypdf import PdfReader
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDGE_PATHS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)
OUTPUT = PROJECT_ROOT / "README.pdf"


def _edge_path() -> Path:
    for path in EDGE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("未找到 Microsoft Edge")


def _embed_images(html: str) -> str:
    """把 README 中的相对图片内嵌为 data URI，确保 PDF 中可见。"""

    pattern = re.compile(r'src="(docs/images/[^"]+)"')

    def replace(match: re.Match[str]) -> str:
        relative = match.group(1)
        path = PROJECT_ROOT / relative
        if not path.exists():
            return match.group(0)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        suffix = path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        return f'src="data:image/{mime};base64,{encoded}"'

    return pattern.sub(replace, html)


def _architecture_html() -> str:
    return """
<div class="arch">
  <div class="arch-title">建库流程</div>
  <div class="arch-row">
    <span class="arch-box">PDF / MD / TXT</span><span class="arch-arrow">→</span>
    <span class="arch-box">Document Loader</span><span class="arch-arrow">→</span>
    <span class="arch-box">Text Splitter</span><span class="arch-arrow">→</span>
    <span class="arch-box">Embedding</span><span class="arch-arrow">→</span>
    <span class="arch-box arch-accent">Vector Store</span>
  </div>
  <div class="arch-title">问答流程</div>
  <div class="arch-row">
    <span class="arch-box">用户问题</span><span class="arch-arrow">→</span>
    <span class="arch-box">Retriever</span><span class="arch-arrow">→</span>
    <span class="arch-box">Prompt</span><span class="arch-arrow">→</span>
    <span class="arch-box">在线 / 离线 / 演示</span><span class="arch-arrow">→</span>
    <span class="arch-box arch-accent">答案 + 来源</span>
  </div>
</div>
"""


def _build_html(markdown_text: str) -> str:
    markdown_text = re.sub(r"^<!--.*?-->\s*", "", markdown_text, flags=re.DOTALL)
    markdown_text = re.sub(r"^# .+?\n", "", markdown_text, count=1)
    markdown_text = re.sub(r"\n---\s*\n\*\*课程作业\*\*.*\Z", "", markdown_text, flags=re.DOTALL)
    markdown_text = re.sub(r"^- \[x\]\s+", "- ☑ ", markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(r"^- \[ \]\s+", "- ☐ ", markdown_text, flags=re.MULTILINE)
    markdown_text = re.sub(
        r"```mermaid.*?```",
        "\n\nARCHITECTURE_DIAGRAM_TOKEN\n\n",
        markdown_text,
        flags=re.DOTALL,
    )
    body = markdown.markdown(
        markdown_text,
        extensions=["extra", "tables", "fenced_code", "sane_lists", "toc"],
    )
    body = body.replace(
        "<p>ARCHITECTURE_DIAGRAM_TOKEN</p>",
        _architecture_html(),
    )
    body = _embed_images(body)
    base_uri = PROJECT_ROOT.as_uri() + "/"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<base href="{base_uri}">
<title>智学助手 README</title>
<style>
@page {{
  size: A4;
  margin: 16mm 15mm 18mm;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: #18302d;
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
  font-size: 10.5pt;
  line-height: 1.72;
}}
.cover {{
  min-height: 250mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 20mm 16mm;
  border-radius: 8mm;
  color: #fffdf6;
  background:
    radial-gradient(circle at 92% 8%, rgba(223,243,233,.22), transparent 65mm),
    linear-gradient(135deg, #083d37, #12604f);
  page-break-after: always;
}}
.cover .kicker {{ letter-spacing: .22em; font-size: 9pt; color: #bce5d5; font-weight: 700; }}
.cover h1 {{ margin: 10mm 0 4mm; color: #fffdf6; font-size: 31pt; line-height: 1.1; letter-spacing: -1.2pt; }}
.cover p {{ margin: 0; color: #dff3e9; font-size: 12pt; line-height: 1.8; }}
.cover .meta {{ margin-top: 20mm; padding-top: 8mm; border-top: 1px solid rgba(255,255,255,.25); font-size: 10pt; color: #e9f7f0; }}
.cover .meta strong {{ color: #fffdf6; }}
main {{ padding-top: 4mm; }}
h1, h2, h3, h4 {{ color: #0d4f46; page-break-after: avoid; }}
h1 {{ font-size: 19pt; margin: 2mm 0 6mm; }}
h2 {{ font-size: 15pt; margin: 8mm 0 3mm; padding-bottom: 2mm; border-bottom: 1px solid #cfe0d8; }}
h3 {{ font-size: 12pt; margin: 6mm 0 2mm; }}
p, ul, ol, blockquote, table, pre {{ page-break-inside: avoid; }}
a {{ color: #0b6d5e; text-decoration: none; }}
blockquote {{
  margin: 4mm 0;
  padding: 3mm 5mm;
  border-left: 4px solid #8ecab5;
  background: #eef7f2;
  color: #425d55;
}}
code {{
  padding: 1px 4px;
  border-radius: 4px;
  background: #edf3ef;
  color: #205a4d;
  font-family: "Cascadia Code", Consolas, monospace;
  font-size: 9pt;
}}
pre {{
  padding: 4mm;
  border-radius: 3mm;
  border: 1px solid #d7e3dc;
  background: #f3f7f4;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 8.5pt;
  line-height: 1.55;
}}
pre code {{ padding: 0; background: transparent; }}
table {{
  width: 100%;
  margin: 4mm 0;
  border-collapse: collapse;
  font-size: 9pt;
}}
th {{ background: #e3f1ea; color: #0d4f46; text-align: left; }}
th, td {{ border: 1px solid #cfddd6; padding: 2.2mm 2.5mm; vertical-align: top; }}
tr:nth-child(even) td {{ background: #f8faf8; }}
img {{
  display: block;
  max-width: 100%;
  max-height: 155mm;
  margin: 4mm auto 7mm;
  border: 1px solid #cdded6;
  border-radius: 3mm;
  box-shadow: 0 4mm 12mm rgba(20, 65, 54, .10);
  page-break-inside: avoid;
}}
hr {{ border: 0; border-top: 1px solid #cfddd6; margin: 7mm 0; }}
li {{ margin: 1.2mm 0; }}
.arch {{
  margin: 4mm 0 6mm;
  padding: 5mm;
  border: 1px solid #cbded5;
  border-radius: 4mm;
  background: linear-gradient(145deg, #f5faf7, #e9f4ee);
}}
.arch-title {{
  margin: 1mm 0 2.5mm;
  color: #0d4f46;
  font-size: 10pt;
  font-weight: 800;
}}
.arch-title:not(:first-child) {{ margin-top: 5mm; }}
.arch-row {{
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 1.5mm;
}}
.arch-box {{
  display: inline-block;
  padding: 2.2mm 2.8mm;
  border: 1px solid #b9d4c8;
  border-radius: 2mm;
  background: #ffffff;
  color: #274d43;
  font-size: 8.5pt;
  font-weight: 650;
}}
.arch-accent {{
  border-color: #0d4f46;
  background: #0d4f46;
  color: #fffdf6;
}}
.arch-arrow {{ color: #6e9c8d; font-weight: 800; }}
</style>
</head>
<body>
  <section class="cover">
    <div class="kicker">OPEN SOURCE · LANGCHAIN · RAG</div>
    <h1>智学助手<br>Course Knowledge RAG</h1>
    <p>基于 LangChain 的课程知识库问答应用<br>可溯源回答 · 资料不足拒答 · 在线与离线模型可切换</p>
    <div class="meta">
      <strong>课程：</strong>《开源软件构建技术》第七章 智能应用开发<br>
      <strong>作者：</strong>张天健<br>
      <strong>仓库：</strong>https://github.com/dusk5-Surtr/course-rag-assistant<br>
      <strong>版本日期：</strong>2026-11-14
    </div>
  </section>
  <main>
  {body}
  </main>
</body>
</html>"""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    markdown_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    html = _build_html(markdown_text)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(_edge_path()),
            headless=True,
            args=["--disable-gpu", "--no-first-run"],
        )
        page = browser.new_page(viewport={"width": 1200, "height": 1600})
        page.set_content(html, wait_until="load")
        page.evaluate(
            """async () => {
                await Promise.all(Array.from(document.images).map(img => {
                    if (img.complete) return Promise.resolve();
                    return new Promise(resolve => { img.onload = img.onerror = resolve; });
                }));
            }"""
        )
        page.emulate_media(media="print")
        page.pdf(
            path=str(OUTPUT),
            format="A4",
            print_background=True,
            display_header_footer=False,
            margin={"top": "16mm", "right": "15mm", "bottom": "18mm", "left": "15mm"},
            prefer_css_page_size=True,
        )
        browser.close()

    pages = len(PdfReader(str(OUTPUT)).pages)
    size = OUTPUT.stat().st_size
    print(f"created={OUTPUT}")
    print(f"pages={pages} bytes={size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
