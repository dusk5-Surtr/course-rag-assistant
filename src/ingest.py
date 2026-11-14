"""课程资料加载与切分。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings


class IngestError(RuntimeError):
    """资料加载失败。"""


@dataclass(frozen=True)
class Corpus:
    source_files: tuple[Path, ...]
    chunks: tuple[Document, ...]
    raw_characters: int

    @property
    def file_count(self) -> int:
        return len(self.source_files)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.source_files)


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise IngestError(f"无法识别文件编码：{path.name}")


def _document_title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _load_text_file(path: Path) -> list[Document]:
    text = _read_text(path)
    if not text.strip():
        raise IngestError(f"文件为空：{path.name}")
    return [
        Document(
            page_content=text,
            metadata={
                "source": path.name,
                "file_name": path.name,
                "title": _document_title(text, path.stem),
                "page": None,
            },
        )
    ]


def _load_pdf_file(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时提示
        raise IngestError(
            "读取 PDF 需要安装 pypdf，请执行 pip install -r requirements.txt"
        ) from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise IngestError(f"PDF 无法打开：{path.name}（{exc}）") from exc

    pages: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_name": path.name,
                        "title": path.stem,
                        "page": page_number,
                    },
                )
            )
    if not pages:
        raise IngestError(
            f"PDF 未提取到文本：{path.name}。扫描版 PDF 需要先做 OCR。"
        )
    return pages


def load_documents(data_dir: Path) -> list[Document]:
    """加载目录下所有受支持的资料。"""

    if not data_dir.exists():
        raise IngestError(f"资料目录不存在：{data_dir}")

    files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith(".")
    )
    if not files:
        raise IngestError(
            f"资料目录中没有可读取文件：{data_dir}。支持 .md、.txt、.pdf。"
        )

    documents: list[Document] = []
    errors: list[str] = []
    for path in files:
        try:
            if path.suffix.lower() == ".pdf":
                documents.extend(_load_pdf_file(path))
            else:
                documents.extend(_load_text_file(path))
        except IngestError as exc:
            errors.append(str(exc))

    if not documents:
        detail = "；".join(errors) if errors else "没有可用文本"
        raise IngestError(f"所有资料都未能加载：{detail}")

    return documents


def split_documents(
    documents: list[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """切分文档并补充可引用的稳定标识。"""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    counters: dict[str, int] = {}
    for chunk in chunks:
        file_name = str(chunk.metadata.get("file_name") or "unknown")
        counters[file_name] = counters.get(file_name, 0) + 1
        index = counters[file_name]
        chunk.metadata["chunk_index"] = index
        chunk.metadata["citation"] = f"{file_name}#{index:03d}"
        chunk.metadata["source"] = file_name

    return [chunk for chunk in chunks if chunk.page_content.strip()]


def load_corpus(settings: Settings) -> Corpus:
    """完成加载、切分并返回语料统计。"""

    raw_documents = load_documents(settings.data_dir)
    chunks = split_documents(
        raw_documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise IngestError("资料切分后没有可用文本块")

    source_files = tuple(
        sorted(
            {
                settings.data_dir / str(document.metadata["file_name"])
                for document in raw_documents
            },
            key=lambda path: path.name,
        )
    )
    total_characters = sum(len(document.page_content) for document in raw_documents)
    return Corpus(
        source_files=source_files,
        chunks=tuple(chunks),
        raw_characters=total_characters,
    )
