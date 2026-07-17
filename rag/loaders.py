from dataclasses import dataclass
from pathlib import Path

import fitz
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass(frozen=True)
class DocumentPage:
    text: str
    source: str
    page: int | None


def load_document(path: Path, source: str | None = None) -> list[DocumentPage]:
    source_name = source or path.name
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            return [
                DocumentPage(page.get_text("text"), source_name, number + 1)
                for number, page in enumerate(pdf)
                if page.get_text("text").strip()
            ]
    if suffix == ".docx":
        document = Document(path)
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
        return [DocumentPage(text, source_name, None)] if text else []
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [DocumentPage(text, source_name, None)] if text.strip() else []
    raise ValueError(f"不支持的文件类型：{suffix}")


def load_directory(directory: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            source = path.relative_to(directory).as_posix()
            pages.extend(load_document(path, source=source))
    return pages
