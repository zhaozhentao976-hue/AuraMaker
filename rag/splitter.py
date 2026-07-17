from dataclasses import dataclass

from rag.loaders import DocumentPage


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int | None
    index: int


def split_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = max(normalized.rfind("。", start, end), normalized.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_pages(pages: list[DocumentPage]) -> list[Chunk]:
    result: list[Chunk] = []
    for page in pages:
        for index, text in enumerate(split_text(page.text)):
            result.append(Chunk(text, page.source, page.page, index))
    return result

