from rag.loaders import DocumentPage
from rag.splitter import split_pages, split_text


def test_split_text_preserves_content_with_overlap():
    chunks = split_text("第一段。第二段。第三段。", size=8, overlap=2)
    assert len(chunks) >= 2
    assert chunks[0]
    assert chunks[-1]


def test_split_pages_keeps_source_metadata():
    chunks = split_pages([DocumentPage("SolidWorks 拉伸特征说明。", "manual.pdf", 3)])
    assert chunks[0].source == "manual.pdf"
    assert chunks[0].page == 3
