from config import get_settings
from rag.loaders import load_directory
from rag.splitter import split_pages
from rag.vector_store import KnowledgeBase


def main() -> None:
    settings = get_settings()
    pages = load_directory(settings.document_dir)
    chunks = split_pages(pages)
    knowledge_base = KnowledgeBase(settings.chroma_dir, settings.embedding_model)
    count = knowledge_base.rebuild(chunks)
    sources = {page.source for page in pages}
    print(f"知识库建立完成：{len(sources)} 个文件，{count} 个文本片段。")


if __name__ == "__main__":
    main()
