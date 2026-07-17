from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.splitter import Chunk


COLLECTION_NAME = "solidworks_knowledge"


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    page: int | None
    distance: float | None


class KnowledgeBase:
    def __init__(self, persist_dir: Path, embedding_model: str):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=self.embedding,
            metadata={"hnsw:space": "cosine"},
        )

    def rebuild(self, chunks: list[Chunk]) -> int:
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            COLLECTION_NAME,
            embedding_function=self.embedding,
            metadata={"hnsw:space": "cosine"},
        )
        if not chunks:
            return 0
        ids = [sha256(f"{c.source}:{c.page}:{c.index}:{c.text}".encode()).hexdigest() for c in chunks]
        metadata = [
            {"source": c.source, "page": c.page if c.page is not None else -1, "index": c.index}
            for c in chunks
        ]
        self.collection.add(ids=ids, documents=[c.text for c in chunks], metadatas=metadata)
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_texts=[query], n_results=min(top_k, self.collection.count()))
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            SearchResult(
                text=text,
                source=meta["source"],
                page=None if meta.get("page", -1) == -1 else int(meta["page"]),
                distance=distances[i] if i < len(distances) else None,
            )
            for i, (text, meta) in enumerate(zip(documents, metadatas))
        ]
