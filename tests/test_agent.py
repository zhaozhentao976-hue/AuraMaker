from agent.graph import build_agent
from rag.vector_store import SearchResult


class FakeKnowledgeBase:
    def search(self, query: str, top_k: int):
        assert query == "如何创建拉伸？"
        return [SearchResult("先创建草图，再使用拉伸凸台。", "手册.pdf", 12, 0.1)]


class FakeLLM:
    def answer(self, question: str, context: str, history: list[dict[str, str]]):
        assert "手册.pdf" in context
        return "根据手册，先创建草图，再使用拉伸凸台。[1]"


def test_agent_retrieves_and_generates_answer():
    agent = build_agent(FakeKnowledgeBase(), FakeLLM(), top_k=5)
    result = agent.invoke(
        {"question": "如何创建拉伸？", "history": [], "sources": [], "context": "", "answer": ""}
    )
    assert result["answer"].endswith("[1]")
    assert result["sources"][0].page == 12

