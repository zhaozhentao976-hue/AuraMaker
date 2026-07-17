from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from llm.client import CloudLLM
from rag.vector_store import KnowledgeBase, SearchResult


class AgentState(TypedDict):
    question: str
    history: list[dict[str, str]]
    sources: list[SearchResult]
    context: str
    answer: str


def build_agent(kb: KnowledgeBase, llm: CloudLLM, top_k: int):
    def retrieve(state: AgentState) -> dict:
        sources = kb.search(state["question"], top_k=top_k)
        context_parts = []
        for number, source in enumerate(sources, 1):
            location = f"{source.source}，第{source.page}页" if source.page else source.source
            context_parts.append(f"[{number}] 来源：{location}\n{source.text}")
        return {"sources": sources, "context": "\n\n".join(context_parts)}

    def generate(state: AgentState) -> dict:
        return {
            "answer": llm.answer(
                question=state["question"],
                context=state["context"],
                history=state["history"],
            )
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

