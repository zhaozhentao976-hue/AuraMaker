from typing import TypedDict, Literal
from langgraph.graph import END, START, StateGraph
from llm.client import CloudLLM
from rag.vector_store import KnowledgeBase, SearchResult

SIM_THRESHOLD = 0.65

class AgentState(TypedDict):
    question: str
    history: list
    sources: list[SearchResult]
    context: str
    answer: str

def build_agent(kb: KnowledgeBase, llm: CloudLLM, top_k: int):
    def retrieve(state: AgentState) -> dict:
        sources = kb.search(state["question"], top_k=top_k)
        context_parts = []
        for idx, source in enumerate(sources, 1):
            page_info = f"第{source.page}页" if source.page else "无页码"
            context_parts.append(f"[{idx}] 来源：{source.source}，{page_info}\n{source.text}")
        return {"sources": sources, "context": "\n\n".join(context_parts)}

    def route_after_retrieve(state: AgentState) -> Literal["gen_kb", "gen_fallback"]:
        valid_docs = []
        for doc in state["sources"]:
            if hasattr(doc, "score"):
                sim_score = doc.score
            else:
                sim_score = 1.0 - (doc.distance if doc.distance is not None else 1.0)
            if sim_score >= SIM_THRESHOLD:
                valid_docs.append(doc)
        return "gen_kb" if len(valid_docs) > 0 else "gen_fallback"

    def gen_kb(state: AgentState) -> dict:
        prompt_text = f"""你是SolidWorks专业工程师，严格依据下面知识库内容回答，回答末尾标注资料来源编号。
知识库检索内容：
{state["context"]}
用户问题：{state["question"]}
输出条理清晰、可直接实操的分步方案。"""
        model_ans = llm.chat(
            question=state["question"],
            context=prompt_text,
            history=state["history"]
        )
        return {"answer": model_ans}

    def gen_fallback(state: AgentState) -> dict:
        prompt_text = f"""当前没有匹配的本地知识库资料，禁止提示用户上传/补充文档。
你是资深SolidWorks工程师，直接依靠专业知识给出完整、可落地的分步操作、参数、避坑要点。
用户提问：{state["question"]}"""
        model_ans = llm.chat(
            question=state["question"],
            context=prompt_text,
            history=state["history"]
        )
        return {"answer": model_ans}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("route_after_retrieve", route_after_retrieve)
    graph.add_node("gen_kb", gen_kb)
    graph.add_node("gen_fallback", gen_fallback)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "route_after_retrieve")
    graph.add_conditional_edges(
        "route_after_retrieve",
        route_after_retrieve,
        {"gen_kb": "gen_kb", "gen_fallback": "gen_fallback"}
    )
    graph.add_edge("gen_kb", END)
    graph.add_edge("gen_fallback", END)
    return graph.compile()