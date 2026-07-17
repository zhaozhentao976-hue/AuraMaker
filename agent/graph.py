from typing import TypedDict, Literal
from langgraph.graph import END, START, StateGraph
from llm.client import CloudLLM
from rag.vector_store import KnowledgeBase, SearchResult

SIM_THRESHOLD = 0.65

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

    def route_after_retrieve(state: AgentState) -> Literal["gen_kb", "gen_fallback"]:
        sources = state["sources"]
        valid_sources = [s for s in sources if s.score >= SIM_THRESHOLD]
        if len(valid_sources) > 0:
            return "gen_kb"
        else:
            return "gen_fallback"

    def gen_kb(state: AgentState) -> dict:
        kb_prompt = f"""
【引用本地知识库文档】
严格仅依据下方知识库原文回答机械智造、SolidWorks相关问题，禁止编造工艺、建模参数。
知识库参考内容：
{state["context"]}
用户提问：{state["question"]}
回答要求：步骤清晰，所有操作、参数必须能在上述原文找到依据，回答中保留文档来源标记。
"""
        return {
            "answer": llm.answer(
                question=state["question"],
                context=kb_prompt,
                history=state["history"],
            )
        }

    def gen_fallback(state: AgentState) -> dict:
        fallback_prompt = f"""
【重要提示：本地知识库未检索到匹配内容，以下内容为DeepSeek V4 PRO通用行业标准操作，无本地实训资料支撑】
仅输出SolidWorks软件原生标准操作流程，不得编造非标工艺参数；若涉及小众特种加工，主动说明无对应本地资料。
用户提问：{state["question"]}
"""
        return {
            "answer": llm.answer(
                question=state["question"],
                context=fallback_prompt,
                history=state["history"],
            )
        }

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
        {
            "gen_kb": "gen_kb",
            "gen_fallback": "gen_fallback"
        }
    )
    graph.add_edge("gen_kb", END)
    graph.add_edge("gen_fallback", END)

    return graph.compile()