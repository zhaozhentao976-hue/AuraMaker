from pathlib import Path

import streamlit as st

from agent.graph import build_agent
from config import get_settings
from llm.client import CloudLLM
from rag.loaders import SUPPORTED_EXTENSIONS, load_directory
from rag.splitter import split_pages
from rag.vector_store import KnowledgeBase


st.set_page_config(page_title="SolidWorks QA Agent", page_icon="🧰", layout="wide")
st.title("SolidWorks 本地 QA Agent")
st.caption("资料和索引保存在本机；问题及检索片段会发送给配置的云端模型。")

settings = get_settings()


@st.cache_resource
def get_kb() -> KnowledgeBase:
    return KnowledgeBase(settings.chroma_dir, settings.embedding_model)


def save_upload(uploaded_file) -> Path:
    safe_name = Path(uploaded_file.name).name
    target = settings.document_dir / safe_name
    target.write_bytes(uploaded_file.getbuffer())
    return target


with st.sidebar:
    st.header("知识库")
    uploads = st.file_uploader(
        "添加资料",
        type=[suffix.lstrip(".") for suffix in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )
    if uploads:
        for upload in uploads:
            save_upload(upload)
        st.success(f"已保存 {len(uploads)} 个文件")

    files = [p.name for p in settings.document_dir.iterdir() if p.is_file()]
    st.write(f"本地资料：{len(files)} 个")
    if files:
        st.caption("、".join(files))

    if st.button("建立 / 更新知识库", type="primary", use_container_width=True):
        try:
            with st.spinner("正在解析资料并建立本地索引……"):
                pages = load_directory(settings.document_dir)
                chunks = split_pages(pages)
                count = get_kb().rebuild(chunks)
            st.success(f"完成，共写入 {count} 个文本片段")
        except Exception as exc:
            st.error(f"建立知识库失败：{exc}")

    st.divider()
    st.caption(f"模型：{settings.llm_model}")
    st.caption(f"接口：{settings.llm_base_url}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("询问 SolidWorks 或本地资料中的问题")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索并回答……"):
                llm = CloudLLM(settings.llm_api_key, settings.llm_base_url, settings.llm_model)
                agent = build_agent(get_kb(), llm, settings.top_k)
                result = agent.invoke(
                    {"question": question, "history": st.session_state.messages[:-1], "sources": [], "context": "", "answer": ""}
                )
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("查看引用来源"):
                    for number, source in enumerate(result["sources"], 1):
                        page = f" · 第 {source.page} 页" if source.page else ""
                        st.markdown(f"**[{number}] {source.source}{page}**")
                        st.caption(source.text)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except Exception as exc:
            st.error(str(exc))

