# SolidWorks 本地 QA Agent

一个最小可用的本地知识库问答应用。文档、向量索引和聊天记录保存在本机；检索到的少量上下文会发送给配置的云端 LLM。

## 功能

- 导入 PDF、DOCX、TXT 和 Markdown
- 使用本地中文 Embedding 和 ChromaDB 建立索引
- 使用 LangGraph 编排检索与回答
- 调用 OpenAI 兼容的云端模型接口
- 回答中展示文件名、页码和原文片段
- 没有可靠资料时要求模型明确说明依据不足

## 安装

建议使用 Python 3.11 或 3.12。部分机器上 Python 3.13 安装 ChromaDB 或机器学习依赖时可能需要额外编译环境。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填写云端模型的密钥、接口地址和模型名称。

## 运行

```powershell
streamlit run app.py
```

打开页面后，在侧边栏上传资料并点击“建立/更新知识库”，然后开始提问。首次建立索引时会下载本地 Embedding 模型。

也可以在命令行重建知识库：

```powershell
.venv\Scripts\python.exe ingest.py
```

## 数据边界

原始文档和完整索引不会主动上传，但用户问题、最近对话和检索出的文档片段会发送给云端模型。不要导入不允许发送任何片段到第三方模型的机密资料。
