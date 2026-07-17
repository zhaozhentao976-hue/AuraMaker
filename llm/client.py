from openai import OpenAI


class CloudLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("尚未配置 LLM_API_KEY，请先填写 .env 文件。")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def answer(self, question: str, context: str, history: list[dict[str, str]]) -> str:
        system = (
            "你是一个严谨的 SolidWorks 本地知识库助手。"
            "优先根据给出的资料回答，不得编造资料中没有的尺寸、标准或操作结论。"
            "若资料不足，请明确说‘当前知识库中没有足够依据’，再说明需要什么资料。"
            "回答使用中文，并在相关结论后标注上下文中的来源编号，例如[1]。"
        )
        user = f"知识库上下文：\n{context or '（没有检索到资料）'}\n\n用户问题：{question}"
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": user})
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.1)
        return response.choices[0].message.content or "模型没有返回内容。"

