import requests

class CloudLLM:
    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key:
            raise ValueError("尚未配置 LLM_API_KEY，请先填写 .env 文件。")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, question: str, context: str, history: list[dict[str, str]]) -> str:
        # DeepSeek专属系统设定：有知识库参考就结合资料，无资料直接AI独立完整回答
        system_prompt = (
            "你是资深SolidWorks专业工程师，精通零件建模、装配、工程图、仿真、公差、出图全套操作。\n"
            "1. 若提供了知识库文档，优先基于文档内容回答，回答中标注来源编号；\n"
            "2. 若没有检索到知识库内容，禁止让用户补充、上传资料，直接依靠你的专业知识给出完整分步操作、参数、注意事项；\n"
            "3. 全程中文作答，条理清晰，不编造不存在的标准、尺寸、软件功能；\n"
            "4. 回答贴合实操，简洁易懂。"
        )

        user_text = f"知识库参考内容：\n{context if context else '暂无匹配知识库资料'}\n用户问题：{question}"
        message_list = [{"role": "system", "content": system_prompt}]
        # 保留最近6轮对话上下文
        message_list.extend(history[-6:])
        message_list.append({"role": "user", "content": user_text})

        request_body = {
            "model": self.model,
            "messages": message_list,
            "temperature": 0.1
        }

        # 请求DeepSeek接口
        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers=self.request_headers,
            json=request_body
        )

        if response.status_code != 200:
            return f"DeepSeek接口调用失败，错误详情：{response.text}"

        res_data = response.json()
        if not res_data.get("choices"):
            return "DeepSeek模型未返回有效内容"

        return res_data["choices"][0]["message"]["content"].strip()