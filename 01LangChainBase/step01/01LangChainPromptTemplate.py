from langchain_core.prompts import PromptTemplate
import os
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_chat_model(model, provider, temperature=0.7, max_tokens=None, base_url=None):
    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )

# 加载 DeepSeek 推理模型
deepseek_model = load_chat_model(
    model="deepseek/deepseek-reasoner",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# 你的提示词模板
template = PromptTemplate(
    input_variables=["user_question"],
    template="""
    你是一个专业的技术支持，回答风格：{style}。
    请先复述用户问题，然后提供解决方案。

    用户问题：{user_question}
    解决方案：""",
    partial_variables={"style": "简洁明了"}
)

# 组合链：提示词 + 模型
chain = template | deepseek_model

# 调用
res = chain.invoke({"user_question": "电脑连不上WiFi怎么办"})

print(res.content)