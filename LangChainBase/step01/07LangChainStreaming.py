import os
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from datetime import datetime
import time

# 配置密钥
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

# 加载模型
model = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)
# # 使用.stream()方法进行流式传输
# for chunk in model.stream("用一段话描述大海。"):
#     print(chunk.content, end="", flush=True)  # 逐块打印

# 初始化变量，用于累积模型返回的完整内容
full = None  # 初始值为空

# 使用流式方式调用模型，逐块接收返回内容
for chunk in model.stream("你好，好久不见"):
    # 如果是第一块内容，则直接赋值；否则拼接到已有内容
    full = chunk if full is None else full + chunk

    # 打印当前累积的文本内容
    print(full.text)

    print(full.content_blocks)