import os
import asyncio  # 必须加
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

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

# 1. 构建 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的 AI 助手。"),
    ("human", "{question}")
])

# 3. 构建链
chain = prompt | model

# ===================== 核心修复：放进 async 函数 =====================
async def main():
    # 4. 流式事件监听
    events = chain.astream_events(
        {"question": "请用一句话介绍一下 LangChain 1.0 的核心思想。"},
        version="v1",
    )

    # 异步遍历事件
    async for event in events:
        print(f"""[Event] type={event["event"]}""")
        if "data" in event:
            print("   data:", event["data"])
        print("-----------------------------")

# ===================== 运行异步函数 =====================
if __name__ == "__main__":
    asyncio.run(main())