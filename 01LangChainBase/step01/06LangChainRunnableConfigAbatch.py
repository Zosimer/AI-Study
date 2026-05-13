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
# ===================== 你的核心逻辑 =====================
async def main():
    # 并发配置（严格按你写的）
    config = RunnableConfig(
        max_concurrency=2,  # 最大并发 2
        abstimeout=8.0,  # 单个任务超时 8 秒
        metadata={"request_id": "abc123", "task": "query"},
    )

    # 模板 + 输入
    prompt_template = PromptTemplate.from_template(
        "为生产{product}的公司起一个好名字？"
    )
    inputs = ["彩色袜子", "环保咖啡杯", "智能水杯"]
    formatted_prompts = [prompt_template.format(product=p) for p in inputs]

    # 异步并发执行 ✅ 用的是 abatch
    start_time = time.time()
    print(f"⏱️ 开始时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    results = await model.abatch(formatted_prompts, config=config)

    # 输出结果
    print(f"⏱️ 结束时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"📊 总耗时: {time.time() - start_time:.2f}s\n")

    for i, r in enumerate(results):
        print(f"=== Query {i + 1} ===")
        print(r.content)
        print("-" * 40)


# 运行异步函数（解决 await 报错的关键）
if __name__ == "__main__":
    asyncio.run(main())