import os
import asyncio  # 必须加上
from langchain.chat_models import init_chat_model
import time
from datetime import datetime
from langchain_core.runnables import RunnableConfig

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

# =============== 核心修复：把所有代码放进 async 函数里 ===============
async def main():
    # 设置并发
    config = RunnableConfig(max_concurrency=3)

    # 记录时间
    start_time = time.time()
    print(f"⏱️  开始时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    # 并发批量请求
    responses = await model.abatch([
        "请介绍下你自己。",
        "请问什么是机器学习？",
        "你知道机器学习和深度学习区别么？"
    ], config=config)

    # 输出结果
    for i, resp in enumerate(responses, 1):
        print(f"\n【回答 {i}】")
        print(resp)

    # 耗时统计
    end_time = time.time()
    print(f"\n⏱️  结束时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    print(f"📊 总耗时: {end_time - start_time:.2f}s")

# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())