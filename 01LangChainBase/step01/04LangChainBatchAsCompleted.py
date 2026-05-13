import os
import time
from datetime import datetime

from langchain.chat_models import init_chat_model

# 记录开始时间
start_time = time.time()
print(f"⏱️  开始时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

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


# 使用具有多模态能力的模型
model = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# 使用 model.batch_as_completed 批量提交多个问题，并逐个获取回答
for response in model.batch_as_completed([
    "请介绍下你自己。",
    "请问什么是机器学习？",
    "你知道机器学习和深度学习区别么？"
]):
 print(response)
end_time = time.time()
total_duration = end_time - start_time
print(f"⏱️  结束时间: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
print(f"📊 总耗时: {total_duration:.2f}s")
