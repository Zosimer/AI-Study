# 1. 定义带速率限制的load_chat_model函数
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
import os

# 配置 OpenRouter API Key（你的密钥）
from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 2. 配置速率限制器
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,
    check_every_n_seconds=1.0
)

# 3. 模型封装函数
def load_chat_model(
    model: str,
    provider: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    base_url: str | None = None,
):
    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        rate_limiter=rate_limiter
    )

"""
你以为的：
我定义一次 llm，整个项目都用这一个，只写一遍。
真实项目里的情况：
你会创建很多不同的模型、不同配置的 llm，比如：
聊天机器人用：gpt-3.5-turbo，温度 0.7
代码生成用：gpt-4o，温度 0.1
总结文章用：gpt-3.5-turbo，温度 0.3，max_tokens 4000
对接本地模型：llama3，base_url 自己填
对接通义千问、文心一言、Claude……
每一种，配置都不一样。
所以你会写出这样的代码（不封装的写法）
# 1. 聊天机器人
llm_chat = init_chat_model(model="gpt-3.5", model_provider="openai", temperature=0.7, rate_limiter=rate_limiter)

# 2. 代码生成
llm_code = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0.1, rate_limiter=rate_limiter)

# 3. 文章总结
llm_summary = init_chat_model(model="gpt-3.5", model_provider="openai", temperature=0.3, max_tokens=4000, rate_limiter=rate_limiter)

# 4. 本地模型
llm_local = init_chat_model(model="llama3", model_provider="ollama", base_url="http://localhost:11434", rate_limiter=rate_limiter)
看到了吗？
你要写 N 遍 rate_limiter=rate_limiter
你要写 N 遍 model_provider=
你要记 N 遍参数顺序
封装成 load_chat_model 之后

llm_chat = load_chat_model("gpt-3.5", "openai")
llm_code = load_chat_model("gpt-4o", "openai", 0.1)
llm_summary = load_chat_model("gpt-3.5", "openai", 0.3, max_tokens=4000)
llm_local = load_chat_model("llama3", "ollama", base_url="http://localhost:11434")

好处：
不用每次写 rate_limiter（函数内部自动传）
不用写长名字（更短）
不会漏参数（更安全）
以后要改，只改函数内部，所有模型自动生效
"""
# 初始化模型（OpenRouter）
model = load_chat_model(
    model="openai/gpt-4o-mini",  # OpenRouter 固定格式
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)
model = model.with_retry(
        stop_after_attempt=3,         # 最多重试3次
        wait_exponential_jitter=True  # 指数退避 + 随机抖动
    )

# 测试调用
res = model.invoke("请介绍一下你自己")
print(res)