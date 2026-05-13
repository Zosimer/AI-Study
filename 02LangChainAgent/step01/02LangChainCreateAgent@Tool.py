# 1.导入相关库
import os
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch  # 新版搜索工具
from dotenv import load_dotenv

# 加载 .env 文件里的密钥
load_dotenv()

# 2.配置环境变量（从 .env 读取）
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 4.创建模型
def load_chat_model(model, provider, temperature=0.7, max_tokens=None, base_url=None):
    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )

# 1.导入模型
model = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)
# 2. 定义一个简单的 Tool (Runnable)
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

# 3.创建Agent
agent = create_agent(model=model,tools=[multiply])

# 4. 调用Agent
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "帮我计算12乘以6等于多少？"
    }]
})

print(response["messages"])
print(response["messages"][-1].content)