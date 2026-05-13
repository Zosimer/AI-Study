# 1.导入相关库
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch  # 新版搜索工具
from dotenv import load_dotenv

# 加载 .env 文件里的密钥
load_dotenv()

# 2.配置环境变量（从 .env 读取）
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")  # 必须加这个

# 3.导入模型和工具
web_search = TavilySearch(max_results=2)  # 新版用法

# 4.创建模型
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

# 5.创建Agent
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="你是一名多才多艺的智能助手，可以调用工具帮助用户解决问题。"
)

# 6.运行Agent获得结果
result = agent.invoke(
    {"messages": [{"role": "user", "content": "请帮我查询2024年诺贝尔物理学奖得主是谁？"}]}
)
print(result['messages'][-1].content)