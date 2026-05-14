# client.py
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
import asyncio

# 加载环境变量
load_dotenv()
# 确保你本地 .env 文件里配置了 OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 模型加载
def load_chat_model(model, provider, temperature=0.7, max_tokens=None, base_url=None):
    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )


from langchain.agents import create_agent
from langchain.tools import tool

model = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)
# 1. 定义一个简单的天气查询工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴朗，气温25°C",
        "上海": "多云，气温28°C",
        "广州": "小雨，气温30°C"
    }
    return f"{city}的天气是：{weather_data.get(city, '未知')}"

# 2. 静态 system_prompt（固定不变）
agent_static = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=(
        "你是一个天气助手，回答不超过20字。\n"
        "调用工具时，严格按照以下格式：\n"
        "1. 使用 `get_weather(city: str)` 获取天气；\n"
        "2. 仅返回天气结果，不解释过程。"
    )
)

print("=== 静态 System Prompt ===")
response1 = agent_static.invoke({
    "messages": [{"role": "user", "content": "北京天气"}]
})
print(f"AI: {response1['messages'][-1].content}")

# 3. 动态提示词（通过中间件实现）
from langchain.agents.middleware import dynamic_prompt
from typing import TypedDict

# 4. 定义上下文结构
class Context(TypedDict):
    user_role: str  # 用户角色

# 5. 动态提示函数
@dynamic_prompt
def role_based_prompt(request):
    """根据用户角色生成不同提示词"""
    user_role = request.runtime.context.get("user_role", "user")

    if user_role == "expert":
        return "你是一个专业气象分析师，提供详细数据"
    elif user_role == "beginner":
        return "你是一个友善的导游，用简单语言解释"
    else:
        return "你是一个简洁的天气助手"

# 6. 创建动态 Agent
agent_dynamic = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[role_based_prompt],  # 注入动态提示
    context_schema=Context
)

print("\n=== 动态 System Prompt（专家角色）===")
response2 = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "北京天气"}]},
    context={"user_role": "expert"}
)
print(f"AI: {response2['messages'][-1].content}")

print("\n=== 动态 System Prompt（新手角色）===")
response3 = agent_dynamic.invoke(
    {"messages": [{"role": "user", "content": "北京天气"}]},
    context={"user_role": "beginner"}
)
print(f"AI: {response3['messages'][-1].content}")