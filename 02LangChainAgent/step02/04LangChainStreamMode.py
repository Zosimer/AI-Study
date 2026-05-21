# client.py
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

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
from langchain_core.tools import tool

# 1. 定义天气查询工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴朗，气温25°C",
        "上海": "多云，气温28°C",
        "广州": "小雨，气温30°C"
    }
    return f"{city}的天气是：{weather_data.get(city, '未知')}"

# 2. 定义数学计算工具
@tool
def calculate(expression: str) -> str:
    """计算一个数学表达式的结果。"""
    try:
        result = eval(expression)
        return f"计算结果是：{result}"
    except Exception as e:
        return f"计算出错：{str(e)}"

# 3. 初始化LLM
llm = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# 4. 创建Agent
agent = create_agent(
    model=llm,
    tools=[get_weather, calculate],
    system_prompt=("""
        你是一个多功能的 AI 助手，能够调用以下工具：
        1. `get_weather(city)`：查询指定城市的天气信息。参数 city 为城市名称（如“北京”）。
        2. `calculate(expression)`：计算数学表达式。参数 expression 为合法的 Python 表达式（如“25 - 28”）。
        请始终遵循以下最佳实践：
        • 当用户询问天气时，先提取城市名，再调用 `get_weather`，并返回自然语言总结。
        • 当用户需要计算时，先提取表达式，再调用 `calculate`，并给出易读的结果说明。
        • 若问题同时涉及天气与计算，按顺序依次调用对应工具，最后整合答案。
        • 禁止编造数据，必须调用工具获取结果后再回答。
        • 所有数字、单位、符号务必与工具返回保持一致，避免主观臆断。
        """
    )
)

# 5. 测试多工具调用
user_queries = [
    "北京和上海的天气怎么样？",
    "如果北京气温是25度，上海是28度，那么北京的温度比上海低多少度？"
]

# 6. 配置会话 ID
config = {"configurable": {"thread_id": "user_123"}}  # 会话 ID

# 7. 流式输出，实时观察推理过程
for step in agent.stream(
    {"messages": [{"role": "user", "content": "北京和上海的天气怎么样？"}]},
    config=config,
    stream_mode="values"    # 返回每个step步骤的完整消息列表，便于调试和观察
):
    # 获取最新消息并格式化打印
    message = step["messages"][-1]
    message.pretty_print()
    print("-" * 50)