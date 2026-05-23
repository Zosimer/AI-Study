# 1.导入相关库
import os
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch  # 新版搜索工具
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
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
"""
通过 TypedDict 扩展 AgentState，添加业务字段（用户ID、偏好等）。
LangChain 1.0 推荐使用 TypedDict 而非 Pydantic。
"""
from typing import TypedDict, Optional
from langchain.agents import AgentState, create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 定义自定义 State 结构
class CustomAgentState(AgentState):
    """扩展的 Agent 状态，包含业务上下文"""
    user_id: str  # 用户唯一标识
    preferences: dict  # 用户偏好（主题、语言等）
    visit_count: int  # 访问次数

# ============ 定义带状态访问的工具 ============
from langchain.tools import ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

# 定义工具函数：更新用户偏好
@tool
def update_user_preference(runtime: ToolRuntime, theme: str) -> Command:
    """
    更新用户主题偏好，写入短期记忆

    ToolRuntime 提供对 state 和 context 的访问能力：
    - runtime.state: 当前状态（含自定义字段）
    - runtime.context: 调用上下文
    - runtime.tool_call_id: 工具调用ID
    """
    # 从当前状态获取偏好（如果不存在则初始化）
    current_prefs = runtime.state.get("preferences", {})
    current_prefs["theme"] = theme

    # 返回 Command 对象，指示状态更新
    return Command(update={
        "preferences": current_prefs,
        "messages": [
            ToolMessage(
                content=f"成功更新主题为: {theme}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

# 定义工具函数：根据用户偏好生成问候
@tool
def greet_user(runtime: ToolRuntime) -> str:
    """根据用户偏好生成个性化问候"""
    user_name = runtime.state.get("user_id", "访客")
    prefs = runtime.state.get("preferences", {})
    theme = prefs.get("theme", "默认")

    return f"欢迎回来，{user_name}！当前主题: {theme}"

# ============ 创建带自定义状态的 Agent ============
def demo_custom_state():
    print("\n" + "=" * 60)
    print("场景 3: 自定义 State 扩展记忆维度")
    print("=" * 60)

    # 使用内存存储
    checkpointer = InMemorySaver()

    # 创建 Agent，指定自定义 state_schema
    agent = create_agent(
        model=model,
        tools=[update_user_preference, greet_user],
        state_schema=CustomAgentState,  # 关键：传入自定义状态类型
        checkpointer=checkpointer
    )

    # 配置线程 ID（用于区分不同用户）
    config = {"configurable": {"thread_id": "custom_state_user"}}

    # 第一轮：初始化用户信息
    result1 = agent.invoke(
        {
            "messages": [{"role": "user", "content": "设置主题为暗黑模式"}],
            "user_id": "user_789",  # 自定义字段
            "preferences": {"language": "zh-CN"},  # 初始偏好
            "visit_count": 1
        },
        config=config
    )
    print(f"第一轮: {result1['messages'][-1].content}")
    print("-" * 40)

    # 第二轮：读取记忆
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "打个招呼"}]},
        config=config
    )
    print(f"第二轮: {result2['messages'][-1].content}")
    print("-" * 40)

    # 查看完整状态
    state = agent.get_state(config)
    print("当前记忆状态:")
    print(f"  用户ID: {state.values.get('user_id')}")
    print(f"  偏好: {state.values.get('preferences')}")
    print(f"  消息数: {len(state.values['messages'])}")

# ============ 运行自定义状态示例 ============
if __name__ == "__main__":
    demo_custom_state()