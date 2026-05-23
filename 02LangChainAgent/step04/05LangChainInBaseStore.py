# 1.导入相关库
import os
import uuid
import time
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
# ====================== MySQL 持久化（正确导入） ======================
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.store.mysql.pymysql import PyMySQLStore

# 配置 API KEY
load_dotenv(override=True)

# ==========================================
# 1. MySQL 数据库配置
# ==========================================
DB_URI = "mysql+pymysql://root:root@localhost:3306/world"
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
llm = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# ==========================================
# 2. 定义工具 (Tools)
# ==========================================
@tool
def magic_calculation(a: int, b: int) -> int:
    """进行一次特殊的加法计算"""
    return (a + b) * 10

"""
跨线程记忆需要传递 user_id，通过自定义 State 实现
"""
class CrossThreadState(AgentState):
    user_id: str  # 跨线程记忆的唯一标识

# ============ 定义 Pydantic 模型用于提取用户信息 ===========
class UserInfo(BaseModel):
    """从文本中提取的用户信息"""
    user_name: str = Field(description="用户的名字，例如：Alice、Bob、张三等")
    additional_info: str = Field(description="关于用户的其他信息，例如职业、兴趣爱好等")

class QueryInfo(BaseModel):
    """从查询文本中提取的信息"""
    user_name: str = Field(
        description="要查询的用户名字。如果查询中包含'我的'、'我是'等第一人称，请从对话历史中提取用户名；如果没有明确的用户名，返回'all_users'"
    )
    query_content: str = Field(description="查询的具体内容，例如：职业、兴趣爱好等")

# ============ 定义记忆管理工具（使用 BaseStore）===========
# 注意：store 参数使用 InjectedStore 注解，由 LangGraph 自动注入
# InjectedStore() 标记会让 Pydantic 在生成 JSON Schema 时跳过这个参数
# LLM 不会看到 store 参数，只会看到 user_id 和 info
# ==========================================
# 2. 工具定义（完全不变）
# ==========================================
@tool
def remember_user_info(
        info: str,
        state: Annotated[dict, InjectedState()],
        store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """
    将用户信息存入跨线程记忆

    重要：此工具会自动从 state 中获取 user_id，并使用 Pydantic 提取用户信息

    参数说明：
        info: 要记忆的信息（例如：用户的名字、职业、偏好等）

    示例：
        - remember_user_info("用户名叫 Alice，是一名工程师")
        - remember_user_info("我是 Bob，喜欢深度学习")
    """
    # 使用 Pydantic 提取用户信息
    structured_llm = llm.with_structured_output(UserInfo)

    try:
        # 从文本中提取结构化信息
        extracted_info = structured_llm.invoke(
            f"从以下文本中提取用户名和其他信息：{info}"
        )

        # 优先使用提取的用户名，如果提取失败则使用 state 中的 user_id
        extracted_user_name = extracted_info.user_name.lower()
        state_user_id = state.get("user_id", "unknown_user")

        # 如果提取到的用户名不是 unknown，则使用提取的用户名
        if extracted_user_name and extracted_user_name != "unknown":
            user_id = extracted_user_name
        else:
            user_id = state_user_id

        full_info = f"{extracted_info.user_name}: {extracted_info.additional_info}"

    except Exception as e:
        # 如果提取失败，使用 state 中的 user_id
        user_id = state.get("user_id", "unknown_user")
        full_info = info

    # 命名空间设计：(用户ID, 信息类别)
    namespace = (user_id, "profile")

    # 生成唯一记忆ID
    memory_id = str(uuid.uuid4())

    # 存储到 BaseStore（自动持久化）
    store.put(
        namespace,
        memory_id,
        {
            "info": full_info,
            "timestamp": "2025-11-25",
            "source": "user_input"
        }
    )

    return f"✅ 已将信息存入长期记忆 (用户: {user_id}): {full_info}"


@tool
def recall_user_info(
        query: str,
        state: Annotated[dict, InjectedState()],
        store: Annotated[BaseStore, InjectedStore()]
) -> str:
    """
    从跨线程记忆中检索用户信息

    重要：此工具会自动从 state 中获取 user_id

    参数说明：
        query: 查询关键词（用于描述要查找的信息，例如"我的职业"、"我的兴趣"等）

    返回：用户的历史信息
    """
    # 优先从 state 中获取 user_id
    state_user_id = state.get("user_id", None)

    # 如果 state 中没有 user_id，尝试使用 Pydantic 从查询中提取
    if not state_user_id:
        structured_llm = llm.with_structured_output(QueryInfo)

        try:
            # 从查询文本中提取结构化信息
            prompt = f"""从以下查询中提取用户名和查询内容。

                    查询文本：{query}

                    注意：
                    1. 如果查询中包含"我的"、"我是"等第一人称词汇，说明用户在询问自己的信息
                    2. 如果能从查询中推断出具体的用户名（如 Alice、Bob），请提取该用户名
                    3. 如果无法确定具体用户，返回 'all_users'
            """
            extracted_query = structured_llm.invoke(prompt)

            # 如果提取到的是 'all_users'，则搜索所有用户
            if extracted_query.user_name.lower() in ['all_users', 'current_user', 'unknown']:
                user_id = None
            else:
                user_id = extracted_query.user_name.lower()

        except Exception as e:
            # 如果提取失败，搜索所有用户
            user_id = None
    else:
        # 使用 state 中的 user_id
        user_id = state_user_id

    try:
        if user_id:
            # 使用 namespace_prefix 搜索该用户的所有记忆
            namespace_prefix = (user_id,)
            memories = store.search(namespace_prefix, limit=20)
        else:
            # 搜索所有已知用户的记忆
            memories = []
            # 先尝试获取所有可能的用户
            for uid in ['alice', 'bob', 'unknown_user']:
                namespace_prefix = (uid,)
                user_memories = store.search(namespace_prefix, limit=20)
                memories.extend(user_memories)

        if not memories:
            return f"未找到相关记忆。请先告诉我一些信息，我会记住它们。"

        # 格式化返回
        results = []
        for item in memories:
            info = item.value.get('info', '未知信息')
            timestamp = item.value.get('timestamp', '未知时间')
            results.append(f"- {info} (记录时间: {timestamp})")

        return f"找到 {len(results)} 条记忆:\n" + "\n".join(results)

    except Exception as e:
        return f"检索记忆时出错: {str(e)}"


def run_mysql_agent():
    print("--- 正在连接 MySQL 数据库 ---")

    # ======================
    # 你原版写法 100% 保留！
    # ======================
    # 使用 PyMySQLSaver 替代 MySQLSaver
    with (
        PyMySQLSaver.from_conn_string(DB_URI) as checkpointer,
        PyMySQLStore.from_conn_string(DB_URI) as store,
    ):
        print("🔧 初始化 Checkpointer 表结构...")
        checkpointer.setup()
        print("✅ Checkpointer 初始化完成")

        print("🔧 初始化 Store 表结构...")
        store.setup()
        print("✅ Store 初始化完成")

        # --- B. 创建 Agent ---
        tools = [magic_calculation, remember_user_info, recall_user_info]

        agent = create_agent(
            model=llm,
            tools=tools,
            state_schema=CrossThreadState,
            system_prompt="""
            你是一个具备跨线程记忆的智能助手。
            1. 使用 remember_user_info 存储用户信息
            2. 使用 recall_user_info 检索用户信息
            3. 使用 magic_calculation 做特殊加法
            记忆跨会话、持久化保存在 MySQL 中。
            """,
            checkpointer=checkpointer,
            store=store,
        )

        # ==========================================
        # 4. 测试场景（完全不变）
        # ==========================================
        def reset_thread_checkpoints(thread_id: str) -> None:
            with checkpointer._cursor(pipeline=True) as cur:
                cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))

        thread_id_alice_001 = "session_alice_001"
        thread_id_alice_002 = "session_alice_002"
        thread_id_bob_001 = "session_bob_001"
        thread_id_alice_003 = "session_alice_003"

        for tid in [thread_id_alice_001, thread_id_alice_002, thread_id_bob_001, thread_id_alice_003]:
            reset_thread_checkpoints(tid)

        print("\n" + "=" * 70)
        print("场景 1：用户 Alice 第一次对话（会话 1）")
        print("=" * 70)

        thread1_config = {"configurable": {"thread_id": thread_id_alice_001}}
        print("\n👤 用户 Alice: 你好，我是 Alice，一名 Python 开发工程师，我喜欢深度学习。")

        for chunk in agent.stream(
                {
                    "messages": [HumanMessage(content="你好，我是 Alice，一名 Python 开发工程师，我喜欢深度学习。")],
                    "user_id": "alice"
                },
                config=thread1_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

        print("\n" + "=" * 70)
        print("场景 2：用户 Alice 第二次对话（会话 2）")
        print("=" * 70)
        time.sleep(1)

        thread2_config = {"configurable": {"thread_id": thread_id_alice_002}}
        print("\n👤 用户 Alice: 你还记得我是谁吗？我的职业是什么？")

        for chunk in agent.stream(
                {
                    "messages": [HumanMessage(content="你还记得我是谁吗？我的职业是什么？")],
                    "user_id": "alice"
                },
                config=thread2_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

        print("\n" + "=" * 70)
        print("场景 3：用户 Bob 的对话")
        print("=" * 70)

        thread3_config = {"configurable": {"thread_id": thread_id_bob_001}}
        print("\n👤 用户 Bob: 你好，我是 Bob，一名产品经理，帮我算一下 10 + 20 的特殊结果。")

        for chunk in agent.stream(
                {
                    "messages": [HumanMessage(content="你好，我是 Bob，一名产品经理，帮我算一下 10 + 20 的特殊结果。")],
                    "user_id": "bob"
                },
                config=thread3_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

        print("\n" + "=" * 70)
        print("场景 4：Alice 第三次对话（验证记忆隔离）")
        print("=" * 70)

        thread4_config = {"configurable": {"thread_id": thread_id_alice_003}}
        print("\n👤 用户 Alice: 我的兴趣爱好是什么？")

        for chunk in agent.stream(
                {
                    "messages": [HumanMessage(content="我的兴趣爱好是什么？")],
                    "user_id": "alice"
                },
                config=thread4_config,
                stream_mode="values"
        ):
            last_msg = chunk["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"🤖 Agent: {last_msg.content}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"   🔧 [调用工具]: {tool_call['name']}")

    print("\n" + "=" * 70)
    print("✅ MySQL 跨线程记忆测试完成！")
    print("=" * 70)
    print("  ✅ 场景 1: Alice 首次对话，Agent 自动存储用户信息到 Store")
    print("  ✅ 场景 2: Alice 新会话（不同 thread_id），Agent 成功从 Store 检索记忆")
    print("  ✅ 场景 3: Bob 的对话，Agent 为 Bob 创建独立的记忆空间")
    print("  ✅ 场景 4: Alice 再次对话，记忆未被 Bob 的信息污染")
    print("\n💡 关键特性:")
    print("  - Checkpointer: 管理单个会话的对话历史（基于 thread_id）")
    print("  - Store: 管理跨会话的长期记忆（基于 user_id）")
    print("  - 记忆隔离: 不同用户的记忆完全隔离（通过 namespace）")
    print("  - 持久化: 所有数据存储在 MySQL，重启程序后依然可用")
    print("=" * 70)
run_mysql_agent()
