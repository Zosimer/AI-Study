# 1.导入相关库
import os
import uuid
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# 加载.env
load_dotenv()

# ==========================================
# 2. 初始化向量数据库 (OpenRouter 适配版)
# ==========================================
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False
)

vector_store = Chroma(
    collection_name="agent_long_term_memory",
    embedding_function=embeddings,
)


# ==========================================
# 3. 定义记忆工具
# ==========================================
@tool
def save_memory(content: str):
    """保存长期记忆"""
    print(f"\n[记忆操作] 正在保存记忆: '{content}'")
    doc = Document(
        page_content=content,
        metadata={"source": "user_interaction", "timestamp": "simulated_time"}
    )
    vector_store.add_documents([doc])
    return "记忆已成功保存。"


@tool
def search_memory(query: str):
    """搜索长期记忆"""
    print(f"\n[记忆操作] 正在搜索记忆: '{query}'")
    results = vector_store.similarity_search(query, k=2)
    if not results:
        return "没有找到相关的记忆。"
    memory_content = "\n".join([f"- {doc.page_content}" for doc in results])
    return f"找到以下相关记忆:\n{memory_content}"


tools = [save_memory, search_memory]

# ==========================================
# 4. 创建 Agent（也适配 OpenRouter）
# ==========================================
SYSTEM_PROMPT = """你是一个拥有长期记忆的私人助手。
1. 用户告诉你喜好、信息 → 调用 save_memory
2. 用户问历史信息 → 调用 search_memory
3. 闲聊不用调用工具
"""

# ✅ LLM 也走 OpenRouter
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

checkpointer = MemorySaver()

agent_app = create_agent(
    llm,
    tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)


# ==========================================
# 5. 运行演示
# ==========================================
def run_demo():
    config_a = {"configurable": {"thread_id": "session_today"}}
    print("--- 场景 A：用户告诉喜好 ---")
    user_input_1 = "你好，记住我最喜欢的水果是草莓，而且我对花生过敏。"

    for chunk in agent_app.stream({"messages": [HumanMessage(content=user_input_1)]}, config=config_a,
                                  stream_mode="values"):
        pass
    print(f"Agent: {chunk['messages'][-1].content}")

    config_b = {"configurable": {"thread_id": "session_tomorrow"}}
    print("\n--- 场景 B：第二天查询忌口 ---")
    user_input_2 = "我想吃点零食，但我忘了我有什么忌口，你能帮我查查吗？"
    print(f"User: {user_input_2}")

    final_response = None
    for chunk in agent_app.stream({"messages": [HumanMessage(content=user_input_2)]}, config=config_b,
                                  stream_mode="values"):
        final_response = chunk['messages'][-1]

    print(f"Agent: {final_response.content}")


if __name__ == "__main__":
    run_demo()