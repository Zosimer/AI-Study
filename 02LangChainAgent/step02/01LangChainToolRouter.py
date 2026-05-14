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


from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Web 搜索工具，用于查询网络公开信息，不适用于内部数据.参数：query 用户查询，如 OpenAI发布会"""
    return f"模拟搜索结果：你搜索了 {query}"

@tool
def extract_pdf_text(path: str) -> str:
    """解析 PDF 文本文件。参数为文件的本地路径.参数：path 文件路径，如 /files/contract.pdf"""
    return f"模拟 PDF 内容：从 {path} 中解析出的内容"

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询，仅限内部业务数据库.参数：sql Sql语句，如 select * from users limit 5"""
    return f"模拟 SQL 执行：{sql}"

@tool
def calculate(expr: str) -> str:
    """计算数学表达式。适用于算式运算.参数：expr 数学表达式，如 (12+3)*(8-2)"""
    return str(eval(expr))

# 1.Tool 工具分组
TOOL_GROUPS = {
    "search": [search_web],
    "pdf": [extract_pdf_text],
    "database": [query_database],
    "math": [calculate],
}

# 2.创建一个意图识别模型
intent_llm = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)


from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

# 3. 定义意图分类系统提示
INTENT_SYSTEM_PROMPT = """
你是一个专业的意图分类器，请只返回以下类别之一：
- search
- pdf
- database
- math
- none

并严格只返回类别名，不要输出其它内容。
"""

# 4. 定义意图分类函数
# 必须返回 result.content！
def classify_intent(user_query: str) -> str:
    result = intent_llm.invoke(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("user", user_query)
        ]
    )
    return result.content.strip()  # 这里是关键！

# 5. 创建智能体函数
def create_agent_for_group(group: str):
    tools = TOOL_GROUPS.get(group, [])

    if not tools:
        return None

    model = load_chat_model(
        model="gpt-4o-mini",
        provider="openai",
        base_url="https://openrouter.ai/api/v1"
    )

    agent = create_agent(
        model=model,tools=tools,system_prompt="你是一个 helpful assistant，可以使用工具回答问题。你必须严格根据工具描述选择工具！如果没有合适的工具，请回答“无合适工具”"
    )

    return agent


# 6. 路由智能体函数
def router_agent(user_query: str):
    # 1. 识别意图
    intent = classify_intent(user_query)
    print(f"[Router] 检测到意图: {intent}")

    # 2. 创建对应子 Agent
    sub_agent = create_agent_for_group(intent)

    if sub_agent is None:
        return "无法为该问题找到合适的工具或 Agent。"

    # 3. 调用子 Agent 执行任务
    result = sub_agent.invoke({
        "messages": [{"role": "user", "content": user_query}]
    })

    return result

res = router_agent("请帮我搜索一下今年Google最新的大模型版本的发布会")

print(res)

# 7. 测试智能体
queries = [
        "请帮我搜索一下今年Google最新的大模型版本的发布会",
        "帮我解析一下这个PDF：/root/files/contract.pdf",
        "执行一个SQL：select * from products limit 5",
        "计算 (17+3)*(8-1)",
    ]

for q in queries:
    print("\n====== 用户问题 ======")
    print(q)
    print("====== Agent 回复 ======")
    print(router_agent(q))