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
# 正确导入：langgraph.checkpoint.mysql.pymysql.PyMySQLSaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langchain.tools import tool

"""
生产环境使用数据库存储，支持：
- 持久化（重启不丢失）
- 多实例共享（分布式部署）
- 大规模并发
"""

print("\n" + "=" * 60)
print("场景 2: MySQL 持久化记忆（生产环境）")
print("=" * 60)

# 定义工具函数：查询用户信息
@tool
def get_user_info(name: str) -> str:
    """查询用户信息，返回姓名、年龄、爱好"""
    user_db = {
        "陈明": {"age": 28, "hobby": "旅游、滑雪、喝茶"},
        "张三": {"age": 32, "hobby": "编程、阅读、电影"}
    }
    info = user_db.get(name, {"age": "未知", "hobby": "未知"})
    return f"姓名: {name}, 年龄: {info['age']}岁, 爱好: {info['hobby']}"

# 模型加载函数
def load_chat_model(model: str, provider: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model)

# MySQL 连接串（端口默认3306）
DB_URI = "mysql+pymysql://root:root@localhost:3306/world"

# 使用 PyMySQLSaver 替代 MySQLSaver
with PyMySQLSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动建表

    # 创建 React 智能体
    from langgraph.prebuilt import create_react_agent
    agent = create_agent(
        model=model,
        tools=[get_user_info],
        checkpointer=checkpointer
    )

    # 用户会话ID
    config = {"configurable": {"thread_id": "production_user_001"}}

    agent.invoke(
        {"messages": [{"role": "user", "content": "我是新用户张三，请记录我的信息"}]},
        config=config
    )

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "我是谁？"}]},
        config=config
    )
    print(f"AI: {response['messages'][-1].content}")