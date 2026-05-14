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

# 初始化大模型（这里保持你原有的 OpenRouter 配置）
model = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# 主逻辑
async def main():
    # 指向我们刚刚写好的 MCP 服务端文件
    mcp_server_path = "mcp_server.py"
    print("正在连接 MCP 服务端...")

    # 连接 MCP 服务
    mcp_client = MultiServerMCPClient(
        {
            "math-weather": {
                "transport": "stdio",
                "command": "python",
                "args": [mcp_server_path],
            },
        }
    )

    # 自动加载 MCP 服务端暴露的所有工具
    try:
        mcp_tools = await mcp_client.get_tools()
        print(f"✅ 成功加载 {len(mcp_tools)} 个 MCP 工具: {[t.name for t in mcp_tools]}")
    except Exception as e:
        print(f"❌ 加载 MCP 工具失败: {e}")
        return

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=mcp_tools,
        system_prompt="你是一个多功能助手，可以调用工具查询天气和进行数学计算。"
    )

    # 执行任务
    print("\n正在向 Agent 提问...")
    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "查询一下北京和上海的气温，并且计算一下北京的温度比上海低多少度？"}]
    })
    print(f"\n🤖 Agent 回复: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())