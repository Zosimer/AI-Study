from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import asyncio

# 加载环境变量
load_dotenv()
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


# 初始化大模型
model = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# MCP 配置
mcp_config = {
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["01mcp_server.py"]
    },
    "amap-maps": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@amap/amap-maps-mcp-server"],
        "env": {
            "AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY"),
        }
    }
}


# 将所有逻辑封装在异步主函数中
async def main():
    print("正在连接 MCP 服务器...")

    # 1. 直接实例化客户端（不再使用 async with）
    client = MultiServerMCPClient(mcp_config)

    # 2. 直接获取并转换工具
    tools = await client.get_tools()
    print(f"成功加载 {len(tools)} 个工具: {[t.name for t in tools]}")

    # 3. 创建 Agent
    agent = create_agent(model, tools, system_prompt="你是会调用工具进行天气查询、地图查询、网页部署的智能助手")

    # 4. 运行 Agent
    print("\n--- 开始测试 Agent ---")
    query = "请帮我搜索查询一下北京市今天的天气，并计算一下最大温差是多少度？"
    inputs = {"messages": [HumanMessage(content=query)]}

    async for chunk in agent.astream(inputs, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        print(f"\n[{type(last_msg).__name__}]:")
        print(last_msg.content)

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f">>> 调用工具详情: {last_msg.tool_calls}")

# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())