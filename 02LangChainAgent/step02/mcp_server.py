# mcp_server.py
from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务端实例
mcp = FastMCP("MathWeatherServer")

@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的当前天气和气温"""
    # 这里为了演示，使用模拟数据。实际使用时可以替换为调用真实天气API
    weather_data = {
        "北京": "晴，气温 25°C",
        "上海": "多云，气温 28°C",
        "合肥": "小雨，气温 22°C"
    }
    return weather_data.get(city, f"抱歉，暂时无法查询到 {city} 的天气信息。")

@mcp.tool()
def calculate_difference(a: float, b: float) -> float:
    """计算两个数字的差值（a - b）"""
    return a - b

if __name__ == "__main__":
    # 启动服务端，使用 stdio（标准输入输出）进行通信
    mcp.run(transport="stdio")