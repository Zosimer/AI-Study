import sys  # 1. 导入 sys 模块
from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务端实例
mcp = FastMCP("MathWeatherServer")

@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的当前天气和气温"""
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
    # 2. 加上 file=sys.stderr，将日志输出到标准错误流
    print("✅ MCP Server 启动成功！正在等待客户端连接...", file=sys.stderr)
    # 启动服务端，使用 stdio（标准输入输出）进行通信
    mcp.run(transport="stdio")