import logging
import sys
from mcp.server.fastmcp import FastMCP

# ===================== 生产级日志初始化 =====================
# 配置日志：同时输出到控制台(stderr)和日志文件(mcp_server.log)
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别，调试时可改为 logging.DEBUG
    format="%(asctime)s | %(levelname)s | %(message)s",  # 日志格式：时间 | 级别 | 内容
    handlers=[
        logging.FileHandler("mcp_server.log", encoding="utf-8"),  # 1. 记录到本地文件
        logging.StreamHandler(sys.stderr)  # 2. 打印到控制台（走标准错误输出，绝对安全）
    ]
)
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# ==========================================================

# 创建 MCP 服务端实例
mcp = FastMCP("MathWeatherServer")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的当前天气和气温"""
    # 使用 logger 记录日志，带时间戳且不会干扰协议
    logger.info(f"收到客户端请求，正在查询城市：{city}")

    weather_data = {
        "北京": "晴，气温 25°C",
        "上海": "多云，气温 28°C",
        "合肥": "小雨，气温 22°C"
    }
    return weather_data.get(city, f"抱歉，暂时无法查询到 {city} 的天气信息。")


@mcp.tool()
def calculate_difference(a: float, b: float) -> float:
    """计算两个数字的差值（a - b）"""
    logger.info(f"收到计算请求，正在计算差值：{a} - {b}")
    return a - b


if __name__ == "__main__":
    # 启动日志
    logger.info("🚀 MCP Server 启动成功！正在通过 stdio 等待客户端连接...")
    # 启动服务端
    mcp.run(transport="stdio")