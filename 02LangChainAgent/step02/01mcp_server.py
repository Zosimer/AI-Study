import logging
import sys
from mcp.server.fastmcp import FastMCP

# ===================== 生产级日志初始化（修复中文乱码） =====================
# 强制设置控制台输出编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 配置日志：同时输出到控制台(stderr)和日志文件(mcp_server.log)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        # 文件日志 UTF-8 编码
        logging.FileHandler("mcp_server.log", encoding="utf-8"),
        # 控制台日志，强制使用 UTF-8
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)
# ==========================================================

# 创建 MCP 服务端实例
mcp = FastMCP("MathWeatherServer")


@mcp.tool()
def get_weather(city: str) -> str:
    """查询指定城市的当前天气和气温"""
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
    logger.info("🚀 MCP Server 启动成功！正在通过 stdio 等待客户端连接...")
    mcp.run(transport="stdio")