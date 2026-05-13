import os

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field,field_validator
from typing import Literal
from langchain.agents import create_agent
# 配置密钥
from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# 1. 定义天气结构化输出模型
class WeatherForecast(BaseModel):
    """天气预报结构化输出"""
    city: str = Field(description="城市名称")
    temperature: int = Field(description="温度(摄氏度)")
    condition: Literal["晴", "雨", "多云", "雪"] = Field(description="天气状况")



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



# 1. 定义天气结构化输出模型
class WeatherForecast(BaseModel):
    """天气预报结构化输出"""
    city: str = Field(description="城市名称")
    temperature: int = Field(description="温度(摄氏度)")
    condition: Literal["晴", "雨", "多云", "雪"] = Field(description="天气状况")

# 3. 创建智能体
agent = create_agent(
    model=model,  # 加载的模型
    tools=[],  # 工具列表，这里为空
    response_format=WeatherForecast  # 指定结构化输出格式
)

# 4. 调用智能体解析天气描述
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "北京今天阳光明媚，温度10度"
    }]
})

# 5. 提取并打印结果
forecast = result["structured_response"]
print(f"{forecast.city}天气: {forecast.condition}, {forecast.temperature}°C")
