import os

from langchain.chat_models import init_chat_model
from typing import Literal
from langchain_core.output_parsers import JsonOutputParser
import json

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
# 配置密钥
from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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




# 1. 定义输出结构
class WeatherInfo(BaseModel):
    """天气信息"""
    city: str = Field(description="城市名称")
    temperature: int = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况")


# 2. 创建 JSON 输出解析器
json_parser = JsonOutputParser(pydantic_object=WeatherInfo)

# 3. 创建提示模板（关键：必须包含 "json" 这个词）
prompt = ChatPromptTemplate.from_template(

    """请根据以下信息提取天气数据，并以 JSON 格式返回。
    
    信息：{weather_info}
    
    请返回包含以下字段的 JSON：
    - city: 城市名称
    - temperature: 温度（摄氏度）
    - condition: 天气状况
    
    必须返回以下 JSON 格式（不要包含任何其他文本）：
    {{"city": "城市名称", "temperature": 温度数字, "condition": "天气状况"}}
    
    例如：{{"city": "北京", "temperature": 25, "condition": "晴"}}
    
    JSON 格式：
    """)
# 5. 构建链
runnable = prompt | model | json_parser

# 6. 调用
result = runnable.invoke({"weather_info": "北京今天晴，温度25度"})
print(result)
print(result["city"])
