import os
import asyncio  # 必须加
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# 配置密钥
from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from typing import List
from langchain_core.utils.pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 1. 定义期望的输出结构 (Pydantic 模型)
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="人的姓名")
    age: int = Field(description="人的年龄")
    high: int = Field(description="人的身高")
    hobbies: List[str] = Field(description="人的爱好列表")

# 2. 初始化模型并绑定结构化输出格式
llm = ChatOpenAI(model="gpt-4o", base_url="https://openrouter.ai/api/v1",temperature=0)

structured_llm = llm.with_structured_output(Person)

# 3. 调用模型并获取 Pydantic 对象，构造提示：要求提取约翰·多伊的姓名、年龄和兴趣爱好
prompt = "提取名为约翰·多伊的人的信息，提取不到的数据就为空值。他30岁，喜欢阅读、远足和弹吉他."

result = structured_llm.invoke(prompt)

# 4. 验证结果
print(f"Type of result: {type(result)}")
print(f"Result object: {result}")

# 5.判断result是否属于Person类
assert isinstance(result, Person)
structured_llm = llm.with_structured_output(Person, include_raw=True)

# 2. 调用模型并获取 Pydantic 对象
prompt = "提取名为约翰·多伊的人的信息。他30岁，喜欢阅读、远足和弹吉他."

# 3. 调用模型，返回结构化结果（包含解析后的 Person 对象和原始文本）
result = structured_llm.invoke(prompt)

# 4. 验证结果：打印返回值的类型与内容，便于调试
print(f"Type of result: {type(result)}")
print(f"Result object: {result}")