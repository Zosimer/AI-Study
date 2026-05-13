from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
import os

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
def load_chat_model(
    model: str,
    provider: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    base_url: str | None = None,
):
    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url

    )

model = load_chat_model(
    model="openai/gpt-4o-mini",  # OpenRouter 固定格式
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# 使用messages模板字符串（最常用）
chat_template = ChatPromptTemplate.from_messages([

    # SystemMessage: 定义AI角色和行为准则
    ("system", "你是一个专业的Python代码审查助手。请严格检查代码风格、潜在Bug和性能问题。"),

    # HumanMessage: 用户输入
    ("human", "请审查以下代码：\n\n{code_snippet}"),

    # AIMessage: 可选，提供示例输出（Few-shot）
    ("ai", "我发现了以下问题：1. 缺少类型注解 2. 使用全局变量"),

    # HumanMessage: 用户的后续指令
    ("human", "{follow_up_instruction}")
])

# 格式化：生成消息列表
messages = chat_template.format_messages(

    code_snippet="def add(a,b):\n    return a+b",

    follow_up_instruction="请给出优化后的代码"
)

print("生成的消息结构：")
for i, msg in enumerate(messages):
    print(f"\n--- 消息 {i + 1} ---")
    print(f"角色: {msg.schema}")
    print(f"内容: {msg.content}")

# 直接传递给模型
response = model.invoke(messages)
print("\n 模型审查结果：")
print(response.content_blocks[0]["text"])

