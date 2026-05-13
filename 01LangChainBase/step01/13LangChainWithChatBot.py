import os

from langchain.chat_models import init_chat_model

from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage, AIMessage, SystemMessage
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



# 2?? 初始化系统提示词（System Prompt）
system_message = SystemMessage(
    content="你叫小智，是一名乐于助人的智能助手。请在对话中保持温和、有耐心的语气。"
)

# 3?? 初始化消息历史
messages = [system_message]

print("?? 输入 exit 退出对话\n")

# 4?? 主循环（支持多轮对话 + 流式输出）
while True:
    user_input = input("?? 你：")
    if user_input.lower() in {"exit", "quit"}:
        print("?? 对话结束，再见！")
        break

    # 追加用户消息
    messages.append(HumanMessage(content=user_input))

    # 实时输出模型生成内容
    print("?? 小智：", end="", flush=True)
    full_reply = ""

    # ? LangChain 1.0 标准写法：流式输出
    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_reply += chunk.content

    print("\n" + "-" * 40)  # 分隔线
    #
    # # 追加 AI 回复消息
    # messages.append(AIMessage(content=full_reply))
    #
    # # 保持消息长度（只保留最近50轮）
    # messages = messages[-50:]