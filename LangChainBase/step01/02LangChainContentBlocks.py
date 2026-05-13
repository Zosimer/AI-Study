import os

from langchain.chat_models import init_chat_model

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


# 使用具有多模态能力的模型
model = load_chat_model(
    model="gpt-4o-mini",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)


from langchain_core.messages import HumanMessage, SystemMessage

# 创建系统提示
system_msg = SystemMessage("你是一个专业的问答专家。")

# 构造用户消息：文本+图像
human_msg = HumanMessage(content=[
    {"type": "text", "text": "请描述图像："},
    {"type": "image_url",
     "image_url": {"url": "https://zrj18330672592.oss-cn-beijing.aliyuncs.com/20251015134735612.png",
     "mime_type": "image/jpeg",
     "metadata": "RAG基础流程图"}
    },
])

# 形成消息列表
messages = [system_msg, human_msg]

# 框架会懒解析 content -> content_blocks
for cb in human_msg.content_blocks:
    print(cb)   # content block 对象视图

res = model.invoke(messages)
"""
常用输入 type：
┌─────────────┬──────────────────────────────────────────────────────┐
│ 内容块类型   │ 标准格式（LangChain 1.0）                            │
├─────────────┼──────────────────────────────────────────────────────┤
│ 文本        │ {"type": "text", "text": "..."}                      │
│ 图像        │ {"type": "image", "url": "...", "mime_type": "..."}  │
│ 音频        │ {"type": "audio", "url": "...", "mime_type": "..."}  │
│ 视频        │ {"type": "video", "url": "...", "mime_type": "..."}  │
│ 文件        │ {"type": "file", "url": "...", "mime_type": "..."}   │
│ Base64 图像 │ {"type": "image", "base64": "...", "mime_type": "..."} │
│ Base64 音频 │ {"type": "audio", "base64": "...", "mime_type": "..."} │
│ OpenAI 图像 │ {"type": "image_url", "image_url": {"url": "..."}}   │
└─────────────┴──────────────────────────────────────────────────────┘

输出

1.text：最终回答文本（必有）
2.reasoning：模型思考过程（Claude 专属，非常常用）
3.tool_call：模型要调用工具（函数调用）
4.image：模型生成图片（部分多模态模型支持）

 """
# 遍历 content_blocks（字典写法，兼容所有版本）
for block in res.content_blocks:
    # 重点：用 ["type"] 而不是 .type
    if block["type"] == "reasoning":
        print("🔍 模型思考过程：")
        print(block["reasoning"])
    elif block["type"] == "text":
        print("\n📌 正式回答：")
        print(block["text"])

