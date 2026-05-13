# 导入操作系统相关模块，用于设置环境变量
import os

# LangChain 核心：加载聊天模型的函数
from langchain.chat_models import init_chat_model
# LangChain 消息类型：用户消息、AI消息、系统提示词
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 导入Gradio库，用于快速构建网页聊天界面
import gradio as gr

# ===================== 1. 配置大模型相关 =====================
# 设置API密钥（通过环境变量方式，这是LangChain推荐的安全方式）
from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def load_chat_model(model, provider, temperature=0.7, max_tokens=None, base_url=None):
    """
    封装加载大模型的函数，方便统一配置
    :param model: 模型名称，如 deepseek/deepseek-chat
    :param provider: 模型提供商，这里用openai协议兼容方式
    :param temperature: 温度系数，控制回答的随机性
    :param max_tokens: 最大生成token数
    :param base_url: API接口地址
    :return: 加载好的模型对象
    """
    return init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )

# 加载DeepSeek模型（通过OpenRouter调用）
model = load_chat_model(
    model="deepseek/deepseek-chat",
    provider="openai",
    base_url="https://openrouter.ai/api/v1"
)

# 系统提示词：定义AI助手的身份、语气、行为规则
system_message = SystemMessage(
    content="你叫小智，是一名乐于助人的智能助手。请在对话中保持友好、有耐心、温和的语气。"
)

# ===================== 2. 界面样式配置 =====================
# 自定义CSS样式，让网页更美观
CSS = """
.main-container {max-width: 1200px; margin: 0 auto; padding: 20px;}
.header-text {text-align: center; margin-bottom: 20px;}
"""

# ===================== 3. 创建聊天界面 =====================
def create_chatbot() -> gr.Blocks:
    """创建Gradio聊天界面，返回界面对象"""
    # Blocks是Gradio的布局容器，用于自定义界面
    with gr.Blocks(title="DeepSeek Chat") as demo:
        # 界面整体布局容器
        with gr.Column(elem_classes=["main-container"]):
            # 标题
            gr.Markdown("# 🤖 LangChain 1.0 × DeepSeek Chatbot", elem_classes=["header-text"])
            # 副标题
            gr.Markdown("基于 LangChain 1.0 标准接口的流式对话机器人", elem_classes=["header-text"])

            # 聊天窗口组件：显示对话记录
            chatbot = gr.Chatbot(height=500)

            # 输入框：用户输入问题
            msg = gr.Textbox(placeholder="请输入您的问题...", container=False)

            # 按钮行：发送 + 清空
            with gr.Row():
                submit = gr.Button("发送", variant="primary")  # 主要按钮
                clear = gr.Button("清空")                      # 清空按钮

        # 状态变量：保存LangChain格式的消息历史（用于上下文对话）
        state = gr.State([])

        # ===================== 核心聊天函数 =====================
        def respond(user_msg, chat_history, messages):
            """
            响应用户输入，流式返回AI回答
            :param user_msg: 用户输入的文本
            :param chat_history: Gradio界面显示的聊天记录
            :param messages: LangChain格式的消息列表
            :yield: 实时更新界面
            """
            # 如果用户输入为空，直接返回，不处理
            if not user_msg.strip():
                yield "", chat_history, messages
                return

            # 第一次对话时，初始化系统提示词
            if not messages:
                messages = [system_message]

            # 把用户消息加入LangChain消息列表
            messages.append(HumanMessage(content=user_msg))

            # 更新Gradio界面：添加用户消息
            chat_history.append({"role": "user", "content": user_msg})
            # 添加空的AI消息占位，用于流式填充
            chat_history.append({"role": "assistant", "content": ""})

            # 先返回一次，把用户消息显示在界面上
            yield "", chat_history, messages

            # 流式生成AI回答
            partial = ""  # 存储实时拼接的回答
            for chunk in model.stream(messages):  # 模型逐字返回
                if chunk.content:  # 如果有内容
                    partial += chunk.content  # 拼接
                    chat_history[-1]["content"] = partial  # 更新最后一条AI消息
                    yield "", chat_history, messages  # 实时刷新界面

            # 对话完成：把完整的AI回答存入LangChain消息列表
            messages.append(AIMessage(content=partial))
            # 限制最多保留50轮对话，防止内存溢出
            messages = messages[-50:]

            # 最后刷新一次，保证状态同步
            yield "", chat_history, messages

        # ===================== 清空对话函数 =====================
        def clear_all():
            """清空所有聊天记录和状态"""
            return "", [], []

        # ===================== 绑定按钮/回车事件 =====================
        # 按回车提交
        msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])
        # 点击发送按钮提交
        submit.click(respond, [msg, chatbot, state], [msg, chatbot, state])
        # 点击清空按钮重置
        clear.click(clear_all, outputs=[msg, chatbot, state])

    return demo

# ===================== 4. 启动应用 =====================
if __name__ == "__main__":
    print("\n🚀 启动 Gradio 应用...")
    # 创建界面
    demo = create_chatbot()
    # 启动服务
    demo.launch(
        server_name="0.0.0.0",    # 允许局域网访问
        server_port=7860,         # 端口号
        css=CSS                   # 加载样式
    )