# 1. 使用init_embeddings初始化嵌入模型
from langchain.embeddings import init_embeddings
import os
from dotenv import load_dotenv

# 加载 .env 文件里的密钥（不会上传到 GitHub）
load_dotenv()

# 从环境变量读取，不再写死在代码里
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# 2. 初始化OpenAI的text-embedding-3-small嵌入模型
embedding = init_embeddings(model="text-embedding-3-small",provider="openai", base_url="https://openrouter.ai/api/v1" )

# 3. 将文本转换为向量表示
res = embedding.embed_query("Hello world")

# 4. 打印向量的前10个元素
print(res[:10])

"""
[-0.00212860107421875, -0.049041748046875, 0.0209808349609375, 0.0313720703125, -0.04534912109375, -0.02642822265625, -0.0289306640625, 0.060272216796875, -0.0257568359375, -0.0148162841796875]

"""