from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# 全局变量定义
API_KEY = "sk-pshjruauvewxlwctpjpsxldewzstyytqibhxbaptmzxsxylm"
BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDINGS_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct"
DATA_DIR = "人民日报 新闻数据集1998-2022.zip" # 下载地址： https://aistudio.baidu.com/datasetdetail/190121/0
DB_DIR = "my_db"


embbedings_model = OpenAIEmbeddings(
    model=EMBEDDINGS_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
)

llm_model = ChatOpenAI(
    model=LLM_MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
)
