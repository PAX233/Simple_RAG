import config
import os
import zipfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def unzip_file(data_dir = config.DATA_DIR):
    # 获取数据集目录
    file_dir = os.getcwd()
    zip_path = os.path.join(file_dir, data_dir)
    if not os.path.exists(zip_path):
        print(f"错误：未找到数据集目录 {zip_path}")
        return

    # 解压数据集
    data_folder = os.path.join(file_dir, "unzipped_data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_folder)

def process_news_data(data_dir = config.DATA_DIR):
    cnt = 0
    data_folder = os.path.join(os.getcwd(),"unzipped_data")
    # 处理数据集,添加日期前缀
    folders = sorted([f for f in os.listdir(data_folder)])
    data_final = []
    for folder in folders:
        if (folder == "people1998") :
            continue    # 1998年的新闻数据集格式不同，跳过

        folder_path = os.path.join(data_folder, folder)
        files = sorted(os.listdir(folder_path))
        for file in files:
            cnt += 1
            if(cnt>=150): # 限制rag大小
                return data_final

            if file.endswith(".txt"):
                # 读取文本
                file_path = os.path.join(folder_path, file)
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                
                #分割文本
                text_splitter = RecursiveCharacterTextSplitter.from_language(language="markdown", chunk_size=512, chunk_overlap=128)
                texts = text_splitter.create_documents(
                    [documents[0].page_content]
                )

                # 添加元信息
                date = file.split(".")[0]

                for text in texts:
                    text.metadata = {"date": date}
                    data_final.append(text)
    return data_final



# unzip_file()
texts = process_news_data()
# 建库
db = FAISS.from_documents(texts, config.embbedings_model)
FAISS.save_local(db, config.DB_DIR)