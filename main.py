import config
import data_process
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


if __name__ == "__main__":
    # data_process.data_process() # 处理数据

    db = data_process.data_load()
    print("数据库已加载")

    retriever = db.as_retriever(search_kwargs={"k": 2})

    # 创建带有 system 消息的模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """你是一个新闻机器人。
                你的任务是根据下述给定的已知信息回答用户问题。
                可以在以下信息的基础上进行联网查找。不要编造答案。
                请用中文回答用户问题。

                已知信息:
                {context} """),
        ("user", "{question}")
    ])

    # 自定义的提示词参数
    chain_type_kwargs = {
        "prompt": prompt_template,
    }

    # 定义RetrievalQA链
    qa_chain = RetrievalQA.from_chain_type(
        llm=config.llm_model,
        chain_type="stuff",  # 使用stuff模式将上下文拼接到提示词中
        chain_type_kwargs=chain_type_kwargs,
        retriever=retriever,
        return_source_documents=True
    )
    response=qa_chain({"query": "2021年1月1日发生了什么大事件"})

    print (response)
    print(response["result"])

    # 如果需要，可以查看源文档
    print(response["source_documents"])


