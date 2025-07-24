import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from dotenv import load_dotenv

chain = None

def get_qa_chain():
    global chain
    if chain is None:
        vectorstore = FAISS.load_local("vectorstore", OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))
        retriever = vectorstore.as_retriever()
        llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain

def answer_question(question):
    chain = get_qa_chain()
    return chain.run(question)
