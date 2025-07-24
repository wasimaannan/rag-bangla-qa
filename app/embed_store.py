import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from app.preprocess import extract_text, chunk_text
from dotenv import load_dotenv

load_dotenv()

def build_vector_store():
    pdf_path = "data/HSC26_Bangla_1st_Paper.pdf"
    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("vectorstore")
