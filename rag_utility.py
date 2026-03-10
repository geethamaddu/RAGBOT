import os
from dotenv import load_dotenv
load_dotenv()
print("API KEY LOADED")


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
print("Imports successful")


working_dir=os.path.dirname(os.path.abspath((__file__)))
print("path loaded")

embedding=HuggingFaceEmbeddings()
print("embeding model loaded")

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)
print("llm model loaded")

def process_document_ingestion(file_name):
    loader=PyMuPDFLoader(f"{working_dir}/{file_name}")
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500
    )
    texts=text_splitter.split_documents(documents)
    vectordb=Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0
print("process_document_ingestion successful")

def answer_question(user_question):
    vectordb=Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    retriever=vectordb.as_retriever()
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    response=qa_chain.invoke({"query":user_question})
    answer=response["result"]

    return answer

print("answer_question successful")