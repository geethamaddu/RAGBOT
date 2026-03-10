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