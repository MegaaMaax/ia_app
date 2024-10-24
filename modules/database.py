import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
# from modules.constants import MISTRAL_API_KEY

def get_vector_store():
    return Chroma(
        collection_name="rag_db",
        embedding_function=MistralAIEmbeddings(model='mistral-embed'),
        persist_directory="db_dir",
    )

def upload_database(directory):
    vector_store = get_vector_store()
    text = ""
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    vector_store.add_texts(texts=splits)
    
    return "Database uploaded successfully"