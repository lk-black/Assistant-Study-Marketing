import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

def create_vector_store(persist_directory:str, file_path:str) -> Chroma:
    """Create a vector store from a PDF file."""
    
    if not os.path.exists(persist_directory):
        print(f"Persistent directory does not exist. Creating directory: {persist_directory}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=10,
        )
        docs = text_splitter.split_documents(documents)
        print(f"Loaded {len(docs)} documents from {file_path}")
        
        print(f"--- Creating vector store ---")
        embeddings = HuggingFaceEmbeddings(
            model_kwargs={"device": "cuda",}
        )
        
        db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=persist_directory,
        )
        print(f"Finished creating vector store with {len(docs)} documents.")
        return db
    
    else:
        print(f"Persistent directory already exists: {persist_directory}")