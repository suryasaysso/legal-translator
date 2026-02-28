import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def process_pdf_to_chroma(pdf_path, db_directory="./chroma_db"):
    """Loads a PDF, chunks it, and saves it as vectors."""
    
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 2. Split into legal-sized chunks
    # 1000 chars is usually enough for a single ToS clause
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(pages)

    # 3. Create Embeddings using Nomic (optimized for M4)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Store in ChromaDB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory
    )
    
    return vector_db