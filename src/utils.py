from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(pdf_path):
    loader=PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def text_split(document, chunk_size=500, chunk_overlap=50):
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(document)
    return text_chunks
    
