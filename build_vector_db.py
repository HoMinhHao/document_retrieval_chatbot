from src.utils import *
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

document_path = "./a-practical-guide-to-building-agents.pdf"
documents = load_pdf(document_path)
text_chunks = text_split(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.from_documents(text_chunks, embedding_model)
vector_db.save_local("vector_db")



# To load the vector database later, you can use:
# vector_db = FAISS.load_local("vector_db", embedding_model)








