import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

# Set page configuration
st.set_page_config(
    page_title="FAISS Document Q&A Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .loading-text {
        color: #ff9800;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():

    token=os.getenv("HUGGINGFACE_TOKEN")
    
    """Load embedding model, vector store, and language model (cached for performance)"""
    try:
        # Load embedding model
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load vector store
        vectorstore = FAISS.load_local("vector_db", embedding, allow_dangerous_deserialization=True)
        
        # Load language model
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=token
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        return embedding, vectorstore, pipe
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def get_answer(query, vectorstore, pipe, k=3, max_tokens=300):
    """Get answer for the query using FAISS search and language model"""
    try:
        # Search for similar documents
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are an expert in Answering Documents named Hao Ho-Minh, you can answer all the questions like a strong LLMs model. 
Especially with questions related to the given document context, you will answer exactly what is in the document provided, 
answer in detail and do not create anything extra:

Context:
{context}

Question: {query}
Answer:"""
        
        # Generate answer
        output = pipe(prompt, max_new_tokens=max_tokens, do_sample=True)[0]["generated_text"]
        answer = output.split("Answer:")[-1].strip()
        
        return answer, docs
    
    except Exception as e:
        return f"Error generating answer: {str(e)}", []

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 FAISS Document Q&A Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        k_docs = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)
        max_tokens = st.slider("Max tokens for answer", min_value=50, max_value=500, value=300)
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.info("""
        **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
        
        **Language Model:** Meta-Llama-3-8B-Instruct
        
        **Vector Store:** FAISS
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load models
    if "models_loaded" not in st.session_state:
        with st.spinner("🔄 Loading models... This may take a few minutes."):
            embedding, vectorstore, pipe = load_models()
            
            if embedding and vectorstore and pipe:
                st.session_state.embedding = embedding
                st.session_state.vectorstore = vectorstore
                st.session_state.pipe = pipe
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
            else:
                st.error("❌ Failed to load models. Please check your setup.")
                return
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
            <div class="chat-message user-message">
                <div class="message-label">👤 You:</div>
                <div>{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="chat-message bot-message">
                <div class="message-label">🤖 Assistant:</div>
                <div>{message["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Show source documents if available
            if "sources" in message:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(message["sources"]):
                        st.write(f"**Document {i+1}:**")
                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.write(f"*Metadata: {doc.metadata}*")
                        st.write("---")
    
    # Chat input
    if query := st.chat_input("💬 Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        st.markdown(f'''
        <div class="chat-message user-message">
            <div class="message-label">👤 You:</div>
            <div>{query}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Generate and display assistant response
        with st.spinner("🤔 Thinking..."):
            answer, docs = get_answer(
                query, 
                st.session_state.vectorstore, 
                st.session_state.pipe, 
                k=k_docs, 
                max_tokens=max_tokens
            )
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": docs
        })
        
        # Display assistant message
        st.markdown(f'''
        <div class="chat-message bot-message">
            <div class="message-label">🤖 Assistant:</div>
            <div>{answer}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Show source documents
        if docs:
            with st.expander("📚 Source Documents"):
                for i, doc in enumerate(docs):
                    st.write(f"**Document {i+1}:**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    if hasattr(doc, 'metadata') and doc.metadata:
                        st.write(f"*Metadata: {doc.metadata}*")
                    st.write("---")
        
        # Rerun to update the display
        st.rerun()

if __name__ == "__main__":
    main()