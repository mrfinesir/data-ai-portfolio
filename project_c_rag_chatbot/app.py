"""
Streamlit Web App for RAG Chatbot - Simplified Version
Run with: streamlit run app.py
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 RAG Chatbot - Ask Questions About Your Documents")
st.markdown("This chatbot answers questions based on the documents you've uploaded.")

# ============================================
# Load models and database (cached)
# ============================================

@st.cache_resource
def load_rag_components():
    """Load embedding model, ChromaDB, and LLM"""
    
    with st.spinner("Loading chatbot components..."):
        # Load ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            collection = client.get_collection("documents")
        except:
            st.error("No vector database found. Run 'python rag_chatbot.py' first.")
            return None, None, None
        
        # Load embedding model
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load LLM
        try:
            llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=200,
                temperature=0.7
            )
        except:
            llm = None
        
        return collection, embedder, llm

# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.header("📁 Documents")
    if os.path.exists("./documents"):
        docs = os.listdir("./documents")
        st.write(f"Loaded: {len(docs)} file(s)")
        for doc in docs:
            st.write(f"- {doc}")
    else:
        st.write("No documents folder")
    
    st.divider()
    st.markdown("### How RAG Works")
    st.markdown("""
    1. Documents are split into chunks
    2. Each chunk gets a vector embedding
    3. Your question finds similar chunks
    4. AI answers using those chunks
    """)
    st.divider()
    st.caption("Built with ChromaDB | Sentence Transformers | Hugging Face")

# ============================================
# Load components
# ============================================

collection, embedder, llm = load_rag_components()

if collection is None:
    st.warning("⚠️ Chatbot not ready. Please run 'python rag_chatbot.py' first.")
    st.stop()

# ============================================
# Query function
# ============================================

def query_chatbot(question, top_k=3):
    """Search and generate answer"""
    
    # Create embedding for question
    question_embedding = embedder.encode(question).tolist()
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    relevant_chunks = results['documents'][0] if results['documents'] else []
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the documents.", []
    
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""Answer based ONLY on this context. If not in context, say "I don't have that information."

Context: {context}

Question: {question}

Answer:"""
    
    if llm:
        answer = llm(prompt, max_length=200)[0]['generated_text']
    else:
        answer = relevant_chunks[0][:300]
    
    return answer, relevant_chunks

# ============================================
# Chat interface
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            answer, sources = query_chatbot(prompt)
            
            with st.expander("📖 View sources"):
                for i, source in enumerate(sources[:3]):
                    st.text(f"Source {i+1}:")
                    st.text(source[:300] + "..." if len(source) > 300 else source)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("RAG Chatbot | Powered by ChromaDB + Sentence Transformers")