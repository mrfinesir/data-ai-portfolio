"""
RAG Chatbot - Deployed on Streamlit Cloud
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# For cloud deployment, use Streamlit secrets
# Local development still uses .env via python-dotenv

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Get API key from secrets (cloud) or environment (local)
if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please add it to Streamlit secrets.")
    st.stop()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about Groq, AI, RAG, and more!")

# Cache the vector database and models
@st.cache_resource
def load_components():
    with st.spinner("Loading chatbot components..."):
        # Create chroma db in a writable location
        import tempfile
        import os
        
        # Use temp directory for chroma (cloud has read-only filesystem)
        temp_dir = tempfile.mkdtemp()
        client = chromadb.PersistentClient(path=temp_dir)
        
        # Check if we have pre-built data or need to create it
        st.warning("Note: First run may take a moment to build the database.")
        
        # Create collection
        collection = client.create_collection(name="documents")
        
        # Add sample documents
        documents = [
            "Groq is a company that builds specialized hardware for ultra-fast AI inference. Their LPU (Language Processing Unit) is designed for running LLMs at high speed.",
            "RAG (Retrieval-Augmented Generation) combines search with LLMs for accurate, grounded answers. It retrieves relevant information from documents and uses it to generate responses.",
            "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed.",
            "ChromaDB is an open-source vector database for storing embeddings and performing semantic search."
        ]
        
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        for i, doc in enumerate(documents):
            embedding = embedder.encode(doc).tolist()
            collection.add(
                ids=[f"doc_{i}"],
                embeddings=[embedding],
                documents=[doc],
                metadatas=[{"source": "builtin", "id": i}]
            )
        
        groq_client = Groq(api_key=GROQ_API_KEY)
        return collection, embedder, groq_client

def search(query, collection, embedder, top_k=3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []

# Load components
collection, embedder, groq_client = load_components()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = search(prompt, collection, embedder)
            context = "\n\n".join(chunks) if chunks else "No relevant documents found."
            
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Answer based only on the context provided."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"}
                ],
                temperature=0.3,
                max_tokens=300
            )
            answer = response.choices[0].message.content
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("RAG Chatbot | Powered by ChromaDB + Groq Llama 3")