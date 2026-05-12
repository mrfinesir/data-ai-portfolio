"""
RAG Chatbot - Streamlit Web App
Run with: streamlit run app.py
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in .env file. Please create a .env file with your API key.")
    st.stop()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG Chatbot")
st.markdown("Ask questions about your documents")

@st.cache_resource
def load_components():
    with st.spinner("Loading..."):
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("documents")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        groq_client = Groq(api_key=GROQ_API_KEY)
        return collection, embedder, groq_client

def search(query, collection, embedder, top_k=3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0] if results['documents'] else []

collection, embedder, groq_client = load_components()

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
            
            with st.expander("📖 Sources"):
                for i, chunk in enumerate(chunks[:2]):
                    st.text(f"Source {i+1}: {chunk[:200]}...")
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("RAG Chatbot | ChromaDB + Groq Llama 3")