"""
RAG Chatbot - Simplified Cloud Version (No ChromaDB)
Works on Streamlit Cloud without any issues
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
import os

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

# ============================================
# Knowledge base (documents)
# ============================================

DOCUMENTS = [
    {
        "text": "Groq is a company that builds specialized hardware for ultra-fast AI inference. Their LPU (Language Processing Unit) is designed specifically for running LLMs at high speed. Groq provides an API for developers to access fast inference.",
        "source": "Groq Documentation"
    },
    {
        "text": "RAG (Retrieval-Augmented Generation) combines search with LLMs for accurate, grounded answers. It retrieves relevant information from documents and uses it to generate responses, reducing hallucinations.",
        "source": "AI Research"
    },
    {
        "text": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed. It includes supervised, unsupervised, and reinforcement learning.",
        "source": "ML Basics"
    },
    {
        "text": "To use Groq API: 1. Sign up at console.groq.com, 2. Get your API key, 3. Install groq package, 4. Use the key in your code. Models include llama-3.1-8b-instant and llama-3.3-70b-versatile.",
        "source": "Groq API Guide"
    },
    {
        "text": "Streamlit is a Python library that turns data scripts into shareable web apps in minutes. It's great for prototyping AI applications.",
        "source": "Streamlit Docs"
    }
]

# ============================================
# Cache models
# ============================================

@st.cache_resource
def load_embedder():
    with st.spinner("Loading AI model..."):
        return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def create_embeddings():
    embedder = load_embedder()
    documents = [doc["text"] for doc in DOCUMENTS]
    embeddings = embedder.encode(documents)
    return embeddings, documents

# ============================================
# Search function
# ============================================

def semantic_search(query, embeddings, documents, top_k=3):
    embedder = load_embedder()
    query_embedding = embedder.encode([query])[0]
    
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "text": documents[idx],
            "score": similarities[idx],
            "source": DOCUMENTS[idx]["source"]
        })
    
    return results

# ============================================
# Generate answer
# ============================================

def generate_answer(question, relevant_docs):
    context = "\n\n".join([
        f"[Source {i+1} - {doc['source']}]:\n{doc['text']}"
        for i, doc in enumerate(relevant_docs)
    ])
    
    prompt = f"""Context:
{context}

Question: {question}

Answer based ONLY on the context above:"""

    client = Groq(api_key=GROQ_API_KEY)
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You answer questions based only on the provided context. Be concise."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content

# ============================================
# Load components
# ============================================

with st.spinner("Loading chatbot..."):
    embeddings, documents = create_embeddings()

st.success("✅ Chatbot ready! Ask me anything.")

# ============================================
# Chat interface
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about Groq, AI, RAG..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            results = semantic_search(prompt, embeddings, documents, top_k=3)
            
            with st.expander("📖 Sources"):
                for i, doc in enumerate(results):
                    st.markdown(f"**Source {i+1} ({doc['score']:.2%} match)**")
                    st.text(doc['text'][:300])
                    st.caption(f"From: {doc['source']}")
                    st.divider()
            
            answer = generate_answer(prompt, results)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.caption("Powered by Groq Llama 3 | Sentence Transformers")