"""
Setup script to create the vector database
Run once: python setup_database.py
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
import shutil
from dotenv import load_dotenv

# Load environment variables (optional, not strictly needed for setup)
load_dotenv()

print("=" * 60)
print("Setting up RAG Chatbot Database")
print("=" * 60)

# Create documents folder
documents_dir = "./documents"
if not os.path.exists(documents_dir):
    os.makedirs(documents_dir)

# Create sample document
sample_content = """Artificial Intelligence (AI) is the simulation of human intelligence in machines.

Machine learning is a subset of AI that enables systems to learn from data.

RAG (Retrieval-Augmented Generation) combines search with LLMs for accurate answers.

Groq is a company that builds specialized hardware for ultra-fast AI inference.

Groq's LPU (Language Processing Unit) is designed specifically for running LLMs at high speed.

Groq provides an API for fast LLM inference. Sign up at console.groq.com.

ChromaDB is a vector database for embeddings.

Streamlit is a Python library for creating web apps quickly."""

with open("documents/sample.txt", "w", encoding='utf-8') as f:
    f.write(sample_content)
print("✅ Created sample document")

# Read and chunk
with open("documents/sample.txt", "r", encoding='utf-8') as f:
    text = f.read()

chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
print(f"✅ Created {len(chunks)} chunks")

# Create vector database
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(name="documents")

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

for i, chunk in enumerate(chunks):
    embedding = embedder.encode(chunk).tolist()
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"source": "sample.txt", "chunk_id": i}]
    )

print(f"✅ Added {len(chunks)} chunks to database")
print("\n✅ Setup complete! Run 'streamlit run app.py'")