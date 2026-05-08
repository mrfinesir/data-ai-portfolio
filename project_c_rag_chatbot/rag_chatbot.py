"""
Day 16: RAG Chatbot - Simplified Working Version
No LangChain dependencies - uses ChromaDB + Transformers directly
"""

import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pypdf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DAY 16: RAG CHATBOT SETUP (SIMPLIFIED)")
print("=" * 60)

# ============================================
# STEP 1: Load and chunk documents
# ============================================

print("\n📂 STEP 1: Loading documents...")

documents_dir = "./documents"

if not os.path.exists(documents_dir):
    os.makedirs(documents_dir)
    print(f"✅ Created {documents_dir} folder")

# Create a sample document if no documents exist
def create_sample_document():
    sample_text = """Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Machine learning is a subset of AI that enables systems to learn from data.
Deep learning uses neural networks with many layers to learn complex patterns.
Large Language Models (LLMs) like GPT-4 can generate human-like text.
RAG (Retrieval-Augmented Generation) combines search with LLMs for accurate answers.
Vector databases store embeddings for fast semantic search.
ChromaDB is an open-source vector database for AI applications.
LangChain is a framework for building LLM applications.
Streamlit is a Python library for creating web apps quickly.
Python is a popular programming language for data science and AI."""
    
    with open("documents/sample.txt", "w", encoding='utf-8') as f:
        f.write(sample_text)
    return "documents/sample.txt"

# Collect all documents
documents = []
file_paths = []

for file in os.listdir(documents_dir):
    filepath = os.path.join(documents_dir, file)
    if file.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            documents.append(text)
            file_paths.append(filepath)
            print(f"  ✅ Loaded: {file}")
    elif file.endswith('.pdf'):
        try:
            reader = pypdf.PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)
            file_paths.append(filepath)
            print(f"  ✅ Loaded: {file}")
        except Exception as e:
            print(f"  ⚠️ Could not load PDF {file}: {e}")

if not documents:
    sample_path = create_sample_document()
    with open(sample_path, 'r', encoding='utf-8') as f:
        documents.append(f.read())
        file_paths.append(sample_path)
    print("  ✅ Created and loaded sample document")

print(f"\n✅ Loaded {len(documents)} document(s)")

# ============================================
# STEP 2: Split into chunks
# ============================================

print("\n✂️ STEP 2: Splitting into chunks...")

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks = []
for doc in documents:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

print(f"✅ Created {len(all_chunks)} text chunks")

# ============================================
# STEP 3: Create embeddings and vector database
# ============================================

print("\n🔢 STEP 3: Creating vector database...")

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
try:
    client.delete_collection("documents")
except:
    pass

collection = client.create_collection(name="documents")

# Create embeddings using sentence-transformers
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded")

# Add chunks to collection
for i, chunk in enumerate(all_chunks):
    embedding = embedder.encode(chunk).tolist()
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"source": f"chunk_{i}", "index": i}]
    )

print(f"✅ Added {len(all_chunks)} chunks to vector database")

# ============================================
# STEP 4: Load LLM
# ============================================

print("\n🤖 STEP 4: Loading LLM...")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    llm = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
        temperature=0.7
    )
    print("✅ Local LLM loaded (flan-t5-small)")
except Exception as e:
    print(f"⚠️ Could not load LLM: {e}")
    print("   Using fallback mode without LLM")
    llm = None

# ============================================
# STEP 5: Query function
# ============================================

def query_chatbot(question, top_k=3):
    """Search for relevant chunks and generate answer"""
    
    # Create embedding for question
    question_embedding = embedder.encode(question).tolist()
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    # Get relevant chunks
    relevant_chunks = results['documents'][0]
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the documents.", []
    
    # Combine chunks as context
    context = "\n\n".join(relevant_chunks)
    
    # Create prompt
    prompt = f"""Answer the question based ONLY on the following context. If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer using LLM or fallback
    if llm:
        answer = llm(prompt, max_length=200)[0]['generated_text']
    else:
        answer = f"Based on the documents: {context[:300]}..."
    
    return answer, relevant_chunks

# ============================================
# STEP 6: Test the chatbot
# ============================================

print("\n" + "=" * 60)
print("💬 STEP 6: Testing the Chatbot")
print("=" * 60)

test_questions = [
    "What is RAG?",
    "What is machine learning?",
    "What is ChromaDB?"
]

for question in test_questions:
    print(f"\n❓ Question: {question}")
    answer, sources = query_chatbot(question)
    print(f"💡 Answer: {answer[:200]}...")

print("\n" + "=" * 60)
print("✅ RAG CHATBOT READY!")
print("=" * 60)
print("\n🚀 Next: Run 'streamlit run app.py'")