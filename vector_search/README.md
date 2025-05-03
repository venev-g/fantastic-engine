# 🔍 Vector Search Examples

This folder contains examples of vector search implementations using different techniques.

## 📊 Basic Vector Search (`vec.ipynb`)

A simple demonstration of text embedding and vector search using Sentence Transformers and FAISS.

### Features
- 🧠 Text embedding with Sentence Transformers
- 🔢 Vector visualization and exploration
- 🔎 Similarity search with FAISS
- 💡 Simple working examples with small document corpus

### Usage
The notebook demonstrates:
1. Converting text to embeddings
2. Exploring embedding dimensions 
3. Creating a FAISS index for fast similarity search
4. Performing semantic search queries

## 🧩 Hybrid Search with Pinecone (`HybridSearch.ipynb`) 

An implementation of hybrid search (combining dense and sparse vectors) using Pinecone and LangChain.

### Features
- 🔄 Combines dense embeddings with sparse BM25 representations
- ☁️ Uses Pinecone as a vector database
- 🔗 Integrates with LangChain for easier retrieval
- 🧮 TF-IDF/BM25 sparse encoding for keyword matching

### Requirements
- Pinecone API key
- Python libraries: pinecone-client, langchain, sentence-transformers

### Usage
The notebook demonstrates:
1. Setting up a Pinecone index
2. Creating and storing hybrid embeddings
3. Performing hybrid searches that combine semantic and keyword matching
4. Using LangChain's retrieval utilities

## 🚀 Getting Started

To run these notebooks:

1. Install dependencies:
```bash
pip install sentence-transformers faiss-cpu pinecone-client langchain-huggingface pinecone-text
```

2. For the hybrid search notebook, you'll need a Pinecone API key

3. Run the notebooks cell by cell to understand each step

## 📝 Note

These notebooks are educational examples showing different approaches to vector search:
- `vec.ipynb` shows the basics of embeddings and vector search
- `HybridSearch.ipynb` demonstrates more advanced hybrid search techniques combining dense and sparse vectors
