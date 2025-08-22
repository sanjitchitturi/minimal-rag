# Minimal RAG Demo

This is a **super simple Retrieval-Augmented Generation (RAG) demo** written in one Python file.  

This repository contains a minimal Retrieval-Augmented Generation (RAG) demo implemented in a single Python script. It reads a docs.txt file, splits the text into chunks, encodes chunks using a Sentence-Transformer model, indexes the embeddings in FAISS, and exposes a simple interactive query loop that returns the top-k most relevant text chunks for any user query. The project is intentionally small and dependency-light so you can run it locally or in Colab and quickly demonstrate understanding of embeddings and vector retrieval workflows.

It shows how to:
- Index a `.txt` file into a FAISS vector store  
- Retrieve the most relevant chunks for a query  
- Build the foundation of a RAG pipeline in under 50 lines of code  

---

## How it works
1. Read text from `docs.txt`  
2. Split text into small chunks (sentences)  
3. Create embeddings with [`sentence-transformers`](https://www.sbert.net/)  
4. Store embeddings in a [FAISS](https://github.com/facebookresearch/faiss) index  
5. Search the index with your query and return the most relevant chunks  

---

## Installation

### Clone repo
```bash
git clone https://github.com/sanjitchitturi/minimal-rag-demo.git
cd minimal-rag-demo
