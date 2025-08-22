# Minimal RAG Demo

This is a **super simple Retrieval-Augmented Generation (RAG) demo** written in one Python file.  
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
git clone https://github.com/YOUR-USERNAME/minimal-rag-demo.git
cd minimal-rag-demo
