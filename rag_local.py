"""
Minimal Local Retrieval-Augmented Generation (RAG) Demo
-------------------------------------------------------
This script:
1. Reads a text file (docs.txt)
2. Splits into chunks
3. Builds embeddings with SentenceTransformers
4. Indexes embeddings in FAISS (or sklearn fallback)
5. Retrieves top-k relevant chunks for a query
6. Uses a small local Hugging Face model (Flan-T5) to generate an answer

Run: python rag_local.py
"""

import os
import re
import sys
import numpy as np

USE_FAISS = True
try:
    import faiss
except ImportError:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

from sentence_transformers import SentenceTransformer
from transformers import pipeline

def load_text(file_path="docs.txt"):
    """Load text from file."""
    if not os.path.exists(file_path):
        sys.exit("docs.txt not found. Create it and try again.")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def split_sentences(text):
    """Naive sentence splitter based on punctuation."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def build_index(embeddings):
    """Build FAISS or sklearn index depending on availability."""
    if USE_FAISS:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        return ("faiss", index)
    else:
        nn = NearestNeighbors(metric="cosine", algorithm="auto")
        nn.fit(embeddings)
        return ("sklearn", nn)

def search(index_tuple, query_vec, k, chunks):
    """Search index for most relevant chunks."""
    backend, index = index_tuple
    k = min(k, len(chunks))
    if backend == "faiss":
        _, I = index.search(query_vec.astype(np.float32), k)
        return [chunks[i] for i in I[0]]
    else:
        _, I = index.kneighbors(query_vec, n_neighbors=k)
        return [chunks[i] for i in I[0]]

def main():
    
    # 1. Load knowledge base
    text = load_text("docs.txt")
    chunks = split_sentences(text)
    print(f"Loaded {len(chunks)} text chunks.")

    # 2. Load embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # 3. Build index
    index = build_index(embeddings)
    backend = index[0]
    print(f"Built index with backend: {backend}")

    # 4. Load generation model
    print("Loading generator (google/flan-t5-small)...")
    generator = pipeline("text2text-generation", model="google/flan-t5-small")

    # 5. Query loop
    while True:
        query = input("\nEnter a query (or 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        q_vec = embedder.encode([query], convert_to_numpy=True)
        retrieved = search(index, q_vec, k=3, chunks=chunks)

        print("\nTop context:")
        for i, c in enumerate(retrieved, 1):
            print(f"{i}. {c}")

        context = " ".join(retrieved)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say 'I don’t know.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        result = generator(prompt, max_new_tokens=100, do_sample=False)
        answer = result[0]["generated_text"].strip()
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()
