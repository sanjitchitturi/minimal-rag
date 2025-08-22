# mini_rag.py
# Minimal Retrieval-Augmented Generation (RAG) demo

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load text file
with open("docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Split into chunks (by sentences)
chunks = text.split(". ")
chunks = [c.strip() for c in chunks if c.strip()]

# 3. Embed chunks
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# 4. Build FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity with normalized vectors
index.add(embeddings)

print(f"Indexed {len(chunks)} chunks from docs.txt")

# 5. Query loop
while True:
    query = input("\nAsk a question (or type 'exit'): ").strip()
    if query.lower() in {"exit", "quit"}:
        break
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, 3)  # top-3 results
    print("\nTop results:")
    for i, score in zip(idxs[0], scores[0]):
        print(f"- (score={score:.3f}) {chunks[i]}")
