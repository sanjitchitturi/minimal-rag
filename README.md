# Minimal RAG

A simple **Retrieval Augmented Generation (RAG)** system implemented in Python. Runs **fully locally** with Hugging Face models **no API keys required**.

---

## Features
- Indexes a text file (`docs.txt`) into embeddings. 
- Retrieves the most relevant chunks for a query.  
- Generates grounded answers using [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small).
- Uses FAISS for fast similarity search.  
- Runs locally on CPU.

---

## Project Structure
```
minimal-rag/
│── LICENCE
│── README.md 
│── docs.txt
│── rag_local.py
│── requirements.txt
````

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/sanjitchitturi/minimal-rag.git
cd minimal-rag

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
````
---

## Usage

1. Put your text into `docs.txt`.
2. Run the script:

   ```bash
   python rag_local.py
   ```
3. Ask questions about your document:

   ```
   Enter a query: What is RAG?
   ```
---

## Example Run

```
Loaded 4 text chunks.
Built index with backend: faiss
Loading generator (google/flan-t5-small)...

Enter a query (or 'quit' to exit): What is RAG?

Top context:
1. Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation.
2. It uses embeddings and similarity search to retrieve relevant text chunks from a knowledge base.
3. The retrieved chunks are given to a language model to produce grounded answers.

Answer: RAG is a technique that combines information retrieval with text generation to produce grounded answers.
```
---

## Requirements

* Python 3.8+
* Packages:

  * `sentence-transformers`
  * `transformers`
  * `torch`
  * `faiss-cpu` (or `scikit-learn` as fallback)

First run will automatically download Hugging Face models (\~80MB).

---

## Notes

* Runs entirely on CPU — no GPU required.
* If `faiss-cpu` fails to install, the script automatically falls back to `scikit-learn`.

---

## Acknowledgements

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)

---
