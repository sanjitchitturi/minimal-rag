# Minimal Local RAG Demo

A simple **Retrieval-Augmented Generation (RAG)** system implemented in Python.  
Runs **fully locally** with Hugging Face models — **no API keys required**.  
Perfect for learning the fundamentals of embeddings, vector search, and RAG pipelines.  

---

## Features
- Indexes a text file (`docs.txt`) into embeddings  
- Retrieves the most relevant chunks for a query  
- Generates grounded answers using [`google/flan-t5-small`](https://huggingface.co/google/flan-t5-small)  
- Uses FAISS for fast similarity search (with sklearn fallback)  
- Runs locally on CPU (Mac, Linux, Windows)  

---

## Project Structure
```

local-rag-demo/
│── docs.txt
│── rag\_local.py
│── requirements.txt
│── README.md 

````

---

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/snajitchitturi/local-rag-demo.git
cd local-rag-demo

# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
````

---

## Usage

1. Put your text into `docs.txt` (use any notes, articles, or documentation).
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
1. Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation.
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
* You can extend this project into:

  * Using smarter chunking (LangChain/LLMs)
  * A Gradio web app (for Hugging Face Spaces)
  * Larger LLMs for better answers

---

## License

This project is open-source under the MIT License.
Feel free to fork, modify, and share!

---

## Acknowledgements

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)

```

---
