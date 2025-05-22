# RAG Bot: Retrieval-Augmented Generation for Science Questions

This project implements a Retrieval-Augmented Generation (RAG) system that answers science-related queries based on provided document context. It uses LangChain for document loading and splitting, SentenceTransformers for embedding, FAISS for vector search, and OpenRouter API with Meta's LLaMA 3.3 model for answer generation.

---

## Features

- Load and split long documents into chunks for efficient retrieval.
- Embed chunks using sentence-transformer embeddings.
- Build a FAISS vector index for fast similarity search.
- Retrieve top-k relevant chunks for a query.
- Generate context-aware answers using Meta LLaMA via OpenRouter API.
- Token-limit context builder using HuggingFace tokenizer.

---

## Setup

1. Clone the repository.
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a .env file in the root directory and add your OpenRouter API key:
   ```bash
   OPENROUTER_API_KEY=your_api_key_here
   ```

5. Place your document file named `climate_def_cause.txt` in the project root.


## Usage

Run the main script:

```bash
python agent_rag.py
```

## Code Structure

- `agent_rag.py`: Main script containing loading, embedding, retrieval, and generation logic.

- `climate_def_cause.txt`: Document source used for retrieval.

- `.env`: Environment variables file storing API key securely.

## Dependencies

- langchain

- sentence-transformers

- faiss-cpu

- transformers

- requests

- python-dotenv

## Notes

- The project uses OpenRouter API for accessing the Meta LLaMA model; ensure you have API access.

- FAISS vector store is built in-memory; large datasets may require persistent storage solutions.

- Context length limited to 450 tokens for API input.
