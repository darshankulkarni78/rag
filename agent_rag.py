import os
import requests
from typing import Dict, List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

def main():
    loader = TextLoader("climate_def_cause.txt", encoding="utf-8")
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    docs = splitter.split_documents(document)

    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in docs]
    embeddings = np.array(embed.encode(texts), dtype='float32')

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    tokeniser = AutoTokenizer.from_pretrained("google/flan-t5-large")

    class Llama33Agent:
        def __init__(self, api_key: str = "", model: str = "meta-llama/llama-3.3-8b-instruct:free"):
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.model = model
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            if not self.api_key:
                raise ValueError("API key is required. Set OPENROUTER_API_KEY in your environment or pass explicitly.")

        def _build_headers(self) -> Dict[str, str]:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

        def ask(self, messages: List[Dict[str, str]]) -> str:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024,
                "top_p": 0.95,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }

            try:
                response = requests.post(self.api_url, headers=self._build_headers(), json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                print("Request Error:", e)
                return f"Error communicating with OpenRouter API: {e}"
            except (KeyError, IndexError) as e:
                print("Parsing Error:", e)
                return "Unexpected API response structure."

    agent = Llama33Agent()

    def retriever(query, k=3):
        query_embeddings = np.array(embed.encode([query]), dtype='float32')
        _, ind = index.search(query_embeddings.reshape(1, -1), k)
        return [texts[i] for i in ind[0]]

    def get_context(chunks, tokenizer, max_tokens=450):
        context = ""
        token_count = 0

        for chunk in chunks:
            tokens = tokenizer(chunk, return_tensors="pt")["input_ids"][0]
            if token_count + len(tokens) > max_tokens:
                break
            context += chunk + "\n"
            token_count += len(tokens)

        return context

    def gen_answer(query):
        context_chunks = retriever(query)
        context = get_context(context_chunks, tokeniser)

        prompt = f"""You are a science expert. Based on the following context, answer the user's question clearly and correctly.

Context:
{context}

Question: {query}
Answer:"""

        messages = [
            {"role": "system", "content": "You are a science expert answering questions using given context only."},
            {"role": "user", "content": prompt}
        ]
        return agent.ask(messages)

    query = "explain in short all the matter of climate change, why is this even important?"
    print("Query:", query)
    response = gen_answer(query)
    print("Answer:", response)

if __name__ == "__main__":
    main()
