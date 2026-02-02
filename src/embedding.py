import ollama
from .config import EMBEDDING_MODEL

def get_embedding(text):
    try:
        response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
        return response.get("embedding")
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None
