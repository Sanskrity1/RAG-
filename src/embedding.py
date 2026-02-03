import ollama
from .config import EMBEDDING_MODEL

def get_embedding(text: str):
    return ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=text
    )["embedding"]
