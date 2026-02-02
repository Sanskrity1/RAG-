import os
import chromadb
from chromadb.config import Settings
from .config import VECTOR_STORE_DIR

def init_vector_store():
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    client = chromadb.Client(Settings(
        persist_directory=VECTOR_STORE_DIR,
        anonymized_telemetry=False,
        is_persistent=True
    ))

    collection = client.get_or_create_collection(name="top_rated_movies")
    return collection


def store_chunks(chunks, collection, embedding_func):
    for i, chunk in enumerate(chunks, 1):
        print(f"Embedding chunk {i}/{len(chunks)}...")
        embedding_vector = embedding_func(chunk["text"])
        if embedding_vector is not None:
            collection.add(
                ids=[str(i)],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
                embeddings=[embedding_vector]
            )
    print("All chunks stored. Total vectors:", collection.count())
