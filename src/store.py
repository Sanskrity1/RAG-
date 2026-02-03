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

    return client.get_or_create_collection("top_rated_movies")

def store_chunks(chunks, collection, embed_fn):
    for i, chunk in enumerate(chunks):
        vector = embed_fn(chunk["text"])

        collection.add(
            ids=[str(i)],
            documents=[chunk["text"]],
            embeddings=[vector],
            metadatas=[{
                "title": chunk["title"],
                "popularity": chunk["popularity"],
                "vote_average": chunk["vote_average"],
                "vote_count": chunk["vote_count"],
                "release_date": chunk["release_date"]
            }]
        )

    print("All chunks stored. Total vectors:", collection.count())
