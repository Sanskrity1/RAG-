def query_vector_store(collection, query, embedding_func, top_k=3):
    query_vector = embedding_func(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    return results
