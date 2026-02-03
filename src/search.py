def query_vector_store(collection, query, embed_fn, top_k=3):
    vector = embed_fn(query)
    return collection.query(
        query_embeddings=[vector],
        n_results=top_k
    )
