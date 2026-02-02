from src.preprocessing import load_data
from src.chunker import create_chunks
from src.store import init_vector_store, store_chunks
from src.search import query_vector_store
from src.embedding import get_embedding

def main():
    df = load_data()

    chunks = create_chunks(df)

    collection = init_vector_store()

    store_chunks(chunks, collection, get_embedding)

    print("\nVector store ready! Ask questions about movies.\n")

    while True:
        query = input("Your question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        try:
            results = query_vector_store(collection, query, get_embedding, top_k=1)

            if results['documents'][0]:
                doc_text = results['documents'][0][0]
                metadata = results['metadatas'][0][0]

                print("\nTop matching movie overview:\n")
                print(f"Overview: {doc_text}")
                print("\nMetadata:")
                for key, value in metadata.items():
                    print(f"{key}: {value}")
                print("\n")
            else:
                print("No matching movie found.\n")

        except Exception as e:
            print(f"Error generating answer: {e}\nAnswer:\nCould not generate an answer.\n")

if __name__ == "__main__":
    main()
