from src.preprocessing import load_data
from src.chunker import create_chunks
from src.store import init_vector_store, store_chunks
from src.search import query_vector_store
from src.embedding import get_embedding

def main():
    df = load_data()
    print(f"Loaded {len(df)} movies.\n")

    chunks = create_chunks(df)
    print(f"Total chunks created: {len(chunks)}\n")

    collection = init_vector_store()
    store_chunks(chunks, collection, get_embedding)

    print("\nVector store ready! Ask questions about movies.\n")

    while True:
        query = input("Your question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        results = query_vector_store(collection, query, get_embedding, top_k=1)

        if not results["documents"][0]:
            print("Sorry, no result found.\n")
            continue

        doc = results["documents"][0][0]
        meta = results["metadatas"][0][0]

        print("\nAnswer:\n")

        q = query.lower()

        if "popularity" in q:
            value = meta.get("popularity")
            if value is not None:
                print(f"{meta.get('title')} has a popularity score of {value}\n")
            else:
                print("Popularity data not available.\n")

        elif "vote average" in q:
            value = meta.get("vote_average")
            if value is not None:
                print(f"{meta.get('title')} has a vote average of {value}\n")
            else:
                print("Vote average not available.\n")

        elif "vote count" in q:
            value = meta.get("vote_count")
            if value is not None:
                print(f"{meta.get('title')} has a vote count of {value}\n")
            else:
                print("Vote count not available.\n")

        elif "release date" in q:
            value = meta.get("release_date")
            if value:
                print(f"{meta.get('title')} was released on {value}\n")
            else:
                print("Release date not available.\n")

        elif "overview" in q:
            print(doc + "\n")

        else:
            print("Sorry, I can only answer questions based on the dataset fields.\n")

if __name__ == "__main__":
    main()
