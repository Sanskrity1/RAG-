from src.preprocessing import load_data
from src.chunker import create_chunks
from src.store import init_vector_store, store_chunks
from src.search import query_vector_store
from src.embedding import get_embedding
from thefuzz import process

def find_movie(df, movie_query, threshold=85):
    titles = df["title"].tolist()
    best_match, score = process.extractOne(movie_query, titles)
    if score >= threshold:
        return df[df["title"] == best_match].iloc[0]
    return None

def main():
    df = load_data()
    print(f"Loaded {len(df)} movies.\n")

    chunks = create_chunks(df)
    print(f"Total chunks created: {len(chunks)}\n")

    collection = init_vector_store()
    store_chunks(chunks, collection, get_embedding)

    print("\nAsk questions about the movies.\n")

    while True:
        query = input("Your question (type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        movie_title = None
        for keyword in ["overview of", "popularity of", "vote average of",
                        "vote count of", "release date of"]:
            if keyword in query.lower():
                movie_title = query.lower().split(keyword)[-1].strip()
                field = keyword.replace(" of", "").replace("overview", "overview")
                break
        else:
            movie_title = None
            field = None

        if movie_title:
            movie_row = find_movie(df, movie_title)
            if movie_row is None:
                print(f"Sorry, movie '{movie_title}' not found.\n")
                continue

            if field == "overview":
                print(f"\nOverview of '{movie_row['title']}':\n{movie_row['overview']}\n")
            elif field == "popularity":
                print(f"\n'{movie_row['title']}' popularity: {movie_row.get('popularity', 'N/A')}\n")
            elif field == "vote average":
                print(f"\n'{movie_row['title']}' vote average: {movie_row.get('vote_average', 'N/A')}\n")
            elif field == "vote count":
                print(f"\n'{movie_row['title']}' vote count: {movie_row.get('vote_count', 'N/A')}\n")
            elif field == "release date":
                print(f"\n'{movie_row['title']}' release date: {movie_row.get('release_date', 'N/A')}\n")
            else:
                print("Sorry, I can only answer questions based on the dataset fields.\n")
        else:
            print("Sorry, I can only answer questions based on the dataset fields.\n")

if __name__ == "__main__":
    main()
