from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def create_chunks(df):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for _, row in df.iterrows():
        overview = row["overview"]
        if not isinstance(overview, str) or not overview.strip():
            continue

        for chunk in splitter.split_text(overview):
            chunks.append({
                "text": chunk,
                "title": row["title"],
                "popularity": row["popularity"],
                "vote_average": row["vote_average"],
                "vote_count": row["vote_count"],
                "release_date": row["release_date"]
            })

    print(f"Total chunks created: {len(chunks)}")
    return chunks[:50] 
