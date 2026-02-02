from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP, TEXT_COLUMN, METADATA_COLUMNS

def create_chunks(df):
    """
    Split movie overviews into chunks and attach metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []
    for _, row in df.iterrows():
        text = row[TEXT_COLUMN]
        if not isinstance(text, str) or not text.strip():
            continue  

        metadata = {col: row[col] for col in METADATA_COLUMNS}
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": metadata
            })

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks[:50]  
