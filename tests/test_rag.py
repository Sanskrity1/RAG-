from src.preprocessing import load_data
from src.chunker import create_chunks
from src.store import init_vector_store

def test_pipeline():
    df = load_data()
    assert not df.empty, "CSV loading failed"
    chunks = create_chunks(df)
    assert len(chunks) > 0, "Chunk creation failed"
    collection = init_vector_store()
    print("Test passed! Vector store initialized.")

if __name__ == "__main__":
    test_pipeline()
