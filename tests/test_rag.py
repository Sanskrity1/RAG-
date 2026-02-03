from src.preprocessing import load_data
from src.chunker import create_chunks

def test_rag():
    df = load_data()
    chunks = create_chunks(df)
    assert len(chunks) > 0
    print("RAG pipeline test passed!")

if __name__ == "__main__":
    test_rag()
