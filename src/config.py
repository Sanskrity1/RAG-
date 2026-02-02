import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

CSV_PATH = os.path.join(DATA_DIR, "top_rated_movies.csv")
TEXT_COLUMN = "overview"
METADATA_COLUMNS = ["title", "release_date", "popularity", "vote_average", "vote_count"]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

EMBEDDING_MODEL = "mxbai-embed-large"  
