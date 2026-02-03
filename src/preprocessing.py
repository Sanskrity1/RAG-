import pandas as pd
from .config import CSV_PATH

def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.fillna("")
    print(f"Loaded {len(df)} movies.")
    return df
