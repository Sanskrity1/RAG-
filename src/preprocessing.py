import pandas as pd
from .config import CSV_PATH, TEXT_COLUMN

def load_data():
    df = pd.read_csv(CSV_PATH)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"{TEXT_COLUMN} not found in CSV")
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")  
    return df
