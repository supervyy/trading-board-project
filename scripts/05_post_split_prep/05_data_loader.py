import pandas as pd
from pathlib import Path

def load_split_data():
    """
    Load pre-split data (Train, Val, Test) from data/processed.
    """
    project_root = Path(__file__).resolve().parents[2]
    processed_path = project_root / "data" / "processed"
    
    print(f"Loading split data from {processed_path}...")
    
    try:
        train = pd.read_parquet(processed_path / "train.parquet")
        val = pd.read_parquet(processed_path / "val.parquet")
        test = pd.read_parquet(processed_path / "test.parquet")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find split data in {processed_path}. "
            "Please run scripts/03_pre_split_prep/03_splitting.py first."
        ) from e

    print(f"   Train: {len(train)} rows")
    print(f"   Val:   {len(val)} rows")
    print(f"   Test:  {len(test)} rows")
    
    return train, val, test
