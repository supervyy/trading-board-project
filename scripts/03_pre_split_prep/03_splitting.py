import pandas as pd
from pathlib import Path

def split_data(df):
    """
    Split data into Train, Validation, and Test sets.
    
    Splitting Logic:
    - Train: First 70% of data
    - Validation: Next 15% of data
    - Test: Last 15% of data
    """
    print("✂️ Splitting Data...")
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # Define split ratios
    train_ratio = 0.70
    val_ratio = 0.15
    # test_ratio = 0.15 (remainder)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split by index position to respect time order
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    # Print stats
    print(f"   Train: {len(train):,} rows ({train.index.min().date()} to {train.index.max().date()})")
    print(f"   Val:   {len(val):,} rows ({val.index.min().date()} to {val.index.max().date()})")
    print(f"   Test:  {len(test):,} rows ({test.index.min().date()} to {test.index.max().date()})")
    
    # Save splits
    processed_path = Path(__file__).resolve().parents[2] / "data" / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    train.to_parquet(processed_path / "train.parquet")
    val.to_parquet(processed_path / "val.parquet")
    test.to_parquet(processed_path / "test.parquet")
    
    print(f"✅ Saved splits to {processed_path}")
    
    return train, val, test
