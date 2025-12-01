import pandas as pd

def generate_targets(df):
    """
    Generate target variables for [5, 15, 30] minutes.
    """
    df = df.copy()
    
    # Target windows as promised in README
    target_windows = [5, 15, 30]
    
    # Generate all targets
    for window in target_windows:
        # Create target: 1 if Close[t+window] > Close[t], else 0
        df[f'target_{window}'] = (df['close'].shift(-window) > df['close']).astype(int)
    
    # Remove rows at the end that don't have enough future data
    # We use the largest window (30) to determine valid cutoff
    max_window = max(target_windows)
    valid_index = df.index[:-max_window]
    df = df.loc[valid_index]
    
    print(f"✅ Generated targets for windows: {target_windows}")
    print(f"   Target distribution:")
    for window in target_windows:
        target_col = f'target_{window}'
        if target_col in df.columns:
            dist = df[target_col].value_counts(normalize=True)
            print(f"   - {window:2d}min: {dist.get(1, 0):.1%} upward trends")
    
    return df