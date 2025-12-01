import pandas as pd

def generate_targets(df):
    """
    Generate target variables.
    """
    df = df.copy()
    
    # Target 30: 1 if Close[t+30] > Close[t]
    # We use 'close' column which should be QQQ close
    df['target_30'] = (df['close'].shift(-30) > df['close']).astype(int)
    
    # Optional: target_5, target_15
    df['target_5'] = (df['close'].shift(-5) > df['close']).astype(int)
    df['target_15'] = (df['close'].shift(-15) > df['close']).astype(int)
    
    # Remove NaNs at the end caused by shifting
    # We drop based on the largest shift (30)
    # To be safe, we drop rows where any target is NaN (which shouldn't happen with astype(int) 
    # unless the shift goes out of bounds, but shift produces NaNs which astype(int) might handle weirdly if not careful with fillna? 
    # Actually shift produces NaNs, astype(int) on NaN fails or converts to something else? 
    # Wait, (NaN > float) is False. So it becomes 0. That's dangerous!
    # We must drop the last 30 rows explicitly BEFORE calculating or be careful.
    
    # Correct approach:
    # The comparison (future > current) returns False if future is NaN.
    # So we must explicitly identify the rows that don't have a valid future value.
    
    valid_index = df.index[:-30]
    df = df.loc[valid_index]
    
    return df
