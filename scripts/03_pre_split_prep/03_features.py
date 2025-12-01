import pandas as pd
import numpy as np

def engineer_qqq_features(df):
    """
    Calculate features for QQQ.
    """
    df = df.copy()
    close = df['close']
    volume = df['volume']
    vwap = df['vwap']
    
    # EMAs
    df['ema_5'] = close.ewm(span=5, adjust=False).mean()
    df['ema_10'] = close.ewm(span=10, adjust=False).mean()
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_diff'] = df['ema_5'] - df['ema_20']
    
    # Returns
    for w in [5, 15, 30]:
        df[f'return_{w}'] = close.pct_change(w)
        
    # Realized Volatility (10 min rolling std of 1-min returns)
    df['realized_vol_10'] = close.pct_change().rolling(10).std()
    
    # Normalized Volume
    df['volume_norm'] = volume / volume.rolling(60).mean()
    
    # Normalized VWAP
    df['vwap_norm'] = vwap / close
    
    return df

def engineer_tech_features(df, symbol):
    """
    Calculate features for a tech stock (NVDA, AAPL, etc.).
    """
    df = df.copy()
    close = df['close']
    volume = df['volume']
    vwap = df['vwap']
    prefix = f"{symbol}_"
    
    # EMAs
    df[f'{prefix}ema_5'] = close.ewm(span=5, adjust=False).mean()
    df[f'{prefix}ema_20'] = close.ewm(span=20, adjust=False).mean()
    
    # EMA Slope
    df[f'{prefix}ema_slope'] = df[f'{prefix}ema_5'].diff()
    
    # Returns
    df[f'{prefix}return_5'] = close.pct_change(5)
    df[f'{prefix}return_30'] = close.pct_change(30)
    
    # Normalized Features
    df[f'{prefix}volume_norm'] = volume / volume.rolling(60).mean()
    df[f'{prefix}vwap_norm'] = vwap / close
    
    return df

def engineer_cross_asset_features(df_final, tech_symbols):
    """
    Calculate cross-asset features on the synchronized dataframe.
    """
    # 1. Rolling Correlation (15m)
    for sym in tech_symbols:
        # Correlation between QQQ return_5 and Tech return_5 (using 15 period rolling window)
        col_name = f'corr_QQQ_{sym}_15'
        df_final[col_name] = df_final['return_5'].rolling(15).corr(df_final[f'{sym}_return_5'])
        
    # 2. Relative Strength
    # QQQ_return_5 - Mean(Tech_Returns_5)
    tech_return_cols = [f'{sym}_return_5' for sym in tech_symbols]
    avg_tech_return = df_final[tech_return_cols].mean(axis=1)
    df_final['relative_strength'] = df_final['return_5'] - avg_tech_return
    
    # 3. Momentum Leader
    # Which tech stock has the highest return_5?
    df_final['momentum_leader'] = df_final[tech_return_cols].idxmax(axis=1)
    
    # Clean up column names to just symbol
    df_final['momentum_leader'] = df_final['momentum_leader'].apply(
        lambda x: x.split('_')[0] if pd.notna(x) else np.nan
    )
    
    # Encode as categorical codes
    df_final['momentum_leader'] = df_final['momentum_leader'].astype('category').cat.codes
    
    return df_final
