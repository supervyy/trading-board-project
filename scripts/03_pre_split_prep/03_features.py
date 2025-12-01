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
    df[f'{prefix}ema_10'] = close.ewm(span=10, adjust=False).mean()
    df[f'{prefix}ema_20'] = close.ewm(span=20, adjust=False).mean()
    
    # EMA Slope
    df[f'{prefix}ema_slope'] = df[f'{prefix}ema_5'] - df[f'{prefix}ema_20']
    
    # Returns
    df[f'{prefix}return_5'] = close.pct_change(5)
    df[f'{prefix}return_15'] = close.pct_change(15)
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
def clean_extreme_outliers(df):
    """
    Remove unrealistic price moves and fix data quality issues.
    """
    df = df.copy()
    
    # 1. Remove extreme returns (beyond ±5% in 5min is unrealistic)
    return_cols = [col for col in df.columns if 'return' in col and 'target' not in col]
    
    for col in return_cols:
        mask = df[col].between(-0.05, 0.05) | df[col].isna()
        df = df[mask]
    
    # 2. Fix negative volume (impossible values)
    volume_cols = [col for col in df.columns if 'volume_norm' in col]
    for col in volume_cols:
        df[col] = df[col].clip(lower=0.01)  # Minimum 1% of average volume
    
    # 3. Remove extreme volume outliers (beyond 10x average)
    for col in volume_cols:
        mask = df[col] <= 10.0
        df = df[mask]
    
    # 4. Fix realized volatility (should never be 0)
    if 'realized_vol_10' in df.columns:
        df['realized_vol_10'] = df['realized_vol_10'].replace(0, np.nan)
    
    return df

def handle_missing_data(df):
    """
    Handle missing values after outlier removal.
    """
    df = df.copy()
    
    # Forward fill then backward fill
    df = df.ffill().bfill()
    
    # Remove any remaining rows with NaNs
    df = df.dropna()
    
    return df