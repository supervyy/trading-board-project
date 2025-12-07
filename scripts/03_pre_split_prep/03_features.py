import pandas as pd
import numpy as np

def engineer_qqq_features(df):
    """
    Calculate QQQ core + context features.
    """
    df = df.copy()
    close = df["close"]
    volume = df["volume"]
    vwap = df["vwap"]

    # --- klassische Basis ---
    # EMAs
    df["ema_5"] = close.ewm(span=5, adjust=False).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_diff"] = df["ema_5"] - df["ema_20"]

    # Returns (1-Min liegt implizit in pct_change(), hier 5/15/30 Min)
    for w in [5, 15, 30]:
        df[f"return_{w}"] = close.pct_change(w)

    # Realized Volatility (10-Min Std der 1-Min-Returns)
    df["realized_vol_10"] = close.pct_change().rolling(10).std()

    # Normalized Volume (im Verhältnis zu letzten 60 Min)
    df["volume_norm"] = volume / volume.rolling(60).mean()

    # Normalized VWAP
    df["vwap_norm"] = vwap / close

    # --- neue Volume-Features ---
    # Beschleunigung des Volumens (Trend im Volumen)
    df["volume_acceleration"] = df["volume_norm"].diff(5)

    # Volume + Richtung (viel Volumen in Trendrichtung)
    df["volume_return_alignment"] = np.sign(df["return_5"]) * df["volume_norm"]

    # Volume-Spike (stark überdurchschnittliches Volumen)
    df["volume_spike"] = (df["volume_norm"] > 2.0).astype(int)

    # --- Order-Flow / Liquidity Proxies ---
    if "high" in df.columns and "low" in df.columns:
        # Intraday-Range als Spread-Proxy
        df["bid_ask_spread_proxy"] = (df["high"] - df["low"]) / df["close"]

        # Efficiency Ratio: wie „gerichtet“ ist die Bewegung?
        high_5 = df["high"].rolling(5).max()
        low_5 = df["low"].rolling(5).min()
        rng = (high_5 - low_5).replace(0, np.nan)
        df["efficiency_ratio"] = df["close"].diff(5).abs() / rng

    # --- Mean-Reversion-Signale ---
    df["overextended_up"] = (df["close"] > df["ema_20"] * 1.005).astype(int)
    df["overextended_down"] = (df["close"] < df["ema_20"] * 0.995).astype(int)

    ret5 = df["return_5"]
    roll_mean = ret5.rolling(10).mean()
    roll_std = ret5.rolling(10).std().replace(0, np.nan)

    # RS numerisch stabil berechnen und begrenzen
    rs_safe = roll_mean / roll_std
    rs_safe = rs_safe.clip(lower=-10, upper=10)

    df["rsi_proxy"] = 100 - 100 / (1 + rs_safe)
    df["rsi_proxy"] = df["rsi_proxy"].clip(0, 100)

    # --- Market-Regime (nur Volatilität, Korrelation kommt in cross-asset) ---
    vol = df["realized_vol_10"]
    df["high_vol_regime"] = (
        vol > vol.rolling(100, min_periods=50).quantile(0.8)
    ).astype(int)

    # --- Zeit-Features (Intraday-Patterns) ---
    if isinstance(df.index, pd.DatetimeIndex):
        df["minute_of_day"] = df.index.hour * 60 + df.index.minute
        df["is_9_30_10_00"] = (
            (df.index.hour == 9) & (df.index.minute >= 30) & (df.index.minute < 60)
        ).astype(int)
        df["is_15_30_16_00"] = (
            (df.index.hour == 15) & (df.index.minute >= 30)
        ).astype(int)

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

def engineer_cross_asset_features(df_final: pd.DataFrame, tech_symbols):
    """
    Calculate cross-asset features on the synchronized dataframe.
    df_final enthält bereits QQQ-Features + Tech-Features.
    """
    df_final = df_final.copy()

    # 1. Rolling Correlation (15m) QQQ vs. jede Tech-Aktie
    for sym in tech_symbols:
        col_ret = f"{sym}_return_5"
        if col_ret in df_final.columns:
            col_name = f"corr_QQQ_{sym}_15"
            df_final[col_name] = df_final["return_5"].rolling(15).corr(
                df_final[col_ret]
            )

    # 2. Relative Strength: QQQ vs. Durchschnitt der Tech-Returns
    tech_return_cols = [f"{sym}_return_5" for sym in tech_symbols
                        if f"{sym}_return_5" in df_final.columns]
    if tech_return_cols:
        avg_tech_return = df_final[tech_return_cols].mean(axis=1)
        df_final["relative_strength"] = df_final["return_5"] - avg_tech_return

        # 3. Momentum-Leader: welche Tech-Aktie hat aktuell den höchsten Return?
        df_final["momentum_leader"] = df_final[tech_return_cols].idxmax(axis=1)
        df_final["momentum_leader"] = (
            df_final["momentum_leader"]
            .apply(lambda x: x.split("_")[0] if pd.notna(x) else np.nan)
            .astype("category")
            .cat.codes
        )

        # 4. Divergenzen: Tech-Return minus QQQ-Return (5-Min)
        for sym in tech_symbols:
            col_ret = f"{sym}_return_5"
            if col_ret in df_final.columns:
                df_final[f"divergence_{sym}_QQQ_5"] = (
                    df_final[col_ret] - df_final["return_5"]
                )

        # 5. Momentum-Spread: wie weit liegen die Tech-Returns auseinander?
        df_final["momentum_spread_5"] = df_final[tech_return_cols].std(axis=1)

        # 6. Tech-Unanimity: Anteil der Tech-Aktien, die die gleiche Richtung
        # wie QQQ haben (alle hoch / alle runter = 1.0, gemischt = <1.0)
        tech_signs = np.sign(df_final[tech_return_cols])
        sign_qqq = np.sign(df_final["return_5"])
        df_final["tech_unanimity"] = (
            tech_signs.eq(sign_qqq, axis=0).sum(axis=1) / len(tech_return_cols)
        )

        # 7. Maximaler Divergenz-Betrag über alle Tech-Aktien
        divergences = df_final[tech_return_cols].sub(df_final["return_5"], axis=0)
        df_final["max_divergence"] = divergences.abs().max(axis=1)

    # 8. NVDA-spezifische Volume-Anomalie (falls vorhanden)
    if "NVDA_volume_norm" in df_final.columns:
        df_final["nvda_volume_anomaly"] = df_final["NVDA_volume_norm"]

    # 9. Low-Correlation-Regime auf Basis NVDA (falls vorhanden)
    if "corr_QQQ_NVDA_15" in df_final.columns:
        df_final["low_corr_regime"] = (
            df_final["corr_QQQ_NVDA_15"] < 0.5
        ).astype(int)

    return df_final


def clean_extreme_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    SAFE outlier handling:
    - clippt extreme Werte, löscht aber KEINE Zeilen.
    - vermeidet Look-Ahead Bias.
    """
    df = df.copy()

    # 1. Returns clippen (±3 % für 5-Min-Returns / ähnliche Skalen)
    return_cols = [
        col for col in df.columns
        if "return" in col and "target" not in col
    ]
    for col in return_cols:
        df[col] = df[col].clip(lower=-0.03, upper=0.03)

    # 2. Volume-Normalisierung clippen (0.05x bis 5x durchschnittliches Volumen)
    volume_cols = [col for col in df.columns if "volume_norm" in col]
    for col in volume_cols:
        df[col] = df[col].clip(lower=0.05, upper=5.0)

    # 3. Realized Volatility: 0 → NaN (numerische Artefakte)
    if "realized_vol_10" in df.columns:
        df["realized_vol_10"] = df["realized_vol_10"].replace(0, np.nan)

    return df



def handle_missing_data(df):
    """
    Handle missing values after outlier removal.
    """
    df = df.copy()
    
    # Forward fill only (no look-ahead bias)
    df = df.ffill()
    
    # Drop rows that still have NaNs (e.g., at the beginning)
    df = df.dropna()
    
    return df