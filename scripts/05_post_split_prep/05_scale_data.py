import pandas as pd
from sklearn.preprocessing import StandardScaler

ESSENTIAL_FEATURES = [
    # QQQ Core (5)
    "ema_diff",
    "return_5",
    "realized_vol_10",
    "volume_norm",
    "volume_acceleration",

    # NVDA Only (3)
    "NVDA_return_5",
    "NVDA_volume_norm",
    "divergence_NVDA_QQQ_5",

    # Cross-Asset Dynamics (5)
    "corr_QQQ_NVDA_15",
    "relative_strength",
    "momentum_spread_5",
    "tech_unanimity",
    "max_divergence",

    # Market Context (4)
    "high_vol_regime",
    "low_corr_regime",
    "overextended_up",
    "overextended_down",

    # Time / Flow (3)
    "bid_ask_spread_proxy",
    "is_9_30_10_00",
    "is_15_30_16_00",
]


def get_feature_and_target_cols(df: pd.DataFrame):
    """
    Identifies feature and target columns based on our domain-selected
    ESSENTIAL_FEATURES and available columns in the dataframe.
    """
    target_cols = [c for c in df.columns if c.startswith("target_")]
    main_target = "target_30m"

    # Nur Essential Features, die in df wirklich vorhanden sind
    feature_cols = [c for c in ESSENTIAL_FEATURES if c in df.columns]

    return feature_cols, target_cols, main_target

def fit_scaler(train_df: pd.DataFrame, feature_cols: list):
    """
    Fits StandardScaler on training data features.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler

def apply_scaler(scaler, df, feature_cols):
    """
    Applies scaler to dataframe features. Returns a new dataframe.
    """
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])
    return df_scaled
