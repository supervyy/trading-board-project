import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_feature_and_target_cols(df: pd.DataFrame):
    """
    Identifies feature and target columns.
    """
    target_cols = [c for c in df.columns if c.startswith("target_")]
    main_target = "target_30m"
    
    # Features are all columns except timestamp (if present) and targets
    feature_cols = [c for c in df.columns if c != "timestamp" and c not in target_cols]
    
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
