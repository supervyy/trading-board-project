import numpy as np
from pathlib import Path

def build_X_y(df, feature_cols, main_target):
    """
    Extracts X (features) and y (target) as numpy arrays.
    """
    X = df[feature_cols].values
    y = df[main_target].values
    return X, y

def save_matrices(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Saves X/y arrays to .npy files in data/processed/.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    output_dir = PROJECT_ROOT / "data" / "processed"
    
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
