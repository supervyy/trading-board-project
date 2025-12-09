import numpy as np
from pathlib import Path

def build_X_y(df, feature_cols, target_cols):
    """
    Extracts X (features) and y (all targets) as numpy arrays.

    X: shape (n_samples, n_features)
    y: shape (n_samples, n_targets)  # z.B. target_5m, target_15m, target_30m
    """
    X = df[feature_cols].values
    y = df[target_cols].values      # alle Targets gleichzeitig
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

def save_matrices(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Saves X/y arrays to .npy files in data/processed/.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    output_dir = PROJECT_ROOT / "data" / "processed"
    
    # Features (SCALED)
    np.save(output_dir / "X_train_scaled.npy", X_train)
    np.save(output_dir / "X_val_scaled.npy", X_val)
    np.save(output_dir / "X_test_scaled.npy", X_test)
    
    # Targets (RAW/UNSCALED)
    np.save(output_dir / "y_train_raw.npy", y_train)
    np.save(output_dir / "y_val_raw.npy", y_val)
    np.save(output_dir / "y_test_raw.npy", y_test)
    
    print(f"Saved scaled features and raw targets to {output_dir}")