import sys
from pathlib import Path

# Add scripts directory to path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
# Add current directory to path to allow importing modules starting with digits (if we kept them)
# or just to import local modules easily.
sys.path.append(str(PROJECT_ROOT / "scripts" / "05_post_split_prep"))


import importlib

# Dynamic imports for modules starting with numbers
data_loader = importlib.import_module("05_data_loader")
scale_data = importlib.import_module("05_scale_data")
build_matrices = importlib.import_module("05_build_matrices")
plot_post_split = importlib.import_module("05_plot_post_split")

def main():
    print("1. Loading data (pre-split)...")
    # Load already split data instead of splitting again
    train_df, val_df, test_df = data_loader.load_split_data()
    
    print("2. Identifying columns...")
    feature_cols, target_cols, main_target = scale_data.get_feature_and_target_cols(train_df)
    print(f"   Main target: {main_target}")
    print(f"   Number of features: {len(feature_cols)}")
    
    print("2b. Saving UNSCALED samples...")
    plot_post_split.save_sample_tables(train_df, val_df, test_df, feature_cols, main_target, suffix="unscaled")
    
    print("3. Fitting scaler on Train...")
    scaler = scale_data.fit_scaler(train_df, feature_cols)
    
    print("4. Applying scaler to Train, Val, Test...")
    train_scaled = scale_data.apply_scaler(scaler, train_df, feature_cols)
    val_scaled = scale_data.apply_scaler(scaler, val_df, feature_cols)
    test_scaled = scale_data.apply_scaler(scaler, test_df, feature_cols)
    
    print("5. Building X/y matrices...")
    X_train, y_train = build_matrices.build_X_y(train_scaled, feature_cols, main_target)
    X_val, y_val = build_matrices.build_X_y(val_scaled, feature_cols, main_target)
    X_test, y_test = build_matrices.build_X_y(test_scaled, feature_cols, main_target)
    
    print("6. Saving matrices...")
    build_matrices.save_matrices(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("7. Generating SCALED samples...")
    plot_post_split.save_sample_tables(train_scaled, val_scaled, test_scaled, feature_cols, main_target, suffix="scaled")
    
    print("\n--- Summary ---")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print("Done.")

if __name__ == "__main__":
    main()
