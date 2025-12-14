import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
sys.path.append(str(PROJECT_ROOT / "scripts" / "05_post_split_prep"))

import importlib
data_loader   = importlib.import_module("05_data_loader")
scale_data    = importlib.import_module("05_scale_data")
build_matrices = importlib.import_module("05_build_matrices")
plot_post_split = importlib.import_module("05_plot_post_split")


def main():
    print("1. Loading data (pre-split)...")
    train_df, val_df, test_df = data_loader.load_split_data()

    print("2. Identifying columns...")
    feature_cols, target_cols, main_target = scale_data.get_feature_and_target_cols(train_df)
    print(f"   Main target (for samples): {main_target}")
    print(f"   Number of features: {len(feature_cols)}")
    print(f"   Targets: {target_cols}")

    print("3. Fitting scaler on Train...")
    scaler = scale_data.fit_scaler(train_df, feature_cols)
    
    # Save scaler for deployment
    import joblib
    MODELS_PATH = PROJECT_ROOT / "models"
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_PATH / "scaler.pkl")
    print(f"   💾 Scaler saved to {MODELS_PATH / 'scaler.pkl'}")

    print("4. Applying scaler to Train, Val, Test...")
    train_scaled = scale_data.apply_scaler(scaler, train_df, feature_cols)
    val_scaled   = scale_data.apply_scaler(scaler, val_df, feature_cols)
    test_scaled  = scale_data.apply_scaler(scaler, test_df, feature_cols)

    # Nach dem Skalieren
    def check_scaling(df_scaled, feature_cols, dataset_name):
        print(f"\nChecking {dataset_name} scaling:")
        for i, col in enumerate(feature_cols[:3]):  # Nur erste 3 Features checken
            mean_val = df_scaled[col].mean()
            std_val = df_scaled[col].std()
            print(f"  {col}: mean={mean_val:.3f}, std={std_val:.3f}")
    check_scaling(train_scaled, feature_cols, "Train")
    check_scaling(val_scaled, feature_cols, "Val")
    check_scaling(test_scaled, feature_cols, "Test")

    print("5. Building X/y matrices (all targets)...")
    X_train, y_train = build_matrices.build_X_y(train_scaled, feature_cols, target_cols)
    X_val,   y_val   = build_matrices.build_X_y(val_scaled,   feature_cols, target_cols)
    X_test,  y_test  = build_matrices.build_X_y(test_scaled,  feature_cols, target_cols)

    print("6. Saving matrices...")
    build_matrices.save_matrices(X_train, y_train, X_val, y_val, X_test, y_test)

    print("7. Generating UN SCALED samples...")
    plot_post_split.save_sample_tables(
    train_df,
    val_df,
    test_df,
    feature_cols,
    target_cols,          # <-- alle 3 Targets
    suffix="unscaled",
    )
    print("7. Generating SCALED samples...")
    plot_post_split.save_sample_tables(
    train_scaled,
    val_scaled,
    test_scaled,
    feature_cols,
    target_cols,          # <-- alle 3 Targets
    suffix="scaled",
    )


    print("\n--- Summary ---")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    print(f"X_train shape: {X_train.shape}")  # (n, n_features)
    print(f"y_train shape: {y_train.shape}")  # (n, 3) -> alle Targets
    print("Done.")

if __name__ == "__main__":
    main()
