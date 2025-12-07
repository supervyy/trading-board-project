import pandas as pd
from pathlib import Path

def save_sample_tables(train_df, val_df, test_df, feature_cols, main_target, suffix="post_split"):
    """
    Saves 5 random samples from each split to CSV, SEPARATING X and y.
    suffix: String to append to filename (e.g. 'scaled' or 'unscaled')
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    output_dir = PROJECT_ROOT / "data" / "processed"
    
    # Debug: Print received target
    print(f"   [DEBUG] Splitting samples for X/y. Target: {main_target}")

    # Helper to prepare sample
    def prepare_and_save(df, name):
        if len(df) == 0:
            return
            
        sample = df.sample(min(5, len(df)), random_state=42)
        
        # Ensure timestamp is available if possible (for X)
        if "timestamp" not in sample.columns:
            sample = sample.reset_index()
            if "timestamp" not in sample.columns and "index" in sample.columns:
                sample = sample.rename(columns={"index": "timestamp"})

        # --- Save X (Features) ---
        # Features + Timestamp (if present)
        x_cols = feature_cols[:]
        if "timestamp" in sample.columns:
            x_cols = ["timestamp"] + x_cols
            
        final_x_cols = [c for c in x_cols if c in sample.columns]
        
        filename_X = f"sample_X_{name}_{suffix}.csv"
        sample[final_x_cols].to_csv(output_dir / filename_X, index=False)
        
        # --- Save y (Target) ---
        # Target + Timestamp (for reference)
        y_cols = [main_target]
        if "timestamp" in sample.columns:
            y_cols = ["timestamp"] + y_cols
            
        final_y_cols = [c for c in y_cols if c in sample.columns]

        if not final_y_cols:
             print(f"   [WARNING] Target {main_target} not found in {name} sample!")
        else:
            filename_y = f"sample_y_{name}_{suffix}.csv"
            sample[final_y_cols].to_csv(output_dir / filename_y, index=False)

    prepare_and_save(train_df, "train")
    prepare_and_save(val_df, "val")
    prepare_and_save(test_df, "test")
