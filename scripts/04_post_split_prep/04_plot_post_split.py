import pandas as pd
from pathlib import Path

def save_sample_tables(train_df, val_df, test_df, feature_cols, main_target):
    """
    Saves 5 random samples from each split to CSV.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    output_dir = PROJECT_ROOT / "data" / "processed"
    
    # Show all columns in the sample to avoid confusion
    cols_to_show = list(train_df.columns)
    # Ensure timestamp is included if it's in the index later
    if "timestamp" not in cols_to_show:
        cols_to_show = ["timestamp"] + cols_to_show
    
    # Helper to prepare sample
    def prepare_and_save(df, name):
        if len(df) == 0:
            return
            
        sample = df.sample(min(5, len(df)), random_state=42)
        
        # If timestamp is not a column, try to reset index
        if "timestamp" not in sample.columns:
            sample = sample.reset_index()
            # If index didn't have a name, it might be 'index' now. Rename if needed or just use it.
            if "timestamp" not in sample.columns and "index" in sample.columns:
                sample = sample.rename(columns={"index": "timestamp"})
        
        # Add timestamp to cols if present
        current_cols = cols_to_show[:]
        if "timestamp" in sample.columns:
            current_cols = ["timestamp"] + current_cols
            
        # Ensure all cols exist
        final_cols = [c for c in current_cols if c in sample.columns]
        
        sample[final_cols].to_csv(output_dir / f"sample_{name}_post_split.csv", index=False)

    prepare_and_save(train_df, "train")
    prepare_and_save(val_df, "val")
    prepare_and_save(test_df, "test")
