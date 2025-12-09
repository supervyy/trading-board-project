# scripts/05_post_split_prep/05_plot_post_split.py

"""
Step 5 – Post-Split Sample Tables

Erzeugt kleine Beispiel-Tabellen für X und y nach dem Split:

- sample_X_train_unscaled.csv
- sample_X_val_unscaled.csv
- sample_X_test_unscaled.csv
- sample_y_train_unscaled.csv
- sample_y_val_unscaled.csv
- sample_y_test_unscaled.csv

- sample_X_train_scaled.csv
- sample_X_val_scaled.csv
- sample_X_test_scaled.csv

Die Funktion wird von 05_main_post_split.py getrennt für
UNSCALED und SCALED aufgerufen:

    save_sample_tables(train_df, val_df, test_df,
                       feature_cols, target_cols,
                       suffix="unscaled")

    save_sample_tables(train_df_scaled, val_df_scaled, test_df_scaled,
                       feature_cols, target_cols,
                       suffix="scaled")
"""

from pathlib import Path
import pandas as pd


def save_sample_tables(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols,
    target_cols,
    suffix: str = "unscaled",
    sample_size: int = 5000,
) -> None:
    """
    Erzeugt kleine Samples für Train/Val/Test.

    - Für beide Suffixe werden X-Samples geschrieben.
    - y-Samples werden NUR geschrieben, wenn die Target-Spalten
      in den übergebenen DataFrames enthalten sind (also typischerweise
      nur beim 'unscaled'-Call).
    """

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for split_name, df in splits.items():
        if df is None or df.empty:
            continue

        n = min(sample_size, len(df))
        sample = df.sample(n=n, random_state=42).copy()

        # ----------------- X-Samples -----------------
        x_cols = [c for c in feature_cols if c in sample.columns]
        x_sample = sample[x_cols]
        x_out = processed_dir / f"sample_X_{split_name}_{suffix}.csv"
        x_sample.to_csv(x_out, index=False)

        # ----------------- y-Samples (nur wenn Targets vorhanden) ----------
        y_cols = [c for c in target_cols if c in sample.columns]
        if y_cols:
            y_sample = sample[y_cols]
            y_out = processed_dir / f"sample_y_{split_name}_{suffix}.csv"
            y_sample.to_csv(y_out, index=False)

    print(f"✅ Sample tables ({suffix}) gespeichert in {processed_dir}")
