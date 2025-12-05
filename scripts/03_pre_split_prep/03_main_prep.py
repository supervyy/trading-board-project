import pandas as pd
from pathlib import Path
import sys

# Add current directory to path to import modules
sys.path.append(str(Path(__file__).parent))

import importlib

# Dynamic imports for modules starting with numbers
features = importlib.import_module("03_features")
targets = importlib.import_module("03_targets")
plots = importlib.import_module("03_plot_features")
reporting = importlib.import_module("03_reporting")
splitting = importlib.import_module("03_splitting")

# Reload modules to ensure latest changes are picked up if run interactively
importlib.reload(features)
importlib.reload(targets)
importlib.reload(plots)
importlib.reload(reporting)
importlib.reload(splitting)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
TECH_SYMBOLS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]


def load_and_sync_data():
    """
    Load raw data and sync timestamps across all symbols.
    """
    dfs = []
    print("⏳ Loading data...")
    for sym in SYMBOLS:
        path = DATA_PATH / f"{sym}_1m.parquet"
        try:
            df = pd.read_parquet(path)

            # Timestamp in Datetime konvertieren und als Index setzen
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()

            # KEIN Timezone-Handling, KEIN between_time mehr!
            dfs.append((sym, df))
            print(f"   ✅ Loaded {sym}: {df.shape[0]:,} rows")
        except FileNotFoundError:
            print(f"   ⚠️ File not found for symbol: {sym} ({path})")
        except Exception as e:
            print(f"   ⚠️ Error loading {sym}: {e}")

    if not dfs:
        print("❌ No data loaded. Please check your raw data files.")
        return []

    print(f"✅ Successfully loaded data for {len(dfs)} symbols.")

    # Align all symbols on the same timestamp index (inner join)
    print("🕒 Aligning timestamps across all symbols...")
    common_index = dfs[0][1].index
    for _, df in dfs[1:]:
        common_index = common_index.intersection(df.index)

    aligned_dfs = []
    for sym, df in dfs:
        df = df.loc[common_index]
        aligned_dfs.append((sym, df))
        print(f"   {sym}: {df.shape[0]:,} aligned rows")

    print("✅ Timestamps aligned across all symbols.")
    return aligned_dfs



def main():
    # 1. Load Data
    dfs_raw = load_and_sync_data()
    if not dfs_raw:
        return

    # 2. Feature Engineering
    print("🛠️ Engineering Features...")
    processed_dfs = []

    for sym, df in dfs_raw:
        if sym == "QQQ":
            df_eng = features.engineer_qqq_features(df)
            # Keep only relevant columns + close for target
            cols = [
                "close",
                "ema_5",
                "ema_10",
                "ema_20",
                "ema_diff",
                "return_5",
                "return_15",
                "return_30",
                "realized_vol_10",
                "volume_norm",
                "vwap_norm",
            ]
            df_eng = df_eng[cols]
        else:
            df_eng = features.engineer_tech_features(df, sym)
            # Keep only relevant columns
            prefix = f"{sym}_"
            cols = [c for c in df_eng.columns if c.startswith(prefix)]
            df_eng = df_eng[cols]

        processed_dfs.append(df_eng)

    # 3. Merge
    print("🔗 Merging Data...")
    df_final = processed_dfs[0]  # QQQ
    for i in range(1, len(processed_dfs)):
        df_final = df_final.join(processed_dfs[i], how="inner")

    # 4. Cross-Asset Features
    print("🔄 Calculating Cross-Asset Features...")
    df_final = features.engineer_cross_asset_features(df_final, TECH_SYMBOLS)

    # 5. Targets
    print("🎯 Generating Targets...")
    df_final = targets.generate_targets(df_final)

    # DEBUG vor Cleaning
    print("\n🔍 DEBUG - VOR Cleaning:")
    print(f"   Shape: {df_final.shape}")
    print(f"   Sample return_5 values:")
    sample = df_final["return_5"].head(10)
    for i, val in enumerate(sample):
        print(f"      Row {i}: {val:.6f}")
    print(
        f"   Stats - return_5: mean={df_final['return_5'].mean():.6f}, "
        f"std={df_final['return_5'].std():.6f}"
    )

    # 6. Cleaning
    print("🧹 Cleaning Data (Outlier Removal + Missing Handling)...")
    df_final = features.clean_extreme_outliers(df_final)
    df_final = features.handle_missing_data(df_final)

    # DEBUG nach Cleaning
    print("\n🔍 DEBUG - NACH Cleaning:")
    print(f"   Shape: {df_final.shape}")
    print(f"   Sample return_5 values:")
    sample = df_final["return_5"].head(10)
    for i, val in enumerate(sample):
        print(f"      Row {i}: {val:.6f}")
    print(
        f"   Stats - return_5: mean={df_final['return_5'].mean():.6f}, "
        f"std={df_final['return_5'].std():.6f}"
    )
    # 5b. Feature Statistics
    print("📊 Saving feature descriptive statistics...")
    reporting.save_feature_stats(df_final)
    # 6. Splitting
    print("✂️ Splitting Data...")
    train_df, val_df, test_df = splitting.split_data(df_final)

    # 7. Plots
    print("📊 Generating Plots...")
    plots.plot_ema(df_final)
    plots.plot_rolling_corr(df_final)
    plots.plot_regression_targets_distribution(df_final)
    print("   ...plotting feature correlation on TRAIN set only...")
    plots.plot_feature_target_correlation(train_df)
    plots.plot_scatter_returns(df_final)

    # 8. Save
    output_file = PROCESSED_PATH / "pre_split_data.parquet"
    df_final.to_parquet(output_file)
    print(f"✅ Saved processed data to: {output_file}")
    print(f"   Final Shape: {df_final.shape}")

    # 9. Reporting
    print("📝 Generating Reports...")
    reporting.save_sample_table(df_final)


if __name__ == "__main__":
    main()
