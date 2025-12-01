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

# Reload modules to ensure latest changes are picked up if run interactively
importlib.reload(features)
importlib.reload(targets)
importlib.reload(plots)
importlib.reload(reporting)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
TECH_SYMBOLS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]

def load_and_sync_data():
    """
    Load raw data, sync timestamps, and filter market hours.
    """
    dfs = []
    print("⏳ Loading data...")
    for sym in SYMBOLS:
        path = DATA_PATH / f"{sym}_1m.parquet"
        try:
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()
            
            # Timezone handling
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("US/Eastern")
            
            # Filter Market Hours
            df = df.between_time("09:30", "16:00")
            
            dfs.append((sym, df))
            print(f"   ✅ {sym}: {len(df):,} rows")
        except Exception as e:
            print(f"   ❌ {sym}: Failed to load - {e}")
            return None
    return dfs

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
            cols = ["close", "ema_5", "ema_10", "ema_20", "ema_diff", 
                    "return_5", "return_15", "return_30", 
                    "realized_vol_10", "volume_norm", "vwap_norm"]
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
    df_final = processed_dfs[0] # QQQ
    for i in range(1, len(processed_dfs)):
        df_final = df_final.join(processed_dfs[i], how="inner")
        
    # 4. Cross-Asset Features
    print("🔄 Calculating Cross-Asset Features...")
    df_final = features.engineer_cross_asset_features(df_final, TECH_SYMBOLS)
    
    # 5. Targets
    print("🎯 Generating Targets...")
    df_final = targets.generate_targets(df_final)
    
    print("\n🔍 DEBUG - VOR Cleaning:")
    print(f"   Shape: {df_final.shape}")
    print(f"   Sample return_5 values:")
    sample = df_final['return_5'].head(10)
    for i, val in enumerate(sample):
        print(f"      Row {i}: {val:.6f}")
    print(f"   Stats - return_5: mean={df_final['return_5'].mean():.6f}, std={df_final['return_5'].std():.6f}")
    
    print("🧹 Cleaning Data (Outlier Removal)...")
    df_final = features.clean_extreme_outliers(df_final)
    df_final = features.handle_missing_data(df_final)
    
    # === DEBUG: Prüfe Values NACH Cleaning ===
    print("\n🔍 DEBUG - NACH Cleaning:")
    print(f"   Shape: {df_final.shape}")
    print(f"   Sample return_5 values:")
    sample = df_final['return_5'].head(10)
    for i, val in enumerate(sample):
        print(f"      Row {i}: {val:.6f}")
    print(f"   Stats - return_5: mean={df_final['return_5'].mean():.6f}, std={df_final['return_5'].std():.6f}")
    
    # 6. Plots
    print("📊 Generating Plots...")
    plots.plot_ema(df_final)
    plots.plot_rolling_corr(df_final)
    plots.plot_target_distribution(df_final)
    plots.plot_feature_target_correlation(df_final)
    plots.plot_scatter_returns(df_final)
    
    # 7. Save
    output_file = PROCESSED_PATH / "pre_split_data.parquet"
    df_final.to_parquet(output_file)
    print(f"✅ Saved processed data to: {output_file}")
    print(f"   Final Shape: {df_final.shape}")
    
    # 8. Reporting
    print("📝 Generating Reports...")
    reporting.save_sample_table(df_final)
    reporting.save_feature_stats(df_final)

if __name__ == "__main__":
    main()
