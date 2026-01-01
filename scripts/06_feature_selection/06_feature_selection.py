"""
Feature Selection Helper for QQQ + Top-Tech Projekt

- Lädt das bereits gesplittete TRAIN-Set (train.parquet) aus data/processed
- Nutzt unsere manuell ausgewählten ESSENTIAL_FEATURES (19 Features, matching Training/Deployment)
- Berechnet die Pearson-Korrelationen dieser Features mit ALLEN Targets:
    target_5m, target_15m, target_30m
- Gibt für jedes Target die sortierten Korrelationen aus
- Erstellt zusätzlich eine Übersichtstabelle (Feature vs. alle 3 Targets)

Benutzung:
- Vom Projekt-Root aus ausführen, z.B.:
    python scripts/06_feature_selection/06_feature_selection.py
"""

import pandas as pd
from pathlib import Path

# -------------------------------------------------
# Pfade & Targets
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

TARGET_COLS = ["target_5m", "target_15m", "target_30m"]

# -------------------------------------------------
# Unsere ausgewählten Features (dein finales Set)
# -------------------------------------------------
# Must match deploy_trading.py and 05_scale_data.py
ESSENTIAL_FEATURES = [
    # 1. Core QQQ
    "ema_diff", "return_5", "realized_vol_10", "volume_norm", "volume_acceleration",
    
    # 2. NVDA Specifics
    "NVDA_return_5", "NVDA_volume_norm", "divergence_NVDA_QQQ_5", 
    "corr_QQQ_NVDA_15",

    # 3. Cross Asset / Tech Breadth
    "relative_strength", "momentum_spread_5", "tech_unanimity", "max_divergence",

    # 4. Regime / Context
    "high_vol_regime", "low_corr_regime", "overextended_up", "overextended_down",
    
    # 5. Microstructure / Time
    "bid_ask_spread_proxy", "is_15_30_16_00"
]

# -------------------------------------------------
# Daten laden
# -------------------------------------------------
print(f"📥 Loading train set from {PROCESSED_PATH} ...")
df = pd.read_parquet(PROCESSED_PATH / "train.parquet")

# Verfügbarkeit prüfen
available_features = [f for f in ESSENTIAL_FEATURES if f in df.columns]
missing_features = set(ESSENTIAL_FEATURES) - set(available_features)
if missing_features:
    print("⚠️ Missing features in train.parquet:", missing_features)

available_targets = [t for t in TARGET_COLS if t in df.columns]
missing_targets = set(TARGET_COLS) - set(available_targets)
if missing_targets:
    print("⚠️ Missing targets in train.parquet:", missing_targets)

cols = available_features + available_targets
df_sub = df[cols].dropna().reset_index(drop=True)

print(f"✅ Data for correlation: {df_sub.shape[0]} rows, "
      f"{len(available_features)} features, {len(available_targets)} targets.")

# -------------------------------------------------
# Korrelationsmatrix berechnen
# -------------------------------------------------
corr_matrix = df_sub.corr()

# 1) Für jedes Target einzeln sortierte Korrelationen ausgeben
for target in available_targets:
    print(f"\n📊 Correlations with {target} (descending by absolute value):")
    sorted_corr = corr_matrix[target].loc[available_features].sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    print(sorted_corr)

# 2) Übersichtstabelle: Feature vs. alle Targets
rows = []
for feat in available_features:
    row = {"feature": feat}
    for target in available_targets:
        row[target] = corr_matrix.loc[feat, target]
    rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df = summary_df.set_index("feature")

print("\n📋 Correlation summary (features vs. all targets):")
print(summary_df)

# Optional: als CSV speichern
reports_path = PROJECT_ROOT / "reports"
reports_path.mkdir(parents=True, exist_ok=True)
out_file = reports_path / "feature_correlations_all_targets.csv"
summary_df.to_csv(out_file)
print(f"\n💾 Saved correlation summary to: {out_file}")

# -----------------------------
# Export Feature List for Deployment (19 features, ordered like training)
# -----------------------------
import numpy as np

# Nimm nur die Features aus ESSENTIAL_FEATURES, die wirklich im DF sind
feature_cols = [f for f in ESSENTIAL_FEATURES if f in df.columns]

# Optionaler Safety-Check gegen X_train_scaled.npy (sollte 19 sein)
x_path = PROCESSED_PATH / "X_train_scaled.npy"
if x_path.exists():
    x_dim = np.load(x_path).shape[1]
    if len(feature_cols) != x_dim:
        raise ValueError(f"Feature count mismatch: ESSENTIAL_FEATURES={len(feature_cols)} vs X_train_scaled has {x_dim}\n"
                         "Update ESSENTIAL_FEATURES in this script to match data generation pipeline.")
else:
    print("⚠️ X_train_scaled.npy not found, skipping dimension check.")

out_path = PROJECT_ROOT / "models" / "feed_forward" / "features_selected.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(feature_cols), encoding="utf-8")

print(f"✅ Saved features_selected.txt ({len(feature_cols)} features) -> {out_path}")
