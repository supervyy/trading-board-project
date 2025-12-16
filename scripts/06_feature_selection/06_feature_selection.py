"""
Feature Selection Helper for QQQ + Top-Tech Projekt

- Lädt das bereits gesplittete TRAIN-Set (train.parquet) aus data/processed
- Nutzt unsere manuell ausgewählten ESSENTIAL_FEATURES
- Berechnet die Pearson-Korrelationen dieser Features mit ALLEN Targets:
    target_5m, target_15m, target_30m
- Gibt für jedes Target die sortierten Korrelationen aus
- Erstellt zusätzlich eine Übersichtstabelle (Feature vs. alle 3 Targets)

Benutzung:
- Vom Projekt-Root aus ausführen, z.B.:
    python scripts/05_post_split_prep/05_feature_selection_corr.py
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
ESSENTIAL_FEATURES = [
    # 1) QQQ Core (Trend, Momentum, Volumen/Volatilität, Microstructure)
    "ema_diff",              # Trend (kurz vs. mittel)
    "return_5",              # Kurzfrist-Momentum QQQ
    "realized_vol_10",       # lokale Volatilität
    "volume_norm",           # Volumen relativ zur Vergangenheit
    "volume_acceleration",   # Änderung Volumenintensität
    "bid_ask_spread_proxy",  # Liquiditäts-/Stress-Proxy

    # 2) Synchronous Tech-Momentum (alle Top-Techs gleichberechtigt)
    "NVDA_return_5",
    "AAPL_return_5",
    "MSFT_return_5",
    "GOOGL_return_5",
    "AMZN_return_5",

    # 3) Cross-Asset / Tech-Breadth
    "tech_unanimity",        # Anteil Techs, die wie QQQ laufen
    "momentum_spread_5",     # Streuung der Tech-Returns
    "max_divergence",        # stärkste Abweichung einer Tech-Aktie von QQQ
    "relative_strength",     # QQQ relativ zu Tech-Sektor

    # 4) Regime-Features
    "high_vol_regime",       # Volatilitäts-Regime
    "low_corr_regime",       # Korrelation-Regime
    "overextended_up",       # überdehnt nach oben (Trend + Mean-Reversion)
    "overextended_down",     # überdehnt nach unten

    # 5) Korrelation (NVDA als eine von mehreren Techs)
    "corr_QQQ_NVDA_15",      # Kopplung QQQ–NVDA als Regime-Info
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
x_dim = np.load(PROCESSED_PATH / "X_train_scaled.npy").shape[1]
if len(feature_cols) != x_dim:
    raise ValueError(f"Feature count mismatch: ESSENTIAL_FEATURES={len(feature_cols)} vs X_train_scaled has {x_dim}")

out_path = PROJECT_ROOT / "models" / "feed_forward" / "features_selected.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(feature_cols), encoding="utf-8")

print(f"✅ Saved features_selected.txt ({len(feature_cols)} features) -> {out_path}")

