import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMG_PATH = PROJECT_ROOT / "images" / "data_preparation"
IMG_PATH.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8")


def plot_ema(df):
    """
    Plot EMA(5) vs EMA(20) vs Close Price (for QQQ).
    """
    plt.figure(figsize=(12, 6))
    subset = df.iloc[-300:]
    plt.plot(subset.index, subset["close"], label="Close", alpha=0.6)
    plt.plot(subset.index, subset["ema_5"], label="EMA 5")
    plt.plot(subset.index, subset["ema_20"], label="EMA 20")
    plt.title("QQQ: EMA(5) vs EMA(20) vs Close")
    plt.legend()
    plt.savefig(IMG_PATH / "qqq_ema_structure.png")
    plt.close()


def plot_rolling_corr(df):
    """
    Rolling 15-Min Correlation (QQQ vs NVDA) – Trading Hours Only, ohne Zeitlücken.
    Entspricht deinem 'langen' Plot mit vielen Tagen.
    """
    # 1. Nur Handelszeiten (falls der Index DatetimeIndex ist)
    if isinstance(df.index, pd.DatetimeIndex):
        df_plot = df.between_time("09:30", "16:00").copy()
    else:
        df_plot = df.copy()

    if "corr_QQQ_NVDA_15" not in df_plot.columns:
        print("⚠️ corr_QQQ_NVDA_15 nicht gefunden")
        return

    # 2. Auf die letzten ~20 Handelstage begrenzen (optional, für bessere Lesbarkeit)
    unique_days = sorted(pd.unique(df_plot.index.date))
    if len(unique_days) > 20:
        last_20_days = unique_days[-20:]
        mask = np.isin(df_plot.index.date, last_20_days)
        df_plot = df_plot[mask]

    # 3. Index zurücksetzen, damit die x-Achse lückenlos ist
    df_plot = df_plot.reset_index()

    # Spalte mit Timestamps finden
    ts_col = "timestamp" if "timestamp" in df_plot.columns else "index"

    # 4. Plot
    sns.set_style("darkgrid")
    plt.figure(figsize=(18, 6))

    plt.plot(df_plot.index, df_plot["corr_QQQ_NVDA_15"], linewidth=1.2)

    # Nulllinie
    plt.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.6)

    # 5. Tagesgrenzen und Datumstexte
    dates = df_plot[ts_col].dt.date
    day_starts = df_plot.groupby(dates).head(1).index
    unique_dates_plot = df_plot[ts_col].dt.date.unique()

    # Vertikale Linien für Tagesanfang
    for start_idx in day_starts:
        plt.axvline(start_idx, color="white", linestyle="-", linewidth=0.8, alpha=0.5)

    # Nur jeden zweiten Tag beschriften
    label_indices = [i for i in range(len(day_starts)) if i % 2 == 0]
    tick_locs = [day_starts[i] for i in label_indices]
    tick_labels = [
        unique_dates_plot[i].strftime("%Y-%m-%d") for i in label_indices
    ]

    plt.xticks(tick_locs, tick_labels, rotation=45, ha="right", fontsize=9)

    plt.title(
        "Rolling 15-Min Correlation (QQQ vs NVDA) – Trading Hours Only",
        fontsize=14,
    )
    plt.ylabel("Correlation", fontsize=12)
    plt.xlabel("")
    plt.xlim(df_plot.index[0], df_plot.index[-1])

    plt.tight_layout()
    plt.savefig(IMG_PATH / "qqq_nvda_rolling_correlation.png", dpi=300)
    plt.close()
    print("✅ Rolling correlation plot saved: qqq_nvda_rolling_correlation.png")


def plot_target_distribution(df):
    """
    (Optional) Klassifikations-Target-Verteilung für target_5, target_15, target_30.
    Wird aktuell von main() nicht benutzt, schadet aber nicht.
    """
    target_windows = [5, 15, 30]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, window in enumerate(target_windows):
        target_col = f"target_{window}"
        if target_col in df.columns:
            counts = df[target_col].value_counts().sort_index()
            ax = counts.plot(
                kind="bar",
                color=["#e74c3c", "#2ecc71"],
                rot=0,
                ax=axes[i],
            )

            axes[i].set_title(f"Target Distribution ({window}-min Trend)")
            axes[i].set_xlabel("Trend Direction")
            axes[i].set_ylabel("Count")
            ax.set_xticklabels(["Down/Flat", "Up"])

            total = len(df)
            for p in ax.patches:
                percentage = "{:.1f}%".format(100 * p.get_height() / total)
                x = p.get_x() + p.get_width() / 2
                y = p.get_height()
                ax.annotate(percentage, (x, y), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(IMG_PATH / "target_distribution_all.png", dpi=300)
    plt.close()


def plot_feature_target_correlation(df):
    """
    Plot correlation between features and REGRESSION targets.
    """
    target_cols = [f"target_{w}m" for w in [5, 15, 30] if f"target_{w}m" in df.columns]
    if not target_cols:
        print("⚠️ No regression targets found")
        return

    feature_cols = [
        "ema_diff",
        "return_5",
        "volume_norm",
        "relative_strength",
        "corr_QQQ_NVDA_15",
        "NVDA_return_5",
    ]
    available_features = [col for col in feature_cols if col in df.columns]

    corr_matrix = df[available_features + target_cols].corr()
    target_correlations = corr_matrix.loc[available_features, target_cols]

    if target_correlations.empty:
        print("⚠️ Correlation matrix is empty")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        target_correlations,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".3f",
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Feature-Target Correlation Matrix (Regression)")
    plt.tight_layout()
    plt.savefig(IMG_PATH / "feature_target_correlation.png", dpi=300)
    plt.close()

    print("\n📊 Feature-Target Correlations (Regression):")
    print(target_correlations.to_string())


def plot_regression_targets_distribution(df):
    """
    Plot distribution of REGRESSION targets (future returns).
    """
    target_cols = [f"target_{w}m" for w in [5, 15, 30] if f"target_{w}m" in df.columns]
    if not target_cols:
        print("⚠️ No regression targets found for distribution plot")
        return

    fig, axes = plt.subplots(1, len(target_cols), figsize=(15, 4))

    # falls nur ein Target: axes in Liste packen
    if len(target_cols) == 1:
        axes = [axes]

    for i, target_col in enumerate(target_cols):
        data = df[target_col].dropna()
        window = target_col.split("_")[1].replace("m", "")

        axes[i].hist(
            data,
            bins=50,
            alpha=0.7,
            edgecolor="black",
        )
        axes[i].axvline(
            data.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {data.mean():.6f}",
        )
        axes[i].axvline(0, color="black", linestyle="-", alpha=0.5)

        axes[i].set_title(f"Future Return Distribution ({window}-min)")
        axes[i].set_xlabel("Return")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMG_PATH / "regression_targets_distribution.png", dpi=300)
    plt.close()
    print("✅ Regression targets distribution plot saved")


def plot_scatter_returns(df):
    """
    Scatter: 5-Min Returns (NVDA vs QQQ) mit Regressionslinie.
    """
    if "NVDA_return_5" not in df.columns or "return_5" not in df.columns:
        print("⚠️ NVDA_return_5 oder return_5 nicht gefunden")
        return

    data = df[["NVDA_return_5", "return_5"]].dropna()

    # Outlier filtern
    limit = 0.025
    mask = (data["NVDA_return_5"].between(-limit, limit)) & (
        data["return_5"].between(-limit, limit)
    )
    data = data[mask]

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))

    sns.regplot(
        data=data,
        x="NVDA_return_5",
        y="return_5",
        scatter_kws={"alpha": 0.2, "s": 18},
        line_kws={"linewidth": 2},
    )

    ticks = np.arange(-0.03, 0.035, 0.005)
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlim(-0.03, 0.03)
    plt.ylim(-0.03, 0.03)

    plt.title("Scatter: 5-Min Returns (NVDA vs QQQ) with Regression Line", fontsize=14)
    plt.xlabel("NVDA 5-Min Return", fontsize=12)
    plt.ylabel("QQQ 5-Min Return", fontsize=12)

    plt.tight_layout()
    plt.savefig(IMG_PATH / "qqq_nvda_scatter_returns.png", dpi=300)
    plt.close()
    print("✅ Scatter plot saved: qqq_nvda_scatter_returns.png")
