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
    Rolling 15-Min Correlation (QQQ vs NVDA) – basierend auf corr_QQQ_NVDA_15.
    Keine zusätzliche Uhrzeit-Filterung mehr, da Step 1 bereits
    die regulären Handelstage/-zeiten per Kalender filtert.
    """
    if "corr_QQQ_NVDA_15" not in df.columns:
        print("⚠️ corr_QQQ_NVDA_15 nicht gefunden")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        print("⚠️ Index ist kein DatetimeIndex, verwende vollen Index ohne Tagesstruktur.")
        df_plot = df.copy()
    else:
        df_plot = df.copy()
        df_plot = df_plot.sort_index()

        # Auf die letzten ~20 Handelstage begrenzen (nur für bessere Lesbarkeit)
        unique_days = sorted(pd.unique(df_plot.index.date))
        if len(unique_days) > 20:
            last_20 = unique_days[-20:]
            mask = np.isin(df_plot.index.date, last_20)
            df_plot = df_plot[mask]

    # Index resetten für „schönen“ X-Achsen-Plot
    df_plot = df_plot.reset_index()
    ts_col = "timestamp" if "timestamp" in df_plot.columns else "index"

    sns.set_style("darkgrid")
    plt.figure(figsize=(18, 6))

    plt.plot(df_plot.index, df_plot["corr_QQQ_NVDA_15"], linewidth=1.2)
    plt.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.6)

    # Tagesgrenzen einzeichnen (wenn Datumsinfos da sind)
    if ts_col in df_plot.columns and isinstance(df_plot[ts_col].iloc[0], pd.Timestamp):
        dates = df_plot[ts_col].dt.date
        day_starts = df_plot.groupby(dates).head(1).index
        unique_dates_plot = df_plot[ts_col].dt.date.unique()

        for start_idx in day_starts:
            plt.axvline(start_idx, color="white", linestyle="-", linewidth=0.8, alpha=0.5)

        label_indices = [i for i in range(len(day_starts)) if i % 2 == 0]
        tick_locs = [day_starts[i] for i in label_indices]
        tick_labels = [unique_dates_plot[i].strftime("%Y-%m-%d") for i in label_indices]

        plt.xticks(tick_locs, tick_labels, rotation=45, ha="right", fontsize=9)

    plt.title("Rolling 15-Min Correlation (QQQ vs NVDA)", fontsize=14)
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
    Plot correlation between selected features and REGRESSION targets.
    """
    target_cols = [f"target_{w}m" for w in [5, 15, 30]
                   if f"target_{w}m" in df.columns]
    if not target_cols:
        print("⚠️ No regression targets found")
        return

    # Wichtigste Features – nur die, die es wirklich gibt, werden verwendet
    feature_cols = [
        # QQQ Core
        "ema_diff",
        "return_5",
        "realized_vol_10",
        "volume_norm",
        "volume_acceleration",
        "volume_spike",
        "bid_ask_spread_proxy",

        # NVDA / Tech
        "NVDA_return_5",
        "divergence_NVDA_QQQ_5",
        "momentum_spread_5",
        "nvda_volume_anomaly",

        # Cross-Asset / Regime
        "relative_strength",
        "tech_unanimity",
        "max_divergence",
        "high_vol_regime",
        "low_corr_regime",
    ]
    available_features = [c for c in feature_cols if c in df.columns]

    if not available_features:
        print("⚠️ No selected features found in dataframe.")
        return

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

def plot_lead_lag_corrected(dfs):
    """
    Lead-Lag-Analyse NVDA vs QQQ: Führt NVDA QQQ?
    - 1-min Returns (klein anfangen)
    - Korrekte Shift-Logik: NVDA(t+1) vs QQQ(t) bedeutet NVDA FÜHRT
    """

    # Falls eine Liste [(sym, df), ...] übergeben wurde -> in dict umwandeln
    if isinstance(dfs, list):
        dfs = {sym: df for sym, df in dfs}

    if "QQQ" not in dfs or "NVDA" not in dfs:
        print("⚠️ Für Lead-Lag werden QQQ und NVDA benötigt.")
        return

    q_df = dfs["QQQ"].copy()
    n_df = dfs["NVDA"].copy()

    # Falls noch eine timestamp-Spalte existiert, als Index setzen
    if "timestamp" in q_df.columns:
        q_df["timestamp"] = pd.to_datetime(q_df["timestamp"])
        q_df = q_df.set_index("timestamp").sort_index()
    if "timestamp" in n_df.columns:
        n_df["timestamp"] = pd.to_datetime(n_df["timestamp"])
        n_df = n_df.set_index("timestamp").sort_index()

    # 1. Prices holen
    q = q_df["close"]
    n = n_df["close"]

    # 2. 1-Min-Returns (statt 5)
    ret_window = 1          # hier klein anfangen
    q_ret = q.pct_change(ret_window)
    n_ret = n.pct_change(ret_window)

    # 3. Gemeinsamen Index erzwingen
    df = pd.DataFrame({"qqq": q_ret, "nvda": n_ret}).dropna()

    # 4. Zeitreihe Plot (NVDA shifted 5 min)
    lag_shift = 1           # auch hier klein anfangen
    df["nvda_shift"] = df["nvda"].shift(-lag_shift)

    plt.figure(figsize=(18, 6))
    plt.plot(df.index, df["qqq"], label="QQQ 1-min Return at time t")
    plt.plot(df.index, df["nvda_shift"], label=f"NVDA 1-min Return at time t+{lag_shift} (future)")
    plt.axhline(0, color="black", linewidth=1)
    plt.legend()
    plt.title(f"Lead-Lag: Does NVDA lead QQQ? (NVDA shifted +{lag_shift} min)")
    plt.tight_layout()
    plt.savefig(IMG_PATH / "lead_lag_timeseries_fixed.png", dpi=300)
    plt.close()

    # 5. Scatterplot
    plt.figure(figsize=(10, 10))
    plt.scatter(df["nvda_shift"], df["qqq"], alpha=0.2)

    # Regression hinzufügen für bessere Visualisierung
    from scipy.stats import linregress
    mask = ~np.isnan(df["nvda_shift"]) & ~np.isnan(df["qqq"])
    if mask.sum() > 10:
        slope, intercept, r_value, p_value, std_err = linregress(
            df["nvda_shift"][mask], df["qqq"][mask]
        )
        x_vals = np.array([df["nvda_shift"].min(), df["nvda_shift"].max()])
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, 'r-', label=f'Regression (r={r_value:.3f})')
        plt.legend()
    plt.title(f"Lead-Lag Scatter: NVDA(t+{lag_shift}) vs QQQ(t)\n"
              f"If NVDA leads, points should align along diagonal")
    plt.xlabel(f"NVDA 1-min Return at time t+{lag_shift} (future NVDA)")
    plt.ylabel("QQQ 1-min Return at time t (current QQQ)")
    plt.tight_layout()
    plt.savefig(IMG_PATH / "lead_lag_scatter_fixed.png", dpi=300)
    plt.close()

    # 6. Cross-Correlation
    lags = range(-30, 31)
    corrs = []
    for lag in lags:
        corrs.append(df["nvda"].shift(lag).corr(df["qqq"]))

    plt.figure(figsize=(16, 6))
    plt.plot(lags, corrs, marker="o")
    plt.axhline(0, color="black")

    # Finde besten Lag
    best_lag = lags[np.argmax(corrs)]
    best_corr = corrs[np.argmax(corrs)]
    
    plt.axvline(best_lag, color='red', linestyle='--', 
                alpha=0.7, label=f'Best Lag: {best_lag} min')
    plt.title("Cross-Correlation: NVDA vs QQQ (1-min Returns)\n"
              f"Peak at lag = {best_lag} min (r = {best_corr:.3f})")
    plt.xlabel("Lag (minutes) → Positive: Compare NVDA(t+lag) with QQQ(t)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_PATH / "lead_lag_ccf_fixed.png", dpi=300)
    plt.close()

   # Interpretation ausgeben
    print(f"\n📊 Lead-Lag Analysis Results:")
    print(f"   Best correlation at lag {best_lag} minutes: {best_corr:.4f}")
    if best_lag > 0:
        print(f"   → NVDA LEADS QQQ by {best_lag} minutes")
    elif best_lag < 0:
        print(f"   → QQQ LEADS NVDA by {abs(best_lag)} minutes")
    else:
        print(f"   → No clear lead-lag relationship (simultaneous movement)")
    
    print("✅ Lead-Lag Plots saved.")
def plot_divergence_nvda_qqq_timeseries(df):
    """
    Zeitreihen-Plot der Divergenz zwischen NVDA und QQQ:
    divergence_NVDA_QQQ_5 = NVDA_return_5 - return_5 (QQQ)

    Zeigt, wie stark NVDA sich intraday vom QQQ entkoppelt.
    """
    col = "divergence_NVDA_QQQ_5"
    if col not in df.columns:
        print(f"⚠️ {col} nicht im DataFrame, überspringe Divergenz-Plot.")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        print("⚠️ Index ist kein DatetimeIndex, verwende vollen Index ohne Tagesstruktur.")
        df_plot = df.copy()
    else:
        df_plot = df.copy().sort_index()
        # Auf die letzten ~20 Handelstage begrenzen (Lesbarkeit)
        unique_days = sorted(pd.unique(df_plot.index.date))
        if len(unique_days) > 20:
            last_20 = unique_days[-20:]
            mask = np.isin(df_plot.index.date, last_20)
            df_plot = df_plot[mask]

    sns.set_style("darkgrid")
    plt.figure(figsize=(18, 6))

    plt.plot(df_plot.index, df_plot[col], linewidth=1.0, alpha=0.9)
    plt.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.7)

    plt.title("Divergenz NVDA vs QQQ (5-Min-Returns)", fontsize=14)
    plt.ylabel("NVDA_return_5 - QQQ_return_5", fontsize=12)
    plt.xlabel("Zeit")

    plt.tight_layout()
    plt.savefig(IMG_PATH / "divergence_nvda_qqq_timeseries.png", dpi=300)
    plt.close()
    print("✅ Divergenz-Plot gespeichert: divergence_nvda_qqq_timeseries.png")


def plot_divergence_nvda_vs_target(df):
    """
    Scatterplot: Divergenz (NVDA vs QQQ, 5-Min-Return) vs. zukünftiger QQQ-Return (target_5m).

    Idee:
    - x: divergence_NVDA_QQQ_5
    - y: target_5m (future QQQ return in 5 Minuten)
    """
    if "divergence_NVDA_QQQ_5" not in df.columns or "target_5m" not in df.columns:
        print("⚠️ divergence_NVDA_QQQ_5 oder target_5m nicht im DataFrame, überspringe Scatter-Plot.")
        return

    data = df[["divergence_NVDA_QQQ_5", "target_5m"]].dropna()

    # Outlier clipping für bessere Lesbarkeit
    limit_x = 0.03
    limit_y = 0.03
    mask = data["divergence_NVDA_QQQ_5"].between(-limit_x, limit_x) & data["target_5m"].between(-limit_y, limit_y)
    data = data[mask]

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))

    sns.regplot(
        data=data,
        x="divergence_NVDA_QQQ_5",
        y="target_5m",
        scatter_kws={"alpha": 0.2, "s": 18},
        line_kws={"linewidth": 2},
    )

    plt.title("Divergenz NVDA–QQQ vs. zukünftiger QQQ-Return (5 Min)", fontsize=14)
    plt.xlabel("Divergenz NVDA vs QQQ (5-Min-Return)", fontsize=12)
    plt.ylabel("Future QQQ Return (target_5m)", fontsize=12)

    plt.tight_layout()
    plt.savefig(IMG_PATH / "divergence_nvda_vs_target_5m.png", dpi=300)
    plt.close()
    print("✅ Scatter-Plot gespeichert: divergence_nvda_vs_target_5m.png")


def plot_momentum_spread_timeseries(df):
    """
    Zeitreihen-Plot des Momentum-Spreads:
    momentum_spread_5 = Std-Abweichung der 5-Min-Returns der Tech-Aktien.

    Hohe Werte => Tech-Aktien driften auseinander (mehr Disagreement).
    Niedrige Werte => Tech-Aktien bewegen sich sehr ähnlich (starker gemeinsamer Trend).
    """
    col = "momentum_spread_5"
    if col not in df.columns:
        print(f"⚠️ {col} nicht im DataFrame, überspringe Momentum-Spread-Plot.")
        return

    if not isinstance(df.index, pd.DatetimeIndex):
        print("⚠️ Index ist kein DatetimeIndex, verwende vollen Index ohne Tagesstruktur.")
        df_plot = df.copy()
    else:
        df_plot = df.copy().sort_index()
        unique_days = sorted(pd.unique(df_plot.index.date))
        if len(unique_days) > 20:
            last_20 = unique_days[-20:]
            mask = np.isin(df_plot.index.date, last_20)
            df_plot = df_plot[mask]

    sns.set_style("darkgrid")
    plt.figure(figsize=(18, 6))

    plt.plot(df_plot.index, df_plot[col], linewidth=1.0)
    plt.title("Momentum-Spread der Tech-Aktien (5-Min-Returns)", fontsize=14)
    plt.ylabel("Std. der Tech 5-Min-Returns", fontsize=12)
    plt.xlabel("Zeit")

    plt.tight_layout()
    plt.savefig(IMG_PATH / "momentum_spread_5_timeseries.png", dpi=300)
    plt.close()
    print("✅ Momentum-Spread-Plot gespeichert: momentum_spread_5_timeseries.png")

