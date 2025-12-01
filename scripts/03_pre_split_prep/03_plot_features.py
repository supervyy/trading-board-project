import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMG_PATH = PROJECT_ROOT / "images" / "data_preparation"
IMG_PATH.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8')

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
    Plot Rolling Correlation (QQQ vs NVDA).
    """
    plt.figure(figsize=(12, 4))
    subset = df.iloc[-1000:]
    if "corr_QQQ_NVDA_15" in subset.columns:
        plt.plot(subset.index, subset["corr_QQQ_NVDA_15"], color="purple")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Rolling 15-Min Correlation: QQQ vs NVDA")
        plt.savefig(IMG_PATH / "plot_B_rolling_corr.png")
    plt.close()

def plot_target_distribution(df):
    """
    Plot Target Distribution for ALL targets [5, 15, 30] minutes.
    """
    target_windows = [5, 15, 30]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, window in enumerate(target_windows):
        target_col = f'target_{window}'
        if target_col in df.columns:
            counts = df[target_col].value_counts().sort_index()
            ax = counts.plot(kind="bar", color=["#e74c3c", "#2ecc71"], rot=0, ax=axes[i])
            
            axes[i].set_title(f"Target Distribution ({window}-min Trend)")
            axes[i].set_xlabel("Trend Direction")
            axes[i].set_ylabel("Count")
            
            # Set custom x-labels
            ax.set_xticklabels(["Down/Flat", "Up"])
            
            # Add percentage labels
            total = len(df)
            for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_height()/total)
                x = p.get_x() + p.get_width()/2
                y = p.get_height()
                ax.annotate(percentage, (x, y), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(IMG_PATH / "target_distribution_all.png", dpi=300)
    plt.close()

def plot_feature_target_correlation(df):
    """
    Plot correlation between features and REGRESSION targets.
    """
    # REGRESSION targets only
    target_cols = [f'target_{w}m' for w in [5, 15, 30] if f'target_{w}m' in df.columns]
    
    if not target_cols:
        print("⚠️ No regression targets found")
        return
    
    feature_cols = ['ema_diff', 'return_5', 'volume_norm', 'relative_strength', 
                   'corr_QQQ_NVDA_15', 'NVDA_return_5']
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Korrelationsmatrix
    corr_matrix = df[available_features + target_cols].corr()
    target_correlations = corr_matrix.loc[available_features, target_cols]
    
    if target_correlations.empty:
        print("⚠️ Correlation matrix is empty")
        return
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(target_correlations, annot=True, cmap='coolwarm', center=0,
                fmt='.3f', cbar_kws={'shrink': 0.8})
    
    plt.title('Feature-Target Correlation Matrix (Regression)')
    plt.tight_layout()
    plt.savefig(IMG_PATH / "feature_target_correlation.png", dpi=300)
    plt.close()
    
    print("\n📊 Feature-Target Correlations (Regression):")
    print(target_correlations.to_string())

def plot_regression_targets_distribution(df):
    """
    Plot distribution of REGRESSION targets (future returns).
    """
    target_cols = [f'target_{w}m' for w in [5, 15, 30] if f'target_{w}m' in df.columns]
    
    if not target_cols:
        print("⚠️ No regression targets found for distribution plot")
        return
    
    fig, axes = plt.subplots(1, len(target_cols), figsize=(15, 4))
    
    for i, target_col in enumerate(target_cols):
        data = df[target_col].dropna()
        window = target_col.split('_')[1].replace('m', '')
        
        axes[i].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(data.mean(), color='red', linestyle='--', 
                       label=f'Mean: {data.mean():.6f}')
        axes[i].axvline(0, color='black', linestyle='-', alpha=0.5)
        
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
    Plot Scatter: 5-Min Returns (NVDA vs QQQ) with Regression Line.
    """
    import numpy as np
    
    # 1. Prepare Data
    if "NVDA_return_5" not in df.columns or "return_5" not in df.columns:
        return
        
    data = df[["NVDA_return_5", "return_5"]].dropna()
    
    # Filter outliers (-2.5% to +2.5%)
    limit = 0.025
    mask = (data["NVDA_return_5"].between(-limit, limit)) & \
           (data["return_5"].between(-limit, limit))
    data = data[mask]
    
    # 2. Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8), facecolor='white')
    
    sns.regplot(
        data=data,
        x="NVDA_return_5",
        y="return_5",
        scatter_kws={'alpha': 0.2, 's': 18},
        line_kws={'color': 'red', 'linewidth': 2}
    )
    
    # Custom Ticks (0.005 steps)
    # Range from -0.03 to 0.03 to cover the data comfortably
    ticks = np.arange(-0.03, 0.035, 0.005)
    plt.xticks(ticks)
    plt.yticks(ticks)
    
    # Limits slightly larger than data to show edge ticks
    plt.xlim(-0.03, 0.03)
    plt.ylim(-0.03, 0.03)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMG_PATH = PROJECT_ROOT / "images" / "data_preparation"
IMG_PATH.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8')

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
    Plot Rolling 15-Min Correlation (QQQ vs NVDA) – Trading Hours Only (No Gaps).
    Matches the visual style of the provided reference image (Darkgrid, Clean Ticks).
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Filter Trading Hours (09:30 - 16:00)
    df_plot = df.between_time("09:30", "16:00").copy()
    
    # 2. Subset to last 20 days (approx 1 month) to match image density
    # Ensure index is datetime
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        return

    unique_days = sorted(pd.unique(df_plot.index.date))
    if len(unique_days) > 20:
        last_20_days = unique_days[-20:]
        mask = np.isin(df_plot.index.date, last_20_days)
        df_plot = df_plot[mask]
    
    # 3. Reset Index for Gap Removal
    # This creates a standard RangeIndex (0, 1, 2, ...) -> x-axis
    df_plot = df_plot.reset_index() 
    
    # Check if we have the data
    if "corr_QQQ_NVDA_15" not in df_plot.columns:
        return

    # 4. Setup Plot
    plt.figure(figsize=(14, 6))
    sns.set_style("darkgrid") # Match the gray background
    
    # Main Line
    plt.plot(df_plot.index, df_plot["corr_QQQ_NVDA_15"], 
             color="#3b7cff", linewidth=1.5, label="Correlation")
    
    # Zero Line
    plt.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    
    # 5. Daily Boundaries & Labels
    ts_col = "timestamp" if "timestamp" in df_plot.columns else "index"
    dates = df_plot[ts_col].dt.date
    
    # Find start indices of each day
    day_starts = df_plot.groupby(dates).head(1).index
    unique_dates = df_plot[ts_col].dt.date.unique()
    
    # Draw vertical lines
    for start_idx in day_starts:
        plt.axvline(start_idx, color="white", linestyle="-", linewidth=0.8, alpha=0.5)
        
    # Set X-Ticks (Every 2nd day to avoid crowding)
    label_indices = [i for i in range(len(day_starts)) if i % 2 == 0]
    tick_locs = [day_starts[i] for i in label_indices]
    tick_labels = [unique_dates[i].strftime("%Y-%m-%d") for i in label_indices]
    
    plt.xticks(tick_locs, tick_labels, rotation=45, ha="right", fontsize=10)

    # 6. Final Design
    plt.title("Rolling 15-Min Correlation (QQQ vs NVDA) – Trading Hours Only", fontsize=14)
    plt.ylabel("Correlation", fontsize=12)
    plt.xlabel("") 
    
    # Clean up spines if desired, but darkgrid usually keeps them subtle
    # sns.despine(left=True, bottom=True) 
    
    # Remove side margins
    plt.xlim(df_plot.index[0], df_plot.index[-1])
    
    plt.tight_layout()
    
    # Save to data_preparation as requested
    plt.savefig(IMG_PATH / "qqq_nvda_rolling_corr.png", dpi=300)
    plt.close()

def plot_target_distribution(df):
    """
    Plot Target Distribution (Bar Plot: Down/Flat vs Up).
    """
    plt.figure(figsize=(6, 4))
    if "target_30" in df.columns:
        counts = df["target_30"].value_counts().sort_index()
        ax = counts.plot(kind="bar", color=["#e74c3c", "#2ecc71"], rot=0)
        plt.title("Target Distribution (30-min Trend)")
        plt.xlabel("Trend Direction")
        plt.ylabel("Count")
        
        # Set custom x-labels
        ax.set_xticklabels(["Down/Flat", "Up"])
        
        # Add percentage labels
        total = len(df)
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height()/total)
            x = p.get_x() + p.get_width()/2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom')
            
        plt.savefig(IMG_PATH / "target_distribution_30min.png")
    plt.close()

def plot_scatter_returns(df):
    """
    Plot Scatter: 5-Min Returns (NVDA vs QQQ) with Regression Line.
    """
    import numpy as np
    
    # 1. Prepare Data
    if "NVDA_return_5" not in df.columns or "return_5" not in df.columns:
        return
        
    data = df[["NVDA_return_5", "return_5"]].dropna()
    
    # Filter outliers (-2.5% to +2.5%)
    limit = 0.025
    mask = (data["NVDA_return_5"].between(-limit, limit)) & \
           (data["return_5"].between(-limit, limit))
    data = data[mask]
    
    # 2. Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8), facecolor='white')
    
    sns.regplot(
        data=data,
        x="NVDA_return_5",
        y="return_5",
        scatter_kws={'alpha': 0.2, 's': 18},
        line_kws={'color': 'red', 'linewidth': 2}
    )
    
    # Custom Ticks (0.005 steps)
    # Range from -0.03 to 0.03 to cover the data comfortably
    ticks = np.arange(-0.03, 0.035, 0.005)
    plt.xticks(ticks)
    plt.yticks(ticks)
    
    # Limits slightly larger than data to show edge ticks
    plt.xlim(-0.03, 0.03)
    plt.ylim(-0.03, 0.03)
    
    plt.title("Scatter: 5-Min Returns (NVDA vs QQQ) with Regression Line", fontsize=14)
    plt.xlabel("NVDA 5-Min Return", fontsize=12)
    plt.ylabel("QQQ 5-Min Return", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(IMG_PATH / "qqq_nvda_rolling_correlation.png", dpi=300)
    plt.close()
