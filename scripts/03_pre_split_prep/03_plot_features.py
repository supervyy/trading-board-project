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
