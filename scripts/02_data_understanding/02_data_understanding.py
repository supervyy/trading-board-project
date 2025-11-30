import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw"
IMG_PATH = PROJECT_ROOT / "images" / "data_understanding"
IMG_PATH.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]


def load_data():
    dfs = {}
    for sym in SYMBOLS:
        df = pd.read_parquet(DATA_PATH / f"{sym}_1m.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
        dfs[sym] = df
    return dfs


def describe_data(dfs):
    print("\n=== DATA OVERVIEW ===")
    for sym, df in dfs.items():
        print(f"\n--- {sym} ---")
        print(f"Rows: {len(df):,}")
        print(f"Start: {df.index.min()}")
        print(f"End:   {df.index.max()}")
        print(df[["open","high","low","close","volume","vwap"]].describe())


def plot_price(dfs):
    df = dfs["QQQ"]
    plt.figure(figsize=(16, 8))  # Larger plot
    df["close"].plot(label="QQQ Close", linewidth=1)
    
    # Grid
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    
    # Events
    events = {
        "ChatGPT Launch (AI Boom)": "2022-11-30",
        "NVIDIA Earnings Mega-Beat": "2023-05-24",
        "SVB Banking Crisis": "2023-03-10"
    }
    
    y_min, y_max = df["close"].min(), df["close"].max()
    
    for event, date in events.items():
        dt = pd.Timestamp(date)
        if df.index.min() <= dt <= df.index.max():
            plt.axvline(dt, color='darkred', linestyle='-', alpha=0.5, linewidth=1.5)
            plt.text(dt, y_max, f" {event}", rotation=90, verticalalignment='top', fontsize=10, color='darkred')

    # Set x-axis limits to exact data range
    plt.xlim(df.index.min(), df.index.max())

    plt.title("QQQ Close Price Over Time (with Key Events)", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_PATH / "qqq_close.png", dpi=300)
    plt.close()


def plot_avg_intraday_volume(dfs):
    df = dfs["QQQ"].copy()
    
    # Ensure timezone is US/Eastern for correct 09:30-16:00 filtering/grouping
    if df.index.tz is None:
        # Assuming UTC if naive, based on previous data inspection (14:30 start)
        df.index = df.index.tz_localize('UTC')
    
    # Convert to US/Eastern
    df.index = df.index.tz_convert('US/Eastern')
    
    # Filter Market Hours (09:30 - 16:00)
    # Note: The user stated data is already filtered, but explicit filtering ensures correctness
    # especially if the input data spans slightly outside (e.g. pre/post market)
    market_open = pd.Timestamp("09:30").time()
    market_close = pd.Timestamp("16:00").time()
    df = df.between_time(market_open, market_close)
    
    # Calculate Average Volume per Minute
    # Group by time of day
    avg_volume = df.groupby(df.index.time)['volume'].mean()
    
    # Rolling 5-min smoothing (optional but requested)
    avg_volume_smooth = avg_volume.rolling(window=5, center=True).mean()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Create a dummy date to plot time on x-axis easily
    dummy_date = pd.Timestamp("2000-01-01")
    times = [dummy_date.replace(hour=t.hour, minute=t.minute) for t in avg_volume.index]
    
    plt.plot(times, avg_volume_smooth, color='#3b7cff', linewidth=2, label='Avg Intraday Volume (5-min Smooth)')
    # Also plot raw data faintly if desired, or just the smooth one. User asked for "Glatte Kurve".
    # plt.plot(times, avg_volume, color='#3b7cff', alpha=0.2, linewidth=1)
    
    # Formatting X-Axis
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Explicit Ticks: 09:30, 12:00, 16:00
    specific_ticks = [
        dummy_date.replace(hour=9, minute=30),
        dummy_date.replace(hour=12, minute=0),
        dummy_date.replace(hour=16, minute=0)
    ]
    plt.xticks(list(plt.xticks()[0]) + [mdates.date2num(t) for t in specific_ticks])
    
    # Set x-axis limits to exact trading hours (09:30 - 16:00)
    plt.xlim(specific_ticks[0], specific_ticks[-1])
    plt.margins(x=0)  # Ensure no padding
    
    # Formatting Y-Axis (Thousands separator)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.title("QQQ – Average Intraday Volume Curve", fontsize=14)
    plt.xlabel("Time (US/Eastern)", fontsize=12)
    plt.ylabel("Average Volume", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    save_path = IMG_PATH / "qqq_avg_intraday_volume.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Average Intraday Volume Plot saved: {save_path.name}")


def plot_returns_hist(dfs):
    returns = dfs["QQQ"]["close"].pct_change().dropna()
    
    # Setup Figure
    plt.figure(figsize=(12, 7))
    
    # Parameters for Histogram
    x_min, x_max = -0.005, 0.005
    bins = 60  # "Breitere Balken" -> fewer bins for the range
    
    # Plot Histogram (Log Scale Y)
    counts, bins_edges, patches = plt.hist(
        returns, 
        bins=bins, 
        range=(x_min, x_max), 
        color='skyblue', 
        edgecolor='black', 
        alpha=0.6, 
        log=True,
        label='Frequency (Log)'
    )
    
    # Normal Distribution Overlay
    mu, std = returns.mean(), returns.std()
    x = np.linspace(x_min, x_max, 1000)
    pdf = norm.pdf(x, mu, std)
    
    # Scale PDF to match histogram counts
    # Area of histogram bar = count * bin_width
    # Total area under PDF = 1
    # We want PDF curve to be comparable to counts.
    # Height of PDF curve at x should be: pdf(x) * total_samples * bin_width
    # But since we are in log scale, we just plot this scaled curve.
    bin_width = (x_max - x_min) / bins
    scaled_pdf = pdf * len(returns) * bin_width
    
    plt.plot(x, scaled_pdf, 'r--', linewidth=2, label=f'Normal Dist (μ={mu:.5f}, σ={std:.5f})')
    
    # Customizing Axes
    plt.xlim(x_min, x_max)
    
    # X-Ticks as requested
    x_ticks = [-0.005, -0.0025, 0, 0.0025, 0.005]
    plt.xticks(x_ticks, [f"{t:.2%}" for t in x_ticks], fontsize=11)
    
    # Y-Ticks (Log) - minor ticks handling
    plt.ylabel("Frequency (Log Scale)", fontsize=12)
    plt.xlabel("1-Minute Return", fontsize=12)
    
    # Title and Grid
    plt.title("QQQ 1-Min Returns (Log Histogram)", fontsize=14, fontweight='bold')
    plt.grid(True, which='major', linestyle='--', alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    # Save
    save_path = IMG_PATH / "qqq_returns_hist_improved.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Improved Returns Histogram saved: {save_path.name}")


def plot_correlation_heatmap(dfs):
    returns = pd.DataFrame({
        sym: dfs[sym]["close"].pct_change()
        for sym in SYMBOLS
    }).dropna()

    plt.figure(figsize=(10, 8)) # Larger size
    
    # Warmer colors (RdYlBu_r: Red=High, Blue=Low) and larger text
    sns.heatmap(returns.corr(), annot=True, cmap="RdYlBu_r", fmt=".2f", 
                annot_kws={"size": 12}, vmin=-1, vmax=1)
    
    plt.title("Correlation Heatmap (1-Min Returns)", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_PATH / "corr_heatmap.png", dpi=300)
    plt.close()


def main():
    dfs = load_data()
    describe_data(dfs)
    plot_price(dfs)
    plot_avg_intraday_volume(dfs)
    plot_returns_hist(dfs)
    plot_correlation_heatmap(dfs)
    print("\nData Understanding Completed. Plots saved.")


if __name__ == "__main__":
    main()
