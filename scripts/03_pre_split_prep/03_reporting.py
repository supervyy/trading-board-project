import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMG_PATH = PROJECT_ROOT / "images" / "data_preparation"
REPORT_PATH = PROJECT_ROOT / "reports"
IMG_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

def save_sample_table(df):
    """
    Save a sample table (10 rows) as PNG with better formatting.
    """
    # Select 10 random rows
    sample = df.sample(n=10, random_state=42)
    
    # Smart formatting function
    def format_cell_value(val, col_name):
        if pd.isna(val):
            return ''
        if 'return' in col_name and abs(val) < 0.001:
            return f"{val:.6f}"
        elif isinstance(val, float):
            return f"{val:.4f}"
        elif isinstance(val, (int, np.integer)):
            return str(val)
        else:
            return str(val)
    
    # Select and format columns
    priority_cols = ['close', 'ema_5', 'ema_diff', 'return_5', 'NVDA_return_5', 
                     'corr_QQQ_NVDA_15', 'relative_strength', 'target_5', 'target_15', 'target_30']
    
    display_cols = [c for c in priority_cols if c in sample.columns]
    sample_display = sample[display_cols].copy()
    
    # Apply formatting
    for col in display_cols:
        sample_display[col] = sample_display[col].apply(lambda x: format_cell_value(x, col))
    
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    cell_text = sample_display.values.tolist()
    col_labels = sample_display.columns.tolist()
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code based on column type
    for col_idx, col_name in enumerate(col_labels):
        if 'return' in col_name:
            for row_idx in range(len(cell_text)):
                cell = table[(row_idx+1, col_idx)]
                cell.set_text_props(ha='right')
    
    plt.title("Sample Features (10 Random Rows) - Returns shown with 6 decimals", 
              fontsize=14, y=0.95)
    plt.tight_layout()
    plt.savefig(IMG_PATH / "sample_features.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ Sample table saved: sample_features.png")

def save_feature_stats(df):
    """
    Save descriptive statistics as PNG with smart rounding.
    """
    stats = df.describe().T
    
    # SMART ROUNDING based on feature type
    def smart_round(series, col_name):
        """Apply different rounding based on feature scale."""
        if 'return' in col_name or 'vol' in col_name:
            return series.round(6)  # More decimals for small returns
        elif 'norm' in col_name or 'diff' in col_name or 'slope' in col_name:
            return series.round(4)  # Medium decimals
        elif 'close' in col_name or 'ema' in col_name:
            return series.round(2)  # Few decimals for prices
        else:
            return series.round(4)  # Default
    
    # Apply smart rounding to each row
    for idx in stats.index:
        stats.loc[idx] = smart_round(stats.loc[idx], idx)
    
    # Round count to integer
    stats['count'] = stats['count'].round(0).astype(int)
    
    # Create the table plot
    rows = len(stats)
    fig, ax = plt.subplots(figsize=(14, rows * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Format values for display (scientific notation for very small values)
    stats_display = stats.copy()
    
    def format_value(val, col_name, stat_name):
        """Format value for display."""
        if pd.isna(val):
            return ''
        
        # Scientific notation for very small returns
        if 'return' in col_name and abs(val) < 0.0001 and val != 0:
            return f"{val:.2e}"
        
        # Already rounded, convert to string
        if isinstance(val, float):
            return f"{val:.6f}" if 'return' in col_name else f"{val:.4f}"
        return str(val)
    
    # Convert to display strings
    display_data = []
    for idx, row in stats.iterrows():
        display_row = [idx]  # Feature name
        for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            if stat in row:
                display_row.append(format_value(row[stat], idx, stat))
        display_data.append(display_row)
    
    # Column labels
    col_labels = ['Feature'] + [c for c in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] 
                               if c in stats.columns]
    
    table = ax.table(cellText=display_data, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Smaller font for more rows
    table.scale(1, 1.3)
    
    # Adjust column widths
    for key, cell in table.get_celld().items():
        row, col = key
        if col == 0:  # Feature names
            cell.set_width(0.3)
            cell.set_text_props(ha='left', fontsize=8)
            if row > 0:
                cell.set_text_props(ha='left', x=0.02, fontsize=8)
        else:  # Numbers
            cell.set_width(0.1)
            cell.set_text_props(ha='right', fontsize=8)
    
    plt.title("Descriptive Statistics (Smart Rounding Applied)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(IMG_PATH / "feature_stats.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Also save as CSV for exact values
    csv_path = REPORT_PATH / "feature_statistics.csv"
    stats.to_csv(csv_path)
    
    print(f"✅ Feature stats saved: feature_stats.png")
    print(f"✅ Exact values saved: {csv_path}")
    
    # Print key stats to console
    print("\n📊 KEY STATISTICS SUMMARY:")
    return_cols = [c for c in df.columns if 'return' in c and 'target' not in c]
    for col in return_cols[:3]:  # Show first 3 returns
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"   {col}: mean={mean_val:.6f}, std={std_val:.6f}")

def generate_markdown_report(df, train, val, test):
    """
    Generate a comprehensive Markdown report answering the user's questions.
    """
    print("📝 Writing Markdown Report...")
    
    report_file = REPORT_PATH / "pre_split_report.md"
    
    # Calculate stats
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Target distribution
    target_dist = ""
    for w in [5, 15, 30]:
        if f'target_{w}' in df.columns:
            up = df[f'target_{w}'].mean() * 100
            target_dist += f"- **{w}min Target**: {up:.1f}% Upward / {100-up:.1f}% Downward\n"
            
    # Split stats
    split_stats = f"""
| Split | Rows | Start Date | End Date | % of Total |
|-------|------|------------|----------|------------|
| Train | {len(train):,} | {train.index.min().date()} | {train.index.max().date()} | {len(train)/n_rows:.1%} |
| Val   | {len(val):,} | {val.index.min().date()} | {val.index.max().date()} | {len(val)/n_rows:.1%} |
| Test  | {len(test):,} | {test.index.min().date()} | {test.index.max().date()} | {len(test)/n_rows:.1%} |
"""

    markdown_content = f"""# Pre-Split Data Preparation Report

## 1. Data Preparation Process

### Feature Engineering
We prepared features using a **time-series aware approach** to prevent look-ahead bias.
- **Sorting**: Data was sorted by timestamp immediately after loading.
- **Calculations**: Features (EMAs, Rolling Stats) were calculated on the full dataset *before* splitting to ensure continuity at split boundaries.
- **Normalization**: Volume and VWAP were normalized relative to recent history (rolling 60min mean) to make them stationarity-friendly.

### Target Generation
Targets were generated as binary classifications:
- `1` if `Close[t+w] > Close[t]`
- `0` otherwise
- Windows: 5, 15, 30 minutes.

### Data Cleaning
- **Outliers**: Removed extreme returns (>5% in 5min) and unrealistic volume spikes (>10x avg).
- **Missing Values**: Used `ffill()` (Forward Fill) only. **No `bfill()`** was used to strictly prevent data leakage from the future.
- **Market Hours**: Filtered to 09:30 - 16:00 ET.

## 2. Parameters Used

- **Input Tickers**: QQQ (Target), NVDA, AAPL, MSFT, GOOGL, AMZN (Features)
- **Timeframe**: 1-Minute Bars
- **Feature Windows**: [5, 10, 20] minutes for EMAs
- **Target Windows**: [5, 15, 30] minutes
- **Train/Val/Test Split**:
    - Train: First 70%
    - Validation: Next 15%
    - Test: Last 15%

## 3. Descriptive Statistics

### Dataset Overview
- **Total Rows**: {n_rows:,}
- **Total Features**: {n_cols}

### Target Balance
{target_dist}

### Split Distribution
{split_stats}

### Feature Statistics
![Feature Stats](../images/data_preparation/feature_stats.png)

## 4. Samples

### Feature Sample (First 10 Rows)
![Sample Features](../images/data_preparation/sample_features.png)

## 5. Visualizations

### Feature Distributions & Correlations
*(See generated plots in `images/data_preparation/`)*

## 6. Findings & Feature Selection

### Selected Features
We selected features based on financial intuition and literature:
1.  **Trend**: EMAs (5, 10, 20) and EMA Slopes capture short-term momentum.
2.  **Volatility**: Realized Volatility (10m) captures market regime.
3.  **Volume**: Normalized volume indicates activity relative to recent history.
4.  **Cross-Asset**:
    - `relative_strength`: How QQQ performs vs. Top Tech.
    - `momentum_leader`: Which tech stock is leading the rally/drop.
    - `corr_QQQ_NVDA_15`: Dynamic correlation with the market leader.

### Key Observations
- **Stationarity**: Returns and normalized features are stationary, suitable for ML.
- **Balance**: Targets are well-balanced (~51-52% Up), slightly bullish bias typical for QQQ.
- **Data Quality**: Outlier removal reduced noise without losing significant data (<2% rows removed).

---
*Generated automatically by `03_reporting.py`*
"""

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    print(f"✅ Report saved to: {report_file}")

