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
                     'corr_QQQ_NVDA_15', 'relative_strength', 'target_30']
    
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
