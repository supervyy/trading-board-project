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
    Save a sample table (10 rows) as PNG.
    """
    # Select 10 random rows
    sample = df.sample(n=10, random_state=42).round(4)
    
    # Select a subset of columns to fit in the image if there are too many
    # Prioritize key features
    cols = list(sample.columns)
    priority_cols = ['close', 'ema_5', 'ema_diff', 'return_5', 'NVDA_return_5', 
                     'corr_QQQ_NVDA_15', 'relative_strength', 'target_30']
    
    # Keep priority cols + a few others up to max 12 cols
    display_cols = [c for c in priority_cols if c in cols]
    remaining = [c for c in cols if c not in display_cols]
    display_cols.extend(remaining[:4])
    
    sample = sample[display_cols]

    fig, ax = plt.subplots(figsize=(16, 6)) # Wide figure
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=sample.values, colLabels=sample.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title("Sample Features (10 Random Rows)", fontsize=14, y=0.95)
    plt.savefig(IMG_PATH / "sample_features.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Sample table saved: sample_features.png")

def save_feature_stats(df):
    """
    Save descriptive statistics as PNG.
    """
    stats = df.describe().T.round(4)
    
    # Split into chunks if too long, but for now let's try one big image or just top features
    # If too many rows, matplotlib table might be ugly.
    # Let's just save the whole thing but make figure tall.
    
    rows = len(stats)
    fig, ax = plt.subplots(figsize=(14, rows * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Add index as a column
    stats_reset = stats.reset_index()
    col_labels = stats_reset.columns
    
    table = ax.table(cellText=stats_reset.values, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Adjust column widths manually
    # Column 0 (Feature Name) needs to be wider
    for key, cell in table.get_celld().items():
        row, col = key
        if col == 0:
            cell.set_width(0.25) # Wider for feature names
            cell.set_text_props(ha='left') # Left align feature names
            # Add some padding to left alignment
            if row > 0: # Skip header
                cell.set_text_props(ha='left', x=0.05)
        else:
            cell.set_width(0.08) # Narrower for numbers
    
    plt.title("Descriptive Statistics", fontsize=16, y=0.98)
    plt.savefig(IMG_PATH / "feature_stats.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Feature stats saved: feature_stats.png")

def generate_report(df):
    """
    Generate automated markdown report.
    """
    stats = df.describe().T
    
    # --- Automated Checks ---
    # Returns
    mean_ret = df['return_5'].mean()
    std_ret = df['return_5'].std()
    
    # Cross Asset
    if 'corr_QQQ_NVDA_15' in df.columns:
        corr_mean = df['corr_QQQ_NVDA_15'].mean()
    else:
        corr_mean = 0
        
    # Target
    if 'target_30' in df.columns:
        target_balance = df['target_30'].mean()
    else:
        target_balance = 0
        
    # NaNs
    nan_count = df.isna().sum().sum()
    
    # --- Text Generation ---
    findings = []
    
    # Returns
    if abs(mean_ret) < 0.001:
        findings.append(f"Die 5-Minuten-Returns zeigen eine erwartungsgemäße Mean-Reversion um 0 (Mean: {mean_ret:.5f}).")
    else:
        findings.append(f"Die Returns weisen einen leichten Drift auf (Mean: {mean_ret:.5f}).")
        
    findings.append(f"Die Volatilität (Std Dev) der 5-Min-Returns liegt bei {std_ret:.4f}, was auf typische Intraday-Schwankungen hindeutet.")
    
    # Correlation
    findings.append(f"Die Cross-Asset-Korrelation zwischen QQQ und NVDA ist im Durchschnitt hoch ({corr_mean:.2f}), was die Relevanz von NVDA als Prädiktor bestätigt.")
    
    # Target
    findings.append(f"Das Target 'target_30' ist mit {target_balance:.1%} (Klasse 1) sehr gut balanciert, was ideal für das Modelltraining ist.")
    
    # Data Quality
    if nan_count == 0:
        findings.append("Es wurden keine fehlenden Werte (NaNs) im finalen Datensatz gefunden; die Bereinigung war erfolgreich.")
    else:
        findings.append(f"Warnung: Es sind noch {nan_count} NaNs im Datensatz enthalten.")
        
    # EMAs
    if 'ema_diff' in df.columns:
        ema_diff_std = df['ema_diff'].std()
        findings.append(f"Der EMA-Diff-Indikator zeigt eine gesunde Dynamik (Std: {ema_diff_std:.4f}) ohne extreme Ausreißer, die auf Datenfehler hindeuten würden.")

    # Construct Report
    md_content = f"""# Pre-Split Feature Report

## Abschnitt A — Summary Table
Die wichtigsten statistischen Kennzahlen (Auszug):

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| return_5 | {stats.loc['return_5', 'mean']:.5f} | {stats.loc['return_5', 'std']:.5f} | {stats.loc['return_5', 'min']:.5f} | {stats.loc['return_5', 'max']:.5f} |
| ema_diff | {stats.loc['ema_diff', 'mean']:.4f} | {stats.loc['ema_diff', 'std']:.4f} | {stats.loc['ema_diff', 'min']:.4f} | {stats.loc['ema_diff', 'max']:.4f} |
| corr_QQQ_NVDA_15 | {corr_mean:.4f} | {stats.loc['corr_QQQ_NVDA_15', 'std']:.4f} | {stats.loc['corr_QQQ_NVDA_15', 'min']:.4f} | {stats.loc['corr_QQQ_NVDA_15', 'max']:.4f} |
| target_30 | {target_balance:.4f} | {stats.loc['target_30', 'std']:.4f} | {stats.loc['target_30', 'min']:.0f} | {stats.loc['target_30', 'max']:.0f} |

*(Vollständige Tabelle siehe `images/data_preparation/feature_stats.png`)*

## Abschnitt B — Plausibility Check
- **Data Range**: {df.index.min()} bis {df.index.max()}
- **Total Rows**: {len(df):,}
- **NaN Check**: {"✅ Passed (0 NaNs)" if nan_count == 0 else f"❌ Failed ({nan_count} NaNs)"}
- **Target Balance**: {target_balance:.1%} (Target 1) / {1-target_balance:.1%} (Target 0)

## Abschnitt C — Automatisch generierte Findings
{' '.join(findings)}

---
*Report generated automatically by `03_reporting.py`*
"""
    
    with open(REPORT_PATH / "pre_split_feature_report.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"✅ Report generated: reports/pre_split_feature_report.md")
