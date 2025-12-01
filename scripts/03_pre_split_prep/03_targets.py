def generate_targets(df):
    """
    Generate REGRESSION targets ONLY for [5, 15, 30] minutes.
    """
    df = df.copy()
    
    target_windows = [5, 15, 30]
    
    # ONLY Regression Targets
    for window in target_windows:
        df[f'target_{window}m'] = (df['close'].shift(-window) - df['close']) / df['close']
    
    # Remove rows at the end
    max_window = max(target_windows)
    df = df.iloc[:-max_window]
    
    print(f"✅ Generated REGRESSION targets for windows: {target_windows}")
    for window in target_windows:
        target_col = f'target_{window}m'
        if target_col in df.columns:
            mean_val = df[target_col].mean()
            std_val = df[target_col].std()
            print(f"   - {window:2d}min: mean={mean_val:.6f}, std={std_val:.6f}")
    
    return df

def save_regression_target_statistics(df):
    """
    Create descriptive statistics for REGRESSION targets.
    Returns table exactly like the other project.
    """
    # Regression targets (target_5m, target_15m, target_30m)
    target_cols = [f'target_{w}m' for w in [5, 15, 30] if f'target_{w}m' in df.columns]
    
    if not target_cols:
        print("❌ No regression targets found")
        return None
    
    # Calculate statistics for each target
    stats_list = []
    for target in target_cols:
        stats = df[target].describe()
        stats['target'] = target.replace('target_', 'target_return_')  # Match other project naming
        stats_list.append(stats)
    
    # Create DataFrame
    targets_df = pd.DataFrame(stats_list)
    
    # Reorder and rename columns to match other project
    targets_df = targets_df[['target', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    # Format exactly like other project
    targets_df['count'] = targets_df['count'].astype(int)
    numeric_cols = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    targets_df[numeric_cols] = targets_df[numeric_cols].round(5)  # 5 decimals like other project
    
    # Save as PNG table
    fig, ax = plt.subplots(figsize=(12, len(target_cols) * 0.6 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data (with formatted strings)
    display_data = []
    for _, row in targets_df.iterrows():
        display_row = [
            row['target'],
            f"{row['count']:,}",
            f"{row['mean']:.5f}",
            f"{row['std']:.5f}", 
            f"{row['min']:.5f}",
            f"{row['25%']:.5f}",
            f"{row['50%']:.5f}",
            f"{row['75%']:.5f}",
            f"{row['max']:.5f}"
        ]
        display_data.append(display_row)
    
    col_labels = ['Target', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    
    table = ax.table(cellText=display_data, 
                     colLabels=col_labels, 
                     loc='center',
                     cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style header (blue like other project)
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Style important columns
    for row in range(1, len(targets_df) + 1):
        # Highlight Mean and Std columns
        table[(row, 2)].set_facecolor('#F0F8FF')  # Mean
        table[(row, 3)].set_facecolor('#F0F8FF')  # Std
    
    plt.title("Deskriptive Statistik - Targets (Regression)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(IMG_PATH / "regression_target_statistics.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Also save as CSV
    csv_path = REPORT_PATH / "regression_target_statistics.csv"
    targets_df.to_csv(csv_path, index=False)
    
    print(f"✅ Regression target statistics saved: regression_target_statistics.png")
    print(f"✅ CSV saved: {csv_path}")
    
    # Print summary to console
    print("\n📈 REGRESSION TARGETS SUMMARY:")
    print("=" * 60)
    for _, row in targets_df.iterrows():
        target_name = row['target']
        mean_val = float(row['mean'])
        std_val = float(row['std'])
        print(f"{target_name:20} : mean = {mean_val:.6f}, std = {std_val:.6f}")
    print("=" * 60)
    
    return targets_df