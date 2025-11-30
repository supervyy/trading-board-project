"""
Data Understanding for QQQ Trend Prediction with Top Tech Stocks

This script performs data quality checks and exploratory analysis specifically tailored
for the QQQ trend prediction project using NVDA, AAPL, MSFT, GOOGL, AMZN as predictors.

Focus Areas:
1. Target variable feasibility for t=[5,10,15,20,30,60,120] minutes
2. QQQ technical features calculation validation
3. Cross-asset consistency across all 6 symbols
4. Multi-asset relationship features testing
5. Lead-lag pattern identification

Input:
- Parquet files for QQQ, NVDA, AAPL, MSFT, GOOGL, AMZN from data/raw/

Output:
- Data quality report
- Feature validation results
- Visualizations of key relationships
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw"
IMAGES_PATH = PROJECT_ROOT / "images"
SYMBOLS = ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
TARGET_WINDOWS = [5, 10, 15, 20, 30, 60, 120]

# Create images directory
IMAGES_PATH.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_and_validate_data(symbols):
    """Load all symbol data and validate basic data quality."""
    dataframes = {}
    quality_report = {}

    for symbol in symbols:
        try:
            file_path = DATA_PATH / f"{symbol}_1m.parquet"
            df = pd.read_parquet(file_path)

            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Basic quality metrics
            quality_report[symbol] = {
                'start_date': df['timestamp'].min(),
                'end_date': df['timestamp'].max(),
                'total_rows': len(df),
                'missing_open': df['open'].isnull().sum(),
                'missing_high': df['high'].isnull().sum(),
                'missing_low': df['low'].isnull().sum(),
                'missing_close': df['close'].isnull().sum(),
                'missing_volume': df['volume'].isnull().sum(),
                'missing_vwap': df['vwap'].isnull().sum(),
                'zero_volume': (df['volume'] == 0).sum(),
                'duplicate_timestamps': df['timestamp'].duplicated().sum()
            }

            dataframes[symbol] = df
            print(
                f"✅ {symbol}: Loaded {len(df)} rows from {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

        except Exception as e:
            print(f"❌ {symbol}: Failed to load - {e}")
            quality_report[symbol] = {'error': str(e)}

    return dataframes, quality_report


def check_timezone_info(df_dict):
    """Zeige Zeitzonen-Informationen."""
    qqq_df = df_dict['QQQ']
    print("\n🌍 TIMEZONE ANALYSIS:")
    print(f"Original timestamp timezone: {qqq_df['timestamp'].dt.tz}")
    print(f"Erste 5 Timestamps (UTC): {qqq_df['timestamp'].head(3).tolist()}")
    
    # Test-Konvertierung
    qqq_df = qqq_df.copy()
    if qqq_df['timestamp'].dt.tz is None:
        print("⚠️  Keine Zeitzonen-Info - nehme an UTC")
        qqq_df['timestamp'] = qqq_df['timestamp'].dt.tz_localize('UTC')
    
    qqq_df['timestamp_et'] = qqq_df['timestamp'].dt.tz_convert('US/Eastern')
    print(f"Erste 3 in ET: {qqq_df['timestamp_et'].head(3).tolist()}")


def convert_to_et_and_filter_hours(df_dict):
    """Konvertiere UTC zu Eastern Time und filtere NYSE Handelszeiten."""
    filtered_dfs = {}
    
    for symbol, df in df_dict.items():
        df = df.copy()
        
        # Stelle sicher, dass Zeitzone gesetzt ist
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        
        # Konvertiere zu Eastern Time
        df['timestamp_et'] = df['timestamp'].dt.tz_convert('US/Eastern')
        df['time_et'] = df['timestamp_et'].dt.time
        
        # NYSE Handelszeiten: 9:30-16:00 Eastern Time
        market_hours_mask = (df['time_et'] >= pd.Timestamp('09:30:00').time()) & (df['time_et'] <= pd.Timestamp('16:00:00').time())
        filtered_dfs[symbol] = df[market_hours_mask]
        
        print(f"✅ {symbol}: {len(filtered_dfs[symbol])}/{len(df)} Minuten in NYSE Handelszeiten")
    
    return filtered_dfs


def check_target_variable_feasibility(df_dict):
    """Check if we can calculate target variables for all windows."""
    print("\n" + "=" * 50)
    print("TARGET VARIABLE FEASIBILITY CHECK")
    print("=" * 50)

    qqq_df = df_dict['QQQ']
    results = {}

    for window in TARGET_WINDOWS:
        try:
            # Calculate future return and trend direction
            future_return = qqq_df['close'].shift(-window) / qqq_df['close'] - 1
            trend_direction = (future_return > 0).astype(int)

            # Statistics
            results[window] = {
                'feasible': True,
                'total_calculable': trend_direction.notna().sum(),
                'upward_percentage': trend_direction.mean(),
                'missing_at_end': trend_direction.isna().sum()
            }

            print(f"✅ {window:3d}min: {results[window]['upward_percentage']:.1%} upward trends "
                  f"({results[window]['total_calculable']:,} calculable periods)")

        except Exception as e:
            results[window] = {'feasible': False, 'error': str(e)}
            print(f"❌ {window:3d}min: Failed - {e}")

    return results


def validate_qqq_technical_features(qqq_df):
    """Validate that all planned QQQ technical features can be calculated."""
    print("\n" + "=" * 50)
    print("QQQ TECHNICAL FEATURES VALIDATION")
    print("=" * 50)

    features = {}

    try:
        # EMAs
        for span in [5, 10, 20]:
            features[f'ema_{span}'] = qqq_df['close'].ewm(span=span).mean()
        features['ema_diff'] = features['ema_5'] - features['ema_20']
        print("✅ EMAs [5,10,20] + Differenz berechenbar")

        # Returns
        for minutes in [5, 15, 30]:
            features[f'return_{minutes}min'] = qqq_df['close'].pct_change(minutes)
        print("✅ Returns [5,15,30]min berechenbar")

        # Volatility
        features['volatility_10min'] = qqq_df['close'].pct_change().rolling(10).std()
        print("✅ Realisierte Volatilität (10min) berechenbar")

        # Normalized Volume and VWAP
        features['volume_norm'] = qqq_df['volume'] / qqq_df['volume'].rolling(60).mean()
        features['vwap_norm'] = qqq_df['vwap'] / qqq_df['close']
        print("✅ Normalisiertes Volume & VWAP berechenbar")

        # Check for any NaN issues
        for name, feature in features.items():
            nan_percentage = feature.isna().mean()
            if nan_percentage > 0.1:  # More than 10% missing
                print(f"⚠️  {name}: {nan_percentage:.1%} missing values")

        return True, features

    except Exception as e:
        print(f"❌ Feature Berechnung fehlgeschlagen: {e}")
        return False, features


def check_cross_asset_consistency(df_dict):
    """Check temporal alignment and data consistency across all assets."""
    print("\n" + "=" * 50)
    print("CROSS-ASSET CONSISTENCY CHECK")
    print("=" * 50)

    # Find common timestamp index
    common_index = None
    for symbol, df in df_dict.items():
        if common_index is None:
            common_index = set(df['timestamp'])
        else:
            common_index = common_index.intersection(set(df['timestamp']))

    common_index = sorted(common_index)
    print(f"Gemeinsame Handelsminuten: {len(common_index):,}")

    # Report coverage for each symbol
    for symbol, df in df_dict.items():
        symbol_minutes = len(df)
        common_minutes = len(set(df['timestamp']).intersection(common_index))
        coverage = common_minutes / symbol_minutes
        print(f"📊 {symbol}: {coverage:.1%} Coverage ({common_minutes:,}/{symbol_minutes:,} Minuten)")

    return common_index


def test_multi_asset_relationship_features(df_dict, common_index):
    """Test calculation of multi-asset relationship features."""
    print("\n" + "=" * 50)
    print("MULTI-ASSET RELATIONSHIP FEATURES TEST")
    print("=" * 50)

    try:
        # Calculate 5-minute returns for all symbols on common timeline
        returns_5min = {}
        for symbol in SYMBOLS:
            df = df_dict[symbol]
            df_common = df[df['timestamp'].isin(common_index)].sort_values('timestamp').set_index('timestamp')
            returns_5min[symbol] = df_common['close'].pct_change(5)

        returns_df = pd.DataFrame(returns_5min)
        
        # ✅ NEU: NaN Handling
        original_len = len(returns_df)
        returns_df = returns_df.dropna()
        print(f"✅ Returns nach NaN-Entfernung: {len(returns_df)}/{original_len} Perioden")

        # 1. Correlation QQQ vs Tech Basket (15min)
        tech_symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        tech_basket_returns = returns_df[tech_symbols].mean(axis=1)
        correlation = returns_df['QQQ'].rolling(15).corr(tech_basket_returns)
        print(f"✅ Korrelation QQQ vs Tech-Basket (15min) berechenbar - {correlation.notna().sum():,} Werte")

        # 2. Relative Strength
        relative_strength = returns_df['QQQ'] - tech_basket_returns
        print(f"✅ Relative Stärke (QQQ vs Tech) berechenbar - {relative_strength.notna().sum():,} Werte")

        # 3. Tech Momentum Leader
        tech_returns = returns_df[tech_symbols]
        momentum_leader = tech_returns.idxmax(axis=1)
        leader_counts = momentum_leader.value_counts()
        print(f"✅ Tech Momentum Leader identifizierbar")
        print(f"   Führungshäufigkeit: {dict(leader_counts)}")

        return True, {
            'correlation': correlation,
            'relative_strength': relative_strength,
            'momentum_leader': momentum_leader,
            'returns_df': returns_df
        }

    except Exception as e:
        print(f"❌ Multi-Asset Features fehlgeschlagen: {e}")
        return False, {}


def debug_data_quality(returns_df):
    """Schnelle Diagnose der Datenprobleme."""
    print("\n🔍 DATA QUALITY DEBUG:")
    print(f"Shape: {returns_df.shape}")
    print(f"Zeitraum: {returns_df.index.min()} bis {returns_df.index.max()}")
    print(f"NaN NVDA: {returns_df['NVDA'].isna().sum()} ({returns_df['NVDA'].isna().mean():.1%})")
    print(f"NaN QQQ: {returns_df['QQQ'].isna().sum()} ({returns_df['QQQ'].isna().mean():.1%})")
    print(f"NVDA Mean: {returns_df['NVDA'].mean():.6f}")
    print(f"QQQ Mean: {returns_df['QQQ'].mean():.6f}")
    print(f"NVDA Std: {returns_df['NVDA'].std():.6f}")
    print(f"QQQ Std: {returns_df['QQQ'].std():.6f}")
    
    # Korrelation ohne Lag
    corr = returns_df['NVDA'].corr(returns_df['QQQ'])
    print(f"Sofort-Korrelation NVDA-QQQ: {corr:.3f}")


def test_all_tech_stocks(returns_df, max_lag=3):
    """Teste alle Tech-Aktien schnell."""
    tech_symbols = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    print("\n📊 ALL TECH STOCKS LEAD-LAG TEST:")
    for symbol in tech_symbols:
        correlations = {}
    results = {}
    
    for symbol in tech_symbols:
        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = returns_df[symbol].shift(abs(lag)).corr(returns_df['QQQ'])
            elif lag > 0:
                corr = returns_df[symbol].corr(returns_df['QQQ'].shift(lag))
            else:
                corr = returns_df[symbol].corr(returns_df['QQQ'])
            correlations[lag] = corr
        
        optimal_lag = max(correlations.items(), key=lambda x: abs(x[1]))
        results[symbol] = {
            'optimal_lag': optimal_lag[0],
            'optimal_correlation': optimal_lag[1],
            'all_correlations': correlations
        }
        print(f"🔍 {symbol}: Optimal lag {optimal_lag[0]}min (corr: {optimal_lag[1]:.3f})")
    
    return results


def plot_momentum_leader_analysis(leader_counts):
    """Plot momentum leader frequency."""
    plt.figure(figsize=(10, 6))
    plt.bar(leader_counts.keys(), leader_counts.values(), alpha=0.8, color='lightblue')
    plt.title('Tech Momentum Leader Frequency')
    plt.ylabel('Number of Periods')
    plt.xlabel('Tech Stock')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(IMAGES_PATH / 'momentum_leader.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Momentum Leader Plot gespeichert: momentum_leader.png")


def plot_target_distribution(target_results):
    """Plot upward trend probability by time horizon."""
    windows = list(target_results.keys())
    probabilities = [target_results[window]['upward_percentage'] * 100 for window in windows]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar([str(w) for w in windows], probabilities, alpha=0.7, color='skyblue')
    plt.title('Upward Trend Probability by Time Horizon')
    plt.ylabel('Probability (%)')
    plt.xlabel('Time Window (minutes)')
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Random Chance (50%)')
    
    # Werte auf den Balken anzeigen
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{prob:.1f}%', ha='center', va='bottom')
    
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(IMAGES_PATH / 'target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Target Distribution Plot gespeichert: target_distribution.png")


def plot_qqq_technical_features(qqq_features):
    """Plot distributions of key QQQ technical features."""
    features_to_plot = ['ema_diff', 'volatility_10min', 'return_5min']
    
    for i, feature_name in enumerate(features_to_plot):
        if feature_name in qqq_features:
            plt.figure(figsize=(10, 6))
            plt.hist(qqq_features[feature_name].dropna(), bins=50, alpha=0.7, color='lightgreen')
            plt.title(f'Distribution: {feature_name.replace("_", " ").title()}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(IMAGES_PATH / f'qqq_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ QQQ Feature Plot gespeichert: qqq_{feature_name}.png")


def create_visualizations(returns_df, qqq_features, relationship_features):
    """Create all individual visualizations."""
    print("\n" + "="*50)
    print("CREATING IMPROVED VISUALIZATIONS")
    print("="*50)
    
    try:
        # 1. Lead-Lag Analyse für NVDA (Hauptfokus) - KORRIGIERT
        print("📈 Analyzing NVDA-QQQ lead-lag relationship...")
        correlations, optimal_lag = analyze_lead_lag_correlation(returns_df)
        plot_lead_lag_time_based(returns_df, optimal_lag)
        
        # 2. Lead-Lag für alle Tech-Aktien
        print("📊 Analyzing all tech stocks lead-lag relationships...")
        all_lead_lag = analyze_all_lead_lag(returns_df)
        
        # 3. Korrigierte Korrelations-Analyse
        plot_proper_correlation_analysis(returns_df)
        
        # 4. Momentum Leader mit aktuellen Daten
        if 'momentum_leader' in relationship_features:
            leader_counts = relationship_features['momentum_leader'].value_counts().to_dict()
            plot_momentum_leader_analysis(leader_counts)
        
        # 5. Target Distribution
        plot_target_distribution({
            window: {'upward_percentage': returns_df['QQQ'].shift(-window).gt(returns_df['QQQ']).mean()}
            for window in TARGET_WINDOWS
        })
        
        # 6. QQQ Features
        plot_qqq_technical_features(qqq_features)
        
        print("✅ Alle verbesserten Visualisierungen gespeichert im '/images/' Ordner")
        
    except Exception as e:
        print(f"❌ Visualisierung fehlgeschlagen: {e}")


def generate_summary_report(quality_report, target_results, feature_valid, relationship_valid):
    """Generate final summary report."""
    print("\n" + "="*60)
    print("DATA UNDERSTANDING SUMMARY REPORT")
    print("="*60)
    
    # Data Quality Summary
    total_rows = sum(report.get('total_rows', 0) for report in quality_report.values() if 'error' not in report)
    missing_data = sum(report.get('missing_close', 0) for report in quality_report.values() if 'error' not in report)
    
    print(f"📊 DATENQUALITÄT:")
    print(f"   • Gesamte Handelsminuten: {total_rows:,}")
    print(f"   • Fehlende Close-Preise: {missing_data:,} ({missing_data/total_rows:.1%})")
    print(f"   • Assets erfolgreich geladen: {len([r for r in quality_report.values() if 'error' not in r])}/6")
    
    # Target Variable Summary
    feasible_targets = sum(1 for r in target_results.values() if r.get('feasible', False))
    print(f"🎯 ZIELVARIABLEN:")
    print(f"   • Machbare Zeitfenster: {feasible_targets}/7")
    
    # Features Summary
    print(f"🔧 FEATURES:")
    print(f"   • QQQ Technische Features: {'✅' if feature_valid else '❌'}")
    print(f"   • Multi-Asset Relationships: {'✅' if relationship_valid else '❌'}")
    
    # Recommendations
    print(f"💡 EMPFEHLUNGEN:")
    if feasible_targets == 7:
        print("   • Alle Zielvariablen berechenbar - Fortfahren mit Feature Engineering")
    else:
        print("   • Überprüfe längere Zeitfenster (>120min) auf Machbarkeit")
    
    if feature_valid and relationship_valid:
        print("   • Alle geplanten Features umsetzbar - Projekt fortführen")
        print("   • Trotz schwacher linearer Korrelation: Neuronale Netze können nicht-lineare Muster finden")
    else:
        print("   • Feature-Definition anpassen benötigt")


def main():
    """Main execution function."""
    print("🚀 STARTING DATA UNDERSTANDING FOR QQQ TREND PREDICTION")
    print("="*70)
    
    # 1. Load and validate all data
    df_dict, quality_report = load_and_validate_data(SYMBOLS)
    
    if not df_dict:
        print("❌ Keine Daten geladen - Abbruch")
        return
    
    # ✅ NEU: Zeitzonen prüfen und konvertieren
    check_timezone_info(df_dict)
    
    # ✅ NEU: Zu Eastern Time konvertieren und filtern
    print("\n🔧 CONVERTING TO EASTERN TIME AND FILTERING NYSE HOURS...")
    df_dict = convert_to_et_and_filter_hours(df_dict)
    
    # 2. Check target variable feasibility
    target_results = check_target_variable_feasibility(df_dict)
    
    # 3. Validate QQQ technical features
    qqq_df = df_dict['QQQ']
    feature_valid, qqq_features = validate_qqq_technical_features(qqq_df)
    
    # 4. Check cross-asset consistency
    common_index = check_cross_asset_consistency(df_dict)
    
    # 5. Test multi-asset relationship features
    relationship_valid, relationship_features = test_multi_asset_relationship_features(df_dict, common_index)
    
    # 6. Debug data quality and test all tech stocks
    if relationship_valid and 'returns_df' in relationship_features:
        debug_data_quality(relationship_features['returns_df'])
        test_all_tech_stocks(relationship_features['returns_df'])
    
    # 7. Create IMPROVED visualizations
    if relationship_valid and 'returns_df' in relationship_features:
        create_visualizations(relationship_features['returns_df'], qqq_features, relationship_features)
    
    # 8. Generate final summary
    generate_summary_report(quality_report, target_results, feature_valid, relationship_valid)
    
    print("\n✅ DATA UNDERSTANDING COMPLETED")


if __name__ == "__main__":
    main()