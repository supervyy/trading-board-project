import os
import time
import sys
import datetime
import pytz
import joblib
import torch
import numpy as np
import pandas as pd
import yfinance as yf
import yaml
from pathlib import Path
import yaml
from pathlib import Path

import yaml
from pathlib import Path
import importlib.util

# Add project root to sys.path to allow importing from other scripts (needed for relative imports inside the loaded module)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import the model class from the training script using importlib (bypass numeric package name syntax error)
try:
    model_script_path = PROJECT_ROOT / "scripts" / "07_modeling" / "07_feed_forward.py"
    spec = importlib.util.spec_from_file_location("feed_forward_module", model_script_path)
    ff_module = importlib.util.module_from_spec(spec)
    sys.modules["feed_forward_module"] = ff_module
    spec.loader.exec_module(ff_module)
    MultiHorizonMLP = ff_module.MultiHorizonMLP
    print("✅ Successfully imported MultiHorizonMLP from source.")
except Exception as e:
    print(f"⚠️ Could not import MultiHorizonMLP from {model_script_path}: {e}")
    sys.exit(1)

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- Configuration ---
# API Keys (Ensure these are set in your environment variables)
# Helper for paths (Already defined above)
# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parents[1]
KEYS_PATH = PROJECT_ROOT / "conf" / "keys.yaml"

# API Keys
try:
    with open(KEYS_PATH, "r") as f:
        keys = yaml.safe_load(f)
    print(f"✅ Loaded keys from {KEYS_PATH}")
    ALPACA_API_KEY = keys["KEYS"]["ORIGINAL-APCA-API-KEY-ID-Data"]
    ALPACA_SECRET_KEY = keys["KEYS"]["ORIGINAL-APCA-API-SECRET-KEY-Data"]
except Exception as e:
    print(f"⚠️ Failed to load keys from {KEYS_PATH}: {e}")
    # Fallback to env or None
    ALPACA_API_KEY = os.getenv("ORIGINAL-ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ORIGINAL-ALPACA_SECRET_KEY")

# ALPACA_BASE_URL is handled by paper=True in TradingClient

# Trading Settings
SYMBOL_TRADE = "QQQ"
SYMBOLS_DATA = ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
TECH_SYMBOLS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
PROB_THRESHOLD = 0.55
HOLD_DURATION_MINUTES = 15

# Model Paths
# User requested model from 07_feed_forward
MODEL_PATH = PROJECT_ROOT / "models" / "feed_forward" / "multihorizon_nn.pt"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

# Feature List (Matches training data subset)
# Based on sample_X_train_scaled.csv (19 features)
MODEL_FEATURES = [
    "ema_diff", "return_5", "realized_vol_10", "volume_norm", "volume_acceleration",
    "NVDA_return_5", "NVDA_volume_norm", "divergence_NVDA_QQQ_5", "corr_QQQ_NVDA_15",
    "relative_strength", "momentum_spread_5", "tech_unanimity", "max_divergence",
    "high_vol_regime", "low_corr_regime", "overextended_up", "overextended_down",
    "bid_ask_spread_proxy", "is_15_30_16_00"
]

# Scaler calculates all available features (63+), but model only sees these 19.


# --- Feature Engineering Functions ---
def engineer_tech_features_full(df, symbol):
    """
    Calculate full suite of features for a tech stock to match training data.
    """
    df = df.copy()
    close = df['Close']
    volume = df['Volume']
    prefix = f"{symbol}_"
    
    # EMAs
    df[f'{prefix}ema_5'] = close.ewm(span=5, adjust=False).mean()
    df[f'{prefix}ema_10'] = close.ewm(span=10, adjust=False).mean()
    df[f'{prefix}ema_20'] = close.ewm(span=20, adjust=False).mean()
    
    # EMA Slope
    df[f'{prefix}ema_slope'] = df[f'{prefix}ema_5'] - df[f'{prefix}ema_20']
    
    # Returns
    df[f'{prefix}return_5'] = close.pct_change(5, fill_method=None)
    df[f'{prefix}return_15'] = close.pct_change(15, fill_method=None)
    df[f'{prefix}return_30'] = close.pct_change(30, fill_method=None)
    
    # Normalized Features
    # Fallback if history is short: fill with 1.0 or mean
    roll_vol = volume.rolling(60).mean().replace(0, np.nan)
    df[f'{prefix}volume_norm'] = volume / roll_vol
    
    # VWAP Norm intentionally omitted if raw data lacks vwap or to simplify, 
    # but based on 03_features it was used. If yfinance doesn't provide VWAP, we skip or approx.
    # yfinance 'Downloads' usually has Open, High, Low, Close, Adj Close, Volume. No VWAP.
    # We will assume scaler might need it if it was in 03_features.
    # Calculating approx VWAP from OHLCV?
    vwap_approx = (df['High'] + df['Low'] + df['Close']) / 3
    df[f'{prefix}vwap_norm'] = vwap_approx / close
    
    return df

def calculate_features(df_dict, scaler_feature_names):
    """
    Replicates the exact feature engineering pipeline using the dictionary of DataFrames.
    df_dict: { 'QQQ': df_qqq, 'NVDA': df_nvda, ... }
    scaler_feature_names: list of feature names the scaler expects.
    Returns: DataFrame with correct columns for inference.
    """
    
    # 1. Process QQQ Core Features
    df_qqq = df_dict["QQQ"].copy()
    close_q = df_qqq["Close"]
    volume_q = df_qqq["Volume"]
    
    # EMAs
    df_qqq["ema_5"] = close_q.ewm(span=5, adjust=False).mean()
    df_qqq["ema_10"] = close_q.ewm(span=10, adjust=False).mean() # Likely needed
    df_qqq["ema_20"] = close_q.ewm(span=20, adjust=False).mean()
    
    # Core Features
    df_qqq["ema_diff"] = df_qqq["ema_5"] - df_qqq["ema_20"]
    df_qqq["return_5"] = close_q.pct_change(5, fill_method=None)
    df_qqq["return_15"] = close_q.pct_change(15, fill_method=None) # Likely needed
    df_qqq["return_30"] = close_q.pct_change(30, fill_method=None) # Likely needed
    df_qqq["realized_vol_10"] = close_q.pct_change(fill_method=None).rolling(10).std()
    
    # Volume
    vol_mean_60 = volume_q.rolling(60).mean()
    df_qqq["volume_norm"] = volume_q / vol_mean_60
    df_qqq["volume_acceleration"] = df_qqq["volume_norm"].diff(5)
    df_qqq["volume_spike"] = (df_qqq["volume_norm"] > 2.0).astype(int)
    
    # Spread / Efficiency
    df_qqq["bid_ask_spread_proxy"] = (df_qqq["High"] - df_qqq["Low"]) / close_q
    
    # Efficiency Ratio
    high_5 = df_qqq["High"].rolling(5).max()
    low_5 = df_qqq["Low"].rolling(5).min()
    rng = (high_5 - low_5).replace(0, np.nan)
    df_qqq["efficiency_ratio"] = close_q.diff(5).abs() / rng
    
    # Regime
    df_qqq["overextended_up"] = (close_q > df_qqq["ema_20"] * 1.005).astype(int)
    df_qqq["overextended_down"] = (close_q < df_qqq["ema_20"] * 0.995).astype(int)
    
    # RSI Proxy
    ret5 = df_qqq["return_5"]
    roll_mean = ret5.rolling(10).mean()
    roll_std = ret5.rolling(10).std().replace(0, np.nan)
    rs_safe = (roll_mean / roll_std).clip(-10, 10)
    df_qqq["rsi_proxy"] = (100 - 100 / (1 + rs_safe)).clip(0, 100)

    # Vol Regime
    vol = df_qqq["realized_vol_10"]
    df_qqq["high_vol_regime"] = (vol > vol.rolling(100, min_periods=50).quantile(0.8)).astype(int)
    
    # Time Features
    # Convert index to ET to match training data logic
    if df_qqq.index.tz is None:
        idx_et = df_qqq.index # Assume already ET from fetch
    else:
        idx_et = df_qqq.index.tz_convert('US/Eastern')
        
    df_qqq["minute_of_day"] = idx_et.hour * 60 + idx_et.minute
    df_qqq["is_9_30_10_00"] = ((idx_et.hour == 9) & (idx_et.minute >= 30) & (idx_et.minute < 60)).astype(int)
    df_qqq["is_15_30_16_00"] = ((idx_et.hour == 15) & (idx_et.minute >= 30)).astype(int)

    # 2. Process Tech Features
    tech_dfs = {}
    tech_return_cols = []
    
    for sym in TECH_SYMBOLS:
        if sym in df_dict:
            dft = engineer_tech_features_full(df_dict[sym], sym)
            tech_dfs[sym] = dft
            
            # Track return col for cross-asset
            col_ret = f"{sym}_return_5"
            if col_ret in dft.columns:
                tech_return_cols.append(col_ret)
                
    # Merge everything to Master
    df_master = df_qqq.copy()
    for sym, dft in tech_dfs.items():
        # Join on index (inner)
        # We need to only keep the relevant columns to avoid collision if any?
        # Tech cols are prefixed, so safe.
        df_master = df_master.join(dft, rsuffix='_dup')
        # Drop dups if any
        # df_master = df_master.loc[:, ~df_master.columns.str.endswith('_dup')]

    # 3. Cross-Asset Features
    # Re-calculate on master to ensure alignment
    # (Checking if columns exist in master)
    present_tech_ret_cols = [c for c in tech_return_cols if c in df_master.columns]
    
    if present_tech_ret_cols:
        avg_tech_return = df_master[present_tech_ret_cols].mean(axis=1)
        df_master["relative_strength"] = df_master["return_5"] - avg_tech_return
        
        tech_signs = np.sign(df_master[present_tech_ret_cols])
        sign_qqq = np.sign(df_master["return_5"])
        # summing boolean gives count of True
        df_master["tech_unanimity"] = tech_signs.eq(sign_qqq, axis=0).sum(axis=1) / len(present_tech_ret_cols)
        
        df_master["momentum_spread_5"] = df_master[present_tech_ret_cols].std(axis=1)
        
        divergences = df_master[present_tech_ret_cols].sub(df_master["return_5"], axis=0)
        df_master["max_divergence"] = divergences.abs().max(axis=1)
        
        # Divergence specific
        for sym in TECH_SYMBOLS:
            col_ret = f"{sym}_return_5"
            if col_ret in df_master.columns:
                df_master[f"divergence_{sym}_QQQ_5"] = df_master[col_ret] - df_master["return_5"]

    # Correlation
    if "NVDA_return_5" in df_master.columns:
        df_master["corr_QQQ_NVDA_15"] = df_master["return_5"].rolling(15).corr(df_master["NVDA_return_5"])
        df_master["low_corr_regime"] = (df_master["corr_QQQ_NVDA_15"] < 0.5).astype(int)
        
    if "NVDA_volume_norm" in df_master.columns:
        df_master["nvda_volume_anomaly"] = df_master["NVDA_volume_norm"] # alias if needed based on 03_features

    # Fill NaNs
    df_master = df_master.ffill().fillna(0.0)
    
    # 4. Filter to Scaler Expected Columns
    # Create missing columns with 0.0
    for feature in scaler_feature_names:
        if feature not in df_master.columns:
            # print(f"⚠️ Warning: Missing feature {feature}, filling 0.")
            df_master[feature] = 0.0
            
    # Return only the needed columns in correct order along with index
    return df_master[scaler_feature_names]

# --- Data Fetching ---
def fetch_live_data():
    """
    Fetches last 5 days of 1-minute data for all symbols.
    Returns dictionary of DataFrames.
    """
    data = {}
    print(f"[{datetime.datetime.now()}] Fetching data...")
    
    # yfinance allows fetching multiple tickers at once
    tickers = " ".join(SYMBOLS_DATA)
    try:
        # Fetch 5 days to ensure enough history for rolling windows
        df_all = yf.download(tickers, period="5d", interval="1m", progress=False, group_by='ticker', auto_adjust=False)
        
        # Determine if we have MultiIndex columns (if >1 ticker)
        if len(SYMBOLS_DATA) > 1:
            for sym in SYMBOLS_DATA:
                try:
                    df_sym = df_all[sym].copy()
                    
                    # Basic cleanup
                    if df_sym.empty:
                        print(f"⚠️ Warning: No data for {sym}")
                        continue
                        
                    # Filter RTH (09:30 - 16:00 US/Eastern)
                    # Convert to ET if not already
                    if df_sym.index.tz is None:
                        df_sym.index = df_sym.index.tz_localize('UTC').tz_convert('US/Eastern')
                    else:
                        df_sym.index = df_sym.index.tz_convert('US/Eastern')
                    
                    # Filter logic: between_time is convenient
                    df_sym = df_sym.between_time('09:30', '16:00')
                    
                    data[sym] = df_sym
                except Exception as e:
                    print(f"Error processing {sym}: {e}")
        else:
            # Single ticker handling (just in case)
            pass 

    except Exception as e:
        print(f"Critical Error fetching data: {e}")
        return None

    # Align Timestamps (Inner Join)
    if not data:
        return None
        
    common_index = data[SYMBOL_TRADE].index
    for sym in data:
        common_index = common_index.intersection(data[sym].index)
    
    aligned_data = {}
    for sym in data:
        aligned_data[sym] = data[sym].loc[common_index]
        
    return aligned_data

# --- Trading Helper ---
def check_signals(alpaca_client, model, scaler):
    # 0. Check Market Status
    is_open = True
    next_open = None
    try:
        clock = alpaca_client.get_clock()
        if not clock.is_open:
            is_open = False
            next_open = clock.next_open
            # If we want to display logs AS IF market is open, we simply proceed.
            # print(f"[{datetime.datetime.now()}] Market is CLOSED. Next open: {clock.next_open}")
    except Exception as e:
        print(f"Error checking market clock: {e}")

    # 1. Get Data
    data_dict = fetch_live_data()
    if not data_dict or SYMBOL_TRADE not in data_dict:
        print("Data fetch failed or incomplete.")
        return

    # Dynamic Feature Selection from Scaler
    if hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
    else:
        # Fallback
        expected_features = list(data_dict[SYMBOL_TRADE].columns) 
        
    df_features = calculate_features(data_dict, expected_features)
    
    if df_features.empty:
        print("Feature DataFrame empty.")
        return
        
    last_row = df_features.iloc[[-1]] # DataFrame (1, 63)
    timestamp = last_row.index[0]
    
    # 3. Model Inference
    prob = 0.0
    try:
        # Scale (returns numpy array)
        X_scaled_np = scaler.transform(last_row) # (1, 63)
        
        # Convert back to DataFrame to select model features by name
        if hasattr(scaler, "feature_names_in_"):
            X_scaled_df = pd.DataFrame(X_scaled_np, columns=scaler.feature_names_in_)
        else:
            X_scaled_df = pd.DataFrame(X_scaled_np, columns=expected_features)
            
        # Select the 19 model features
        missing = [f for f in MODEL_FEATURES if f not in X_scaled_df.columns]
        if missing:
             print(f"Missing model features in scaled data: {missing}")
             return

        X_model_input = X_scaled_df[MODEL_FEATURES].values # (1, 19)
        X_tensor = torch.FloatTensor(X_model_input)
        
        model.eval()
        with torch.no_grad():
            logits = model(X_tensor)
            logit_15m = logits[0, 1].item()
            prob = 1 / (1 + np.exp(-logit_15m)) # Sigmoid
        
        status_str = "OPEN" if is_open else "CLOSED"
        price = data_dict['QQQ']['Close'].iloc[-1]
        
        # STANDARD OUTPUT
        print(f"[{timestamp}] Status: {status_str} | Price: {price:.2f} | Prob(Up): {prob:.4f}")
        
    except Exception as e:
        print(f"Inference Error: {e}")
        return

    # 4. Trading Logic (Only if OPEN)
    if is_open:
        try:
            # Check current position
            try:
                position = alpaca_client.get_open_position(SYMBOL_TRADE)
            except Exception:
                position = None
            
            # Define current time
            now = datetime.datetime.now(datetime.timezone.utc)
            
            # BUY LOGIC
            if prob >= PROB_THRESHOLD:
                if not position:
                    from alpaca.trading.requests import GetOrdersRequest
                    req_pending = GetOrdersRequest(status='open', symbols=[SYMBOL_TRADE], side=OrderSide.BUY)
                    pending_orders = alpaca_client.get_orders(req_pending)
                    
                    if not pending_orders:
                        print(f"🚀 SIGNAL: BUY QQQ (Prob {prob:.4f} >= {PROB_THRESHOLD})")
                        req = MarketOrderRequest(
                            symbol=SYMBOL_TRADE,
                            qty=1,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY
                        )
                        alpaca_client.submit_order(req)
                    else:
                        print(f"Signal BUY, but order already pending ({len(pending_orders)}).")
                else:
                    print("Signal BUY, but already holding position.")
                    
            # HOLD/SELL LOGIC (Time-based exit)
            if position:
                # ... check hold time ...
                # Simplified check for brevity as we are just wrapping
                from alpaca.trading.requests import GetOrdersRequest
                req_orders = GetOrdersRequest(status='closed', side=OrderSide.BUY, symbols=[SYMBOL_TRADE], limit=5)
                orders = alpaca_client.get_orders(req_orders)
                last_buy = next((o for o in orders if o.filled_at is not None), None)
                
                if last_buy:
                    filled_at = last_buy.filled_at
                    if filled_at.tzinfo is None:
                        filled_at = filled_at.replace(tzinfo=datetime.timezone.utc)
                    
                    duration = now - filled_at
                    minutes_held = duration.total_seconds() / 60
                    print(f"Position held for {minutes_held:.1f} minutes.")
                    
                    if minutes_held >= HOLD_DURATION_MINUTES:
                        print(f"⏰ TIME EXIT: Selling QQQ (Held > {HOLD_DURATION_MINUTES}m)")
                        req_sell = MarketOrderRequest(
                            symbol=SYMBOL_TRADE,
                            qty=position.qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        alpaca_client.submit_order(req_sell)
        except Exception as e:
            print(f"Trading Error: {e}")
    else:
        # Market Closed behavior
        if next_open:
            print(f"Market Closed. Next Open: {next_open}. Logic skipped.")

# --- Main Execution ---
def main():
    print("--- ORIGINAL Deployment (QQQ) ---")
    
    # 1. Load Model & Scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Critical: Model or Scaler not found.")
        sys.exit(1)
        
    try:
        scaler = joblib.load(SCALER_PATH)
        # Instantiate Model
        # We need input dim from scaler
        if hasattr(scaler, "n_features_in_"):
            # Scaler has 63, but model uses subset
            pass
        
        in_dim = len(MODEL_FEATURES) # 19
            
        # Revert to 3 outputs as per MultiHorizonMLP definition in 07
        model = MultiHorizonMLP(in_dim=in_dim, out_dim=3)
        # Load Weights
        # map_location='cpu' is safer for deployment if no GPU
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print("✅ Model and Scaler loaded.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        sys.exit(1)

    # 2. Init Alpaca
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("❌ Critical: ALPACA_API_KEY or ALPACA_SECRET_KEY not set.")
        # For dry-run testing purposes
        if "--dry-run" in sys.argv:
            print("⚠️ Dry Run mode with no Keys - skipping API init.")
            api = None
        else:
            sys.exit(1)
    else:
        try:
            api = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            account = api.get_account()
            print(f"✅ Connected to Alpaca Paper Trading. Status: {account.status}")
        except Exception as e:
            print(f"❌ Alpaca Connection Error: {e}")
            if "--dry-run" in sys.argv:
                api = None
            else:
                sys.exit(1)

    # 3. Loop
    print("🚀 Starting Trading Loop...")
    while True:
        if api:
            check_signals(api, model, scaler)
        else:
            # Dry run loop
            data = fetch_live_data()
            if data:
                # determine expected features
                if hasattr(scaler, "feature_names_in_"):
                    expected_features = list(scaler.feature_names_in_)
                else:
                    expected_features = ESSENTIAL_FEATURES

                df_feat = calculate_features(data, expected_features)
                if not df_feat.empty:
                    print(f"Dry Run: Latest features calculated. Index: {df_feat.index[-1]}")
                    
                    # Also try inference for dry run
                    last_row = df_feat.iloc[[-1]]
                    X_raw = last_row
                    
                    try:
                        X_scaled_np = scaler.transform(X_raw)
                        if hasattr(scaler, "feature_names_in_"):
                            X_scaled_df = pd.DataFrame(X_scaled_np, columns=scaler.feature_names_in_)
                        else:
                            X_scaled_df = pd.DataFrame(X_scaled_np, columns=expected_features)
                            
                        # Select 19
                        X_model_input = X_scaled_df[MODEL_FEATURES].values
                        X_tensor = torch.FloatTensor(X_model_input)
                        
                        model.eval()
                        with torch.no_grad():
                            logits = model(X_tensor)
                            logit_15m = logits[0, 1].item()
                            prob = 1 / (1 + np.exp(-logit_15m))
                        print(f"Dry Run Inference: P(up)={prob:.4f}")
                    except Exception as e:
                        print(f"Dry Run Inference Failed: {e}")

        # Sleep 60s
        time.sleep(60)

if __name__ == "__main__":
    main()