"""
This script downloads historical 1-minute adjusted bar data for QQQ and NVDA
from the Alpaca Market Data API for a configured date range.
It uses direct HTTP requests (requests lib) to the Alpaca API to avoid SDK auth issues,
converts timestamps to US/Eastern, filters only regular market hours (09:30–16:00 ET),
and writes one cleaned Parquet file per symbol.

Inputs:
- API keys from ../../conf/keys.yaml
- Parameters (DATA_PATH, START_DATE, END_DATE) from ../../conf/params.yaml
- Ticker list: ["QQQ", "NVDA"]

Outputs:
- One Parquet per symbol under <DATA_PATH> (e.g. ../../data/raw/QQQ_1m.parquet)
"""

import requests
import pandas as pd
from datetime import datetime, time
import pytz
import yaml
import os
import time as time_module

# ============================================================
# LOAD CONFIGURATION (API KEYS + PARAMETERS)
# ============================================================
# Load API credentials from YAML configuration file
try:
    keys = yaml.safe_load(open("../../conf/keys.yaml"))
    API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
    SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]
except Exception as e:
    print(f"Error loading keys.yaml: {e}")
    exit(1)

# Load data acquisition parameters from YAML configuration file
try:
    params = yaml.safe_load(open("../../conf/params.yaml"))
    PATH_BARS = params["DATA_ACQUISITON"]["DATA_PATH"]
    START_DATE = datetime.strptime(params["DATA_ACQUISITON"]["START_DATE"], "%Y-%m-%d")
    END_DATE = datetime.strptime(params["DATA_ACQUISITON"]["END_DATE"], "%Y-%m-%d")
except Exception as e:
    print(f"Error loading params.yaml: {e}")
    exit(1)

# Ensure output directory exists
os.makedirs(PATH_BARS, exist_ok=True)

# ============================================================
# ALPACA API CONFIGURATION
# ============================================================
BASE_URL = "https://data.alpaca.markets/v2"

# ============================================================
# DEFINE TICKERS
# ============================================================
TICKERS = ["QQQ", "NVDA"]

# ============================================================
# DEFINE REGULAR TRADING HOURS
# ============================================================
eastern = pytz.timezone("US/Eastern")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

def is_regular_trading_hour(ts_str):
    """
    Returns True if the given timestamp falls within regular trading hours
    (09:30-16:00 ET) on a weekday.
    """
    # Parse timestamp string (Alpaca returns ISO format)
    ts = pd.to_datetime(ts_str)
    
    # Convert to US/Eastern
    if ts.tzinfo:
        ts_eastern = ts.astimezone(eastern)
    else:
        ts_eastern = ts.tz_localize("UTC").astimezone(eastern)

    # Check if it's a weekday (Monday=0, Sunday=6)
    if ts_eastern.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if time is within market hours
    ts_time = ts_eastern.time()
    return MARKET_OPEN <= ts_time < MARKET_CLOSE

# ============================================================
# MAIN DATA ACQUISITION LOOP
# ============================================================
print(f"Fetching 1-minute bars for {TICKERS} from {START_DATE.date()} to {END_DATE.date()}")

for symbol in TICKERS:
    print(f"\nDownloading 1m bars for {symbol}...")
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }
    
    # Format dates for API
    start_str = START_DATE.strftime("%Y-%m-%d")
    end_str = END_DATE.strftime("%Y-%m-%d")
    
    all_bars = []
    page_token = None
    
    while True:
        # Request parameters
        params_req = {
            "start": start_str,
            "end": end_str,
            "timeframe": "1Min",
            "limit": 10000,
            "adjustment": "all",
            "feed": "iex",  # Free IEX feed
            "sort": "asc"   # Explicitly request oldest first
        }
        
        if page_token:
            params_req["page_token"] = page_token
            
        try:
            response = requests.get(
                f"{BASE_URL}/stocks/{symbol}/bars", 
                headers=headers, 
                params=params_req
            )
            
            if response.status_code != 200:
                print(f"  Error: {response.status_code} - {response.text}")
                break
                
            data = response.json()
            bars = data.get("bars", [])
            
            if not bars:
                break
                
            all_bars.extend(bars)
            
            # Progress update
            last_bar_time = bars[-1]['t']
            print(f"  Fetched {len(bars)} bars. Total: {len(all_bars)}. Last timestamp: {last_bar_time}")
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
                
            time_module.sleep(0.5) # Rate limit niceness
            
        except Exception as e:
            print(f"  Exception fetching data: {e}")
            break

    if not all_bars:
        print(f"  No data fetched for {symbol}. Check your API keys or date range.")
        continue

    # Process Data
    df = pd.DataFrame(all_bars)
    
    # Rename columns
    df = df.rename(columns={
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    })
    
    # Keep relevant columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    # Convert timestamp immediately to see range
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print(f"  Raw data rows: {len(df)}")
    print(f"  Raw date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save RAW data for debugging
    raw_out_path = os.path.join(PATH_BARS, f"{symbol}_raw_debug.parquet")
    df.to_parquet(raw_out_path, index=False)
    print(f"  Saved RAW debug file: {raw_out_path}")

    # Filter RTH
    print(f"  Filtering for regular trading hours (09:30-16:00 ET)...")
    
    # Debug: Check a sample of recent rows to see why they might fail
    recent_rows = df.tail(5).copy()
    for idx, row in recent_rows.iterrows():
        ts = row['timestamp']
        is_rth = is_regular_trading_hour(ts)
        print(f"    Debug Check {ts}: RTH={is_rth}")

    df["is_rth"] = df["timestamp"].map(is_regular_trading_hour)
    df_filtered = df[df["is_rth"]].copy()
    
    print(f"  Rows after filtering: {len(df_filtered)} (Dropped {len(df) - len(df_filtered)})")
    if not df_filtered.empty:
        print(f"  Filtered date range: {df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}")
    
    df_filtered.drop(columns=["is_rth"], inplace=True)
    
    # Save as Parquet
    out_path = os.path.join(PATH_BARS, f"{symbol}_1m.parquet")
    df_filtered.to_parquet(out_path, index=False)

    print(f"  Saved cleaned file: {out_path}")
    print(f"  Rows: {len(df_filtered)}")

print("\nData acquisition completed.")