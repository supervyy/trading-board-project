"""
This script downloads historical 1-minute adjusted bar data for QQQ, NVDA,
AAPL, MSFT, GOOGL and AMZN from the Alpaca Market Data API for a configured
date range.

It uses direct HTTP requests (requests lib) to the Alpaca API to avoid SDK
auth issues, uses the official US market calendar to keep only regular
trading session bars (including short days), and writes one cleaned
Parquet file per symbol.

Inputs:
- API keys from ../../conf/keys.yaml
- Parameters (DATA_PATH, START_DATE) from ../../conf/params.yaml
- Ticker list: ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]

Outputs:
- One Parquet per symbol under <DATA_PATH> (e.g. ../../data/raw/QQQ_1m.parquet)
- Columns: timestamp, open, high, low, close, volume, trade_count, vwap
  (timestamp stored as ISO string: YYYY-MM-DD HH:MM:SS)
"""

import requests
import pandas as pd
from datetime import datetime, time, timezone
import pytz
import yaml
import os
import time as time_module
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

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
    # Create timezone-aware datetime objects in UTC
    START_DATE = datetime.strptime(
        params["DATA_ACQUISITON"]["START_DATE"], "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)
    end_date_str = params["DATA_ACQUISITON"].get("END_DATE")
    if end_date_str:
        END_DATE = datetime.strptime(end_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        END_DATE = datetime.now(timezone.utc)
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
TICKERS = ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]

# ============================================================
# BUILD OFFICIAL US MARKET CALENDAR (REGULAR TRADING HOURS)
# ============================================================
trading_client = TradingClient(API_KEY, SECRET_KEY)

# Get market calendar for the requested period (dates only)
cal_request = GetCalendarRequest(start=START_DATE.date(), end=END_DATE.date())
calendar = trading_client.get_calendar(cal_request)

# Build lookup table (date → open_dt, close_dt) in US/Eastern
cal_map = {}
eastern = pytz.timezone("US/Eastern")

for c in calendar:
    # c.open and c.close are naive datetimes in ET
    open_dt = eastern.localize(c.open)
    close_dt = eastern.localize(c.close)
    cal_map[c.date] = (open_dt, close_dt)


def check_open(ts: pd.Timestamp) -> bool:
    """
    Return True if the timestamp lies within the regular session
    for its day based on the official exchange calendar.
    """
    # ts is expected to be timezone-aware (UTC) after pd.to_datetime(..., utc=True)
    ts_eastern = ts.tz_convert(eastern) if ts.tzinfo else ts.tz_localize("UTC").astimezone(eastern)
    d = ts_eastern.date()
    if d not in cal_map:
        return False
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt


# ============================================================
# MAIN DATA ACQUISITION LOOP
# ============================================================

print(
    f"Fetching 1-minute bars for {TICKERS} "
    f"from {START_DATE.date()} to {END_DATE.date()}"
)

for symbol in TICKERS:
    print(f"\nDownloading 1m bars for {symbol}...")
    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY
    }

    # Format dates for API (date strings)
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
            last_bar_time = bars[-1]["t"]
            print(
                f"  Fetched {len(bars)} bars. Total: {len(all_bars)}. "
                f"Last timestamp: {last_bar_time}"
            )

            page_token = data.get("next_page_token")
            if not page_token:
                break

            time_module.sleep(0.5)  # Rate limit niceness

        except Exception as e:
            print(f"  Exception fetching data: {e}")
            break

    if not all_bars:
        print(
            f"  No data fetched for {symbol}. "
            f"Check your API keys or date range."
        )
        continue

    # --------------------------------------------------------
    # Process Data
    # --------------------------------------------------------
    df = pd.DataFrame(all_bars)

    # Rename columns (including trade_count and vwap)
    df = df.rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "trade_count",
            "vw": "vwap",
        }
    )

    # Keep relevant columns
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]

    # Convert timestamp to timezone-aware datetime (UTC)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    print(f"  Raw data rows: {len(df)}")
    print(f"  Raw date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Filter using official exchange calendar (regular trading hours incl. short days)
    print("  Filtering for regular market hours based on exchange calendar...")

    df["is_open"] = df["timestamp"].map(check_open)
    df_filtered = df[df["is_open"]].copy()

    print(
        f"  Rows after filtering: {len(df_filtered)} "
        f"(Dropped {len(df) - len(df_filtered)})"
    )
    if not df_filtered.empty:
        print(
            f"  Filtered date range: "
            f"{df_filtered['timestamp'].min()} to {df_filtered['timestamp'].max()}"
        )

    df_filtered.drop(columns=["is_open"], inplace=True)

    df_filtered = df_filtered.sort_values("timestamp")

    before_dedup = len(df_filtered)
    df_filtered = df_filtered.drop_duplicates(subset=["timestamp"])
    after_dedup = len(df_filtered)

    if before_dedup != after_dedup:
        print(f"  Removed {before_dedup - after_dedup} duplicate timestamps")
    if df_filtered.empty:
        print(f"  ⚠️ Skipping {symbol}: No market hour data after filtering")
        continue


    # Optional: store timestamp as string for compatibility (as in your pipeline)
    df_filtered["timestamp"] = df_filtered["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save as Parquet
    out_path = os.path.join(PATH_BARS, f"{symbol}_1m.parquet")
    df_filtered.to_parquet(out_path, index=False)

    print(f"  Saved cleaned file: {out_path}")
    print(f"  Rows: {len(df_filtered)}")

print("\nData acquisition completed.")
