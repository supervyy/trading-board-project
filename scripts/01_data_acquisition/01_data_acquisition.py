"""
This script downloads historical 1-minute adjusted bar data for QQQ and NVDA
from the Alpaca Market Data API for the date range 2020-01-01 → 2025-06-25.
It fetches the official US trading calendar, converts timestamps to US/Eastern,
filters only regular market hours (09:30–16:00), removes duplicates, and writes
the cleaned data as Parquet files.

Inputs:
- API Keys from ../../conf/keys.yaml
- Params (data path, start/end dates) from ../../conf/params.yaml
- Ticker list: ["QQQ", "NVDA"]

Outputs:
- One cleaned Parquet file per symbol under <DATA_PATH>/Bars_1m_adj/

Requirements:
- alpaca-py, pandas, pytz, pyyaml, pyarrow
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

import pandas as pd
from datetime import datetime
import pytz
import yaml
import os

# ============================================================
# LOAD CONFIG
# ============================================================
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

params = yaml.safe_load(open("../../conf/params.yaml"))
PATH_BARS = params["DATA_ACQUISITON"]["DATA_PATH"]
START_DATE = datetime.strptime(params["DATA_ACQUISITON"]["START_DATE"], "%Y-%m-%d")
END_DATE = datetime.strptime(params["DATA_ACQUISITON"]["END_DATE"], "%Y-%m-%d")

# Ensure output directory exists
os.makedirs(PATH_BARS, exist_ok=True)
# Ensure subdirectory for 1m bars exists
os.makedirs(os.path.join(PATH_BARS, "Bars_1m_adj"), exist_ok=True)


# ============================================================
# INIT ALPACA CLIENTS
# ============================================================
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY)

# ============================================================
# DEFINE TICKERS
# ============================================================
TICKERS = ["QQQ", "NVDA"]   # Only the assets needed for this project

# ============================================================
# GET OFFICIAL US TRADING CALENDAR
# ============================================================
cal_request = GetCalendarRequest(start=START_DATE, end=END_DATE)
calendar = trading_client.get_calendar(cal_request)

# Build lookup table: date → (market open, market close)
cal_map = {}
eastern = pytz.timezone("US/Eastern")

for c in calendar:
    open_dt = eastern.localize(c.open)
    close_dt = eastern.localize(c.close)
    cal_map[c.date] = (open_dt, close_dt)

# ============================================================
# HELPER: CHECK IF TIMESTAMP IS DURING REGULAR MARKET HOURS
# ============================================================
def check_open(ts):
    # Convert tz-naive timestamps to UTC → then to Eastern
    ts_eastern = ts.tz_convert(eastern) if ts.tzinfo else ts.tz_localize("UTC").astimezone(eastern)

    d = ts_eastern.date()
    if d not in cal_map:
        return False

    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt

# ============================================================
# MAIN DATA ACQUISITION LOOP
# ============================================================
print(f"Fetching 1m bars for QQQ + NVDA from {START_DATE} to {END_DATE}")

for symbol in TICKERS:
    print(f"Downloading {symbol}...")

    # Step 1: Request 1-minute bars
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        adjustment=Adjustment.ALL,
        start=START_DATE,
        end=END_DATE
    )

    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()

    # Step 2: Remove duplicate symbols column (Alpaca adds it)
    df = df.drop(columns=["symbol"], errors="ignore")

    # Step 3: Mark only market-open timestamps
    df["is_open"] = df["timestamp"].map(check_open)

    # Step 4: Filter Regular Trading Hours
    df = df[df["is_open"]]

    # Step 5: Remove helper column
    df = df.drop(columns=["is_open"])

    # Step 6: Save cleaned parquet (try engines, otherwise fallback to CSV)
    out_dir = os.path.join(PATH_BARS, "Bars_1m_adj")
    out_path = os.path.join(out_dir, f"{symbol}.parquet")

    saved_as_parquet = False
    try:
        # Prefer pyarrow, then fastparquet. Use importlib to avoid import errors at module import time.
        import importlib

        if importlib.util.find_spec("pyarrow") is not None:
            df.to_parquet(out_path, index=False, engine="pyarrow")
            saved_as_parquet = True
        elif importlib.util.find_spec("fastparquet") is not None:
            df.to_parquet(out_path, index=False, engine="fastparquet")
            saved_as_parquet = True
        else:
            # No parquet engine available
            raise ImportError("No parquet engine ('pyarrow' or 'fastparquet') installed")
    except Exception as e:
        # Fallback: write CSV and inform the user how to enable parquet support
        csv_out = os.path.join(out_dir, f"{symbol}.csv")
        df.to_csv(csv_out, index=False)
        print(f"Warning: could not write parquet ({e}). Saved CSV to: {csv_out}")
        print("To enable Parquet output install pyarrow (recommended):\n    pip install pyarrow")

    if saved_as_parquet:
        print(f"Saved: {out_path}")

print("Data acquisition completed successfully.")
