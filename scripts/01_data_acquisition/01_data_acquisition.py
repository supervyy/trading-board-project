"""
This script downloads historical 5-minute adjusted bar data for QQQ and NVDA
from the Alpaca Market Data API for a configured date range (e.g. 2020-01-01 → 2025-06-25).
It fetches the official US trading calendar, converts timestamps to US/Eastern,
filters only regular market hours (09:30–16:00 ET), removes non-RTH bars,
and writes one cleaned Parquet file per symbol.

Inputs:
- API keys from ../../conf/keys.yaml
- Parameters (DATA_PATH, START_DATE, END_DATE) from ../../conf/params.yaml
- Ticker list: ["QQQ", "NVDA"]

Outputs:
- One Parquet per symbol under <DATA_PATH> (e.g. ../../data/raw/QQQ_5m.parquet)

Requirements:
- alpaca-py, pandas, pytz, pyyaml, pyarrow
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment

import pandas as pd
from datetime import datetime
import pytz
import yaml
import os

# ============================================================
# LOAD CONFIGURATION (API KEYS + PARAMETERS)
# ============================================================
# Load API credentials from YAML configuration file
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

# Load data acquisition parameters from YAML configuration file
params = yaml.safe_load(open("../../conf/params.yaml"))
PATH_BARS = params["DATA_ACQUISITON"]["DATA_PATH"]
START_DATE = datetime.strptime(params["DATA_ACQUISITON"]["START_DATE"], "%Y-%m-%d")
END_DATE = datetime.strptime(params["DATA_ACQUISITON"]["END_DATE"], "%Y-%m-%d")

# Ensure output directory exists (e.g. ../../data/raw)
os.makedirs(PATH_BARS, exist_ok=True)

# ============================================================
# INITIALIZE ALPACA CLIENTS
# ============================================================
# Historical data client (market data)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Trading client (used here only to fetch the official US trading calendar)
trading_client = TradingClient(API_KEY, SECRET_KEY)

# ============================================================
# DEFINE TICKERS FOR THE PROJECT
# ============================================================
TICKERS = ["QQQ", "NVDA"]  # Main ETF + cross-asset driver

# ============================================================
# GET OFFICIAL US TRADING CALENDAR
# ============================================================
# Request the market calendar for the full date range
cal_request = GetCalendarRequest(start=START_DATE, end=END_DATE)
calendar = trading_client.get_calendar(cal_request)

# Build a lookup map: date -> (market_open_datetime, market_close_datetime) in US/Eastern
cal_map = {}
eastern = pytz.timezone("US/Eastern")

for c in calendar:
    # c.open and c.close are naive datetime objects in ET
    open_dt = eastern.localize(c.open)
    close_dt = eastern.localize(c.close)
    cal_map[c.date] = (open_dt, close_dt)

# ============================================================
# HELPER FUNCTION: CHECK IF TIMESTAMP IS DURING REGULAR HOURS
# ============================================================
def is_regular_trading_hour(ts):
    """
    Returns True if the given timestamp falls within regular trading hours
    (market open to close) on a valid trading day, False otherwise.
    """
    # Convert timestamp to US/Eastern (Alpaca timestamps are in UTC by default)
    if ts.tzinfo:
        ts_eastern = ts.astimezone(eastern)
    else:
        ts_eastern = ts.tz_localize("UTC").astimezone(eastern)

    d = ts_eastern.date()
    if d not in cal_map:
        return False

    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt

# ============================================================
# MAIN DATA ACQUISITION LOOP (5-MINUTE BARS)
# ============================================================
print(f"Fetching 5-minute bars for {TICKERS} from {START_DATE} to {END_DATE}")

for symbol in TICKERS:
    print(f"Downloading 5m bars for {symbol}...")

    # Step 1: Build request for 5-minute historical bars
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),  # 5-minute timeframe
        adjustment=Adjustment.ALL,                     # adjust for splits/dividends
        start=START_DATE,
        end=END_DATE,
    )

    # Step 2: Fetch bars from Alpaca Market Data
    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()  # MultiIndex -> columns, keep 'timestamp'

    # Step 3: Drop 'symbol' column if present (not needed for single-symbol files)
    df = df.drop(columns=["symbol"], errors="ignore")

    # Step 4: Mark which rows occur during regular trading hours
    df["is_rth"] = df["timestamp"].map(is_regular_trading_hour)

    # Step 5: Filter only regular trading hours (RTH)
    df = df[df["is_rth"]].copy()

    # Step 6: Drop helper column
    df.drop(columns=["is_rth"], inplace=True)

    # Step 7: Save cleaned DataFrame as Parquet file
    out_path = os.path.join(PATH_BARS, f"{symbol}_5m.parquet")
    df.to_parquet(out_path, index=False)

    print(f"Saved: {out_path}")

print("5-minute data acquisition completed successfully.")
