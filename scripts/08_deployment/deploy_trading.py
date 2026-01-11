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
import importlib.util
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# Alpaca Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# --- Project Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import Model Class Dynamically
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

# --- Configuration Loading ---
PARAMS_PATH = PROJECT_ROOT / "conf" / "params.yaml"
TRADING_CONFIG_PATH = PROJECT_ROOT / "conf" / "trading.yaml"
KEYS_PATH = PROJECT_ROOT / "conf" / "keys.yaml"

def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

CONF_PARAMS = load_config(PARAMS_PATH)
CONF_TRADING = load_config(TRADING_CONFIG_PATH)

# Merged Config
TRADING_OPTS = CONF_TRADING.get("TRADING", {})
RISK_OPTS = CONF_TRADING.get("RISK_MANAGEMENT", {})
EXEC_OPTS = CONF_TRADING.get("EXECUTION", {})

SYMBOL = TRADING_OPTS.get("SYMBOL", "QQQ")
SYMBOLS_DATA = ["QQQ", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]
TECH_SYMBOLS = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN"]

MODEL_PATH = PROJECT_ROOT / "models" / "feed_forward" / "multihorizon_nn.pt"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

# Feature List (Must match Training)
MODEL_FEATURES = [
    "ema_diff", "return_5", "realized_vol_10", "volume_norm", "volume_acceleration",
    "NVDA_return_5", "NVDA_volume_norm", "divergence_NVDA_QQQ_5", "corr_QQQ_NVDA_15",
    "relative_strength", "momentum_spread_5", "tech_unanimity", "max_divergence",
    "high_vol_regime", "low_corr_regime", "overextended_up", "overextended_down",
    "bid_ask_spread_proxy", "is_15_30_16_00"
]

# --- Helper Classes ---

@dataclass
class Position:
    symbol: str
    entry_time: datetime.datetime
    entry_price: float
    qty: float
    stop_price: float
    take_profit_price: float
    highest_price: float
    status: str = "OPEN"
    order_id: str = ""
    exit_reason: str = ""

class DataProvider:
    def __init__(self, use_alpaca: bool, alpaca_data_client=None):
        self.use_alpaca = use_alpaca
        self.alpaca_data = alpaca_data_client

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch last 5 days of 1-minute data."""
        if self.use_alpaca and self.alpaca_data:
             return self._fetch_alpaca()
        return self._fetch_yfinance()

    def _fetch_yfinance(self) -> Dict[str, pd.DataFrame]:
        data = {}
        tickers = " ".join(SYMBOLS_DATA)
        try:
            # interval="1m" allows max 7 days lookback
            df_all = yf.download(tickers, period="5d", interval="1m", progress=False, group_by='ticker', auto_adjust=False)
            
            if len(SYMBOLS_DATA) > 1:
                for sym in SYMBOLS_DATA:
                    try:
                        df_sym = df_all[sym].copy()
                        if df_sym.empty: continue
                        
                        # Timezone handling
                        if df_sym.index.tz is None:
                            df_sym.index = df_sym.index.tz_localize('UTC').tz_convert('US/Eastern')
                        else:
                            df_sym.index = df_sym.index.tz_convert('US/Eastern')
                        
                        # Filter RTH
                        df_sym = df_sym.between_time('09:30', '16:00')
                        data[sym] = df_sym
                    except Exception:
                        pass
            else:
                # Handle single ticker case if needed
                pass
        except Exception as e:
            print(f"Data Fetch Error (YF): {e}")
            return {}
        
        return self._align_data(data)

    def _fetch_alpaca(self) -> Dict[str, pd.DataFrame]:
        # Placeholder for Alpaca Data API
        # Implementation depends on subscription level (SIP vs IEX)
        print("Existing Alpaca Data implementation skipped for brevity/paid-data constraint, fallback to YF.")
        return self._fetch_yfinance()

    def _align_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not data: return {}
        common = data[SYMBOL].index
        for sym in data:
            common = common.intersection(data[sym].index)
        
        aligned = {}
        for sym in data:
            aligned[sym] = data[sym].loc[common]
        return aligned

class FeatureEngineer:
    @staticmethod
    def calculate(df_dict: Dict[str, pd.DataFrame], scaler_features: List[str]) -> pd.DataFrame:
        """Standard feature engineering pipeline matching training."""
        if SYMBOL not in df_dict: return pd.DataFrame()
        
        # 1. Core QQQ
        df_qqq = df_dict[SYMBOL].copy()
        close = df_qqq["Close"]
        vol = df_qqq["Volume"]
        
        # Basic
        df_qqq["ema_5"] = close.ewm(span=5, adjust=False).mean()
        df_qqq["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df_qqq["ema_diff"] = df_qqq["ema_5"] - df_qqq["ema_20"]
        
        df_qqq["return_5"] = close.pct_change(5, fill_method=None)
        df_qqq["realized_vol_10"] = close.pct_change(fill_method=None).rolling(10).std()
        
        # Norm
        vol_mean = vol.rolling(60).mean().replace(0, np.nan)
        df_qqq["volume_norm"] = vol / vol_mean
        df_qqq["volume_acceleration"] = df_qqq["volume_norm"].diff(5)
        
        df_qqq["bid_ask_spread_proxy"] = (df_qqq["High"] - df_qqq["Low"]) / close
        
        df_qqq["overextended_up"] = (close > df_qqq["ema_20"] * 1.005).astype(int)
        df_qqq["overextended_down"] = (close < df_qqq["ema_20"] * 0.995).astype(int)
        
        # Vol Regime
        v = df_qqq["realized_vol_10"]
        df_qqq["high_vol_regime"] = (v > v.rolling(100, min_periods=50).quantile(0.8)).astype(int)
        
        # Time
        idx_et = df_qqq.index
        df_qqq["is_15_30_16_00"] = ((idx_et.hour == 15) & (idx_et.minute >= 30)).astype(int)
        
        # 2. Tech Features & Cross Asset
        tech_rets = []
        for sym in TECH_SYMBOLS:
            if sym in df_dict:
                dft = df_dict[sym].copy()
                cl = dft["Close"]
                vl = dft["Volume"]
                # Prefixed
                df_qqq[f"{sym}_return_5"] = cl.pct_change(5, fill_method=None)
                # NVDA specific
                if sym == "NVDA":
                    vm = vl.rolling(60).mean().replace(0, np.nan)
                    df_qqq["NVDA_volume_norm"] = vl / vm
                    
                tech_rets.append(f"{sym}_return_5")

        # 3. Cross Stats
        if tech_rets:
            # We need to make sure we don't have NaNs from join
            # Here we are just assigning columns to df_qqq assuming aligned index
            
            # Avg Tech Return
            # Check availability
            valid_cols = [c for c in tech_rets if c in df_qqq.columns]
            if valid_cols:
                avg_ret = df_qqq[valid_cols].mean(axis=1)
                df_qqq["relative_strength"] = df_qqq["return_5"] - avg_ret
                
                # Unanimity
                signs_tech = np.sign(df_qqq[valid_cols])
                sign_q = np.sign(df_qqq["return_5"])
                # vector broadcast
                agree = signs_tech.eq(sign_q, axis=0).sum(axis=1)
                df_qqq["tech_unanimity"] = agree / len(valid_cols)
                
                df_qqq["momentum_spread_5"] = df_qqq[valid_cols].std(axis=1)
                
                # Divergences
                divs = df_qqq[valid_cols].sub(df_qqq["return_5"], axis=0).abs()
                df_qqq["max_divergence"] = divs.max(axis=1)
                
                if "NVDA_return_5" in df_qqq.columns:
                    df_qqq["divergence_NVDA_QQQ_5"] = df_qqq["NVDA_return_5"] - df_qqq["return_5"]
                    df_qqq["corr_QQQ_NVDA_15"] = df_qqq["return_5"].rolling(15).corr(df_qqq["NVDA_return_5"])
                    df_qqq["low_corr_regime"] = (df_qqq["corr_QQQ_NVDA_15"] < 0.5).astype(int)

        # Fill NaNs
        df_qqq = df_qqq.ffill().fillna(0.0)
        
        # 4. Align with Scaler
        for f in scaler_features:
            if f not in df_qqq.columns:
                df_qqq[f] = 0.0
                
        return df_qqq[scaler_features]

class TradingManager:
    def __init__(self, api: TradingClient, model, scaler):
        self.api = api
        self.model = model
        self.scaler = scaler
        
        self.positions: List[Position] = []
        self.last_trade_time = None
        
        # Risk Params
        self.sl_type = RISK_OPTS.get("STOP_LOSS_TYPE", "pct")
        self.sl_val = RISK_OPTS.get("STOP_LOSS_VALUE", 0.004)
        self.tp_type = RISK_OPTS.get("TAKE_PROFIT_TYPE", "rr")
        self.tp_val = RISK_OPTS.get("TAKE_PROFIT_VALUE", 1.5)
        self.trailing = RISK_OPTS.get("TRAILING_STOP_ENABLED", False)
        self.trailing_pct = RISK_OPTS.get("TRAILING_STOP_PCT", 0.002)
        
        self.max_hold = TRADING_OPTS.get("MAX_HOLD_MINUTES", 15)
        self.cooldown = TRADING_OPTS.get("COOLDOWN_MINUTES", 5)
        self.max_pos = TRADING_OPTS.get("MAX_POSITIONS", 1)
        self.one_per_bar = TRADING_OPTS.get("ONE_TRADE_PER_BAR", True)
        self.risk_per_trade = RISK_OPTS.get("RISK_PER_TRADE_PCT", 0.005)
        
        self.equity = 100000.0 # Default fallback
        self.sync_account()

    def sync_account(self):
        if self.api:
            try:
                acct = self.api.get_account()
                self.equity = float(acct.equity)
            except Exception as e:
                print(f"Error syncing account: {e}")

    def calculate_size(self, entry_price: float, stop_price: float) -> float:
        """Calculate position size based on risk."""
        risk_amt = self.equity * self.risk_per_trade
        dist = abs(entry_price - stop_price)
        if dist == 0: return 0
        
        qty = risk_amt / dist
        
        # Max Notional Cap
        max_notional = self.equity * RISK_OPTS.get("MAX_NOTIONAL_PCT", 0.20)
        qty_cap = max_notional / entry_price
        
        final_qty = min(qty, qty_cap)
        return max(1, int(final_qty)) # At least 1 share

    def check_exits(self, current_price: float, current_time: datetime.datetime):
        """Check all open positions for exit signals."""
        for pos in self.positions[:]:
            if pos.status != "OPEN": continue
            
            # Updates
            pos.highest_price = max(pos.highest_price, current_price)
            
            # Trailing Stop Update
            if self.trailing:
                new_stop = pos.highest_price * (1 - self.trailing_pct)
                if new_stop > pos.stop_price:
                    pos.stop_price = new_stop
            
            # Check Conditions
            exit_signal = None
            if current_price <= pos.stop_price:
                exit_signal = "StopLoss"
            elif current_price >= pos.take_profit_price:
                exit_signal = "TakeProfit"
            else:
                elapsed = (current_time - pos.entry_time).total_seconds() / 60
                if elapsed >= self.max_hold:
                    exit_signal = "MaxHold"
                    
            if exit_signal:
                print(f"📉 EXIT {pos.symbol}: {exit_signal} @ {current_price:.2f} (Entry: {pos.entry_price:.2f})")
                self.close_position(pos, current_price, exit_signal)

    def close_position(self, pos: Position, price: float, reason: str):
        if self.api:
            try:
                # If we had bracket orders, we might need to cancel them or just sell flat
                # Simple approach: Market Sell all
                self.api.close_position(pos.symbol)
            except Exception as e:
                print(f"Error closing position alpaca: {e}")
        
        pos.status = "CLOSED"
        pos.exit_reason = reason
        self.positions.remove(pos)

    def entry_signal(self, prob: float, price: float, current_time: datetime.datetime, atr: float = 0.0):
        """Handle Entry Logic."""
        # Check constraints
        if len(self.positions) >= self.max_pos:
            return
        
        if self.last_trade_time:
            delta = (current_time - self.last_trade_time).total_seconds() / 60
            if delta < self.cooldown:
                return

        # Calc Levels
        if self.sl_type == "atr" and atr > 0:
            stop_dist = atr * 2.0 # Fixed multiplier for now
        else:
            stop_dist = price * self.sl_val
            
        stop_price = price - stop_dist
        
        if self.tp_type == "rr":
            tp_dist = stop_dist * self.tp_val
        else:
            tp_dist = price * self.tp_val # Interpreting as val directly
            
        tp_price = price + tp_dist
        
        # Sizing
        qty = self.calculate_size(price, stop_price)
        if qty < 1:
            print("Risk too high or equity too low for min qty.")
            return

        print(f"🚀 ENTRY LONG {SYMBOL}: Prob {prob:.2f} | Price {price:.2f} | Qty {qty} | SL {stop_price:.2f} | TP {tp_price:.2f}")

        # Execute
        if self.api:
            try:
                # Bracket Order
                req = MarketOrderRequest(
                    symbol=SYMBOL,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    stop_loss={'stop_price': round(stop_price, 2)},
                    take_profit={'limit_price': round(tp_price, 2)}
                )
                res = self.api.submit_order(req)
                print(f"Order Submitted: {res.id}")
            except Exception as e:
                print(f"Order Failed: {e}")
                return

        # Track Internally (Paper/Live sync)
        # Note: If API, ideally we verify fill, but for now we assume immediate fill for tracking logic if not confirming orders
        pos = Position(
            symbol=SYMBOL,
            entry_time=current_time,
            entry_price=price,
            qty=qty,
            stop_price=stop_price,
            take_profit_price=tp_price,
            highest_price=price
        )
        self.positions.append(pos)
        self.last_trade_time = current_time


def main():
    print("--- Advanced Trading Deployment (QQQ) ---")
    
    # 1. Credentials
    keys = {}
    if KEYS_PATH.exists():
        with open(KEYS_PATH) as f:
            y = yaml.safe_load(f)
            keys = y.get("KEYS", {})
    
    API_KEY = keys.get("APCA-API-KEY-ID-Data") or os.getenv("ALPACA_API_KEY")
    SECRET_KEY = keys.get("APCA-API-SECRET-KEY-Data") or os.getenv("ALPACA_SECRET_KEY")
    
    api = None
    if EXEC_OPTS.get("DRY_RUN", True):
        print("⚠️ DRY RUN MODE. No real orders.")
    else:
        if API_KEY and SECRET_KEY:
            try:
                api = TradingClient(API_KEY, SECRET_KEY, paper=True)
                print("✅ Connected to Alpaca.")
            except Exception as e:
                print(f"❌ Connection Failed: {e}")
                sys.exit(1)
        else:
            print("❌ No API Keys found.")
            sys.exit(1)

    # 2. Model
    try:
        scaler = joblib.load(SCALER_PATH)
        in_dim = len(MODEL_FEATURES)
        model = MultiHorizonMLP(in_dim=in_dim, out_dim=3)
        state = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(state)
        model.eval()
        print("✅ Model Locked & Loaded.")
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        sys.exit(1)

    # 3. Helpers
    data_provider = DataProvider(use_alpaca=EXEC_OPTS.get("USE_ALPACA_DATA", False))
    manager = TradingManager(api, model, scaler)
    
    print("Waiting for next minute...")
    
    # 4. Loop
    while True:
        # Simple mechanism to run once per minute
        # Wait for seconds == 01 to allow data to settle
        # In prod, use standard wait or cron
        t = datetime.datetime.now(pytz.utc)
        
        # Data Fetch
        data_dict = data_provider.fetch_data()
        if not data_dict or SYMBOL not in data_dict:
            print("Tick: Data not available.")
            time.sleep(10)
            continue
            
        df_latest = data_dict[SYMBOL]
        current_price = df_latest["Close"].iloc[-1]
        current_time = df_latest.index[-1] # This is likely 16:00 if market closed, or latest minute
        
        # Calculate Features
        if hasattr(scaler, "feature_names_in_"):
            feats = list(scaler.feature_names_in_)
        else:
            # Fallback if scaler metadata missing (unlikely)
            print("Scaler metadata missing, cannot process.")
            time.sleep(60)
            continue
            
        df_features = FeatureEngineer.calculate(data_dict, feats)
        
        if not df_features.empty:
            last_row = df_features.iloc[[-1]]
            
            # Inference
            try:
                x_np = scaler.transform(last_row)
                x_df = pd.DataFrame(x_np, columns=feats)
                # Filter model features
                x_model = x_df[MODEL_FEATURES].values
                x_tensor = torch.FloatTensor(x_model)
                
                with torch.no_grad():
                    logits = model(x_tensor)
                    prob = 1 / (1 + np.exp(-logits[0, 1].item()))
                    
                    
                # Market Open Check
                is_open = True
                next_open = None
                if api:
                    try:
                        clock = api.get_clock()
                        if not clock.is_open:
                            is_open = False
                            next_open = clock.next_open
                    except Exception: pass
                
                status_str = "OPEN" if is_open else "CLOSED"

                # Standard Output
                print(f"[{current_time}] Status: {status_str} | Price: {current_price:.2f} | Prob(Up): {prob:.4f}")
                
                if is_open:
                    # 1. Manage Exits
                    manager.check_exits(current_price, current_time)
                    
                    # 2. Check Entries
                    threshold = TRADING_OPTS.get("PROB_THRESHOLD", 0.55)
                    if prob >= threshold:
                        manager.entry_signal(prob, current_price, current_time)
                else:
                    if next_open:
                        print(f"Market Closed. Next Open: {next_open}. Logic skipped.")
                    
            except Exception as e:
                print(f"Inference/Logic Error: {e}")
                import traceback
                traceback.print_exc()

        # Coarse Wait
        # Wait until next minute
        now = datetime.datetime.now()
        sleep_sec = 60 - now.second + 2 # +2s buffer
        time.sleep(sleep_sec)

if __name__ == "__main__":
    main()
