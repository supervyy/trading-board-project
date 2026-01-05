"""
FF Trade Backtest (Risk & PnL) – Realistic Simulation
=====================================================

Features:
- Intrabar Stop-Loss / Take-Profit using High/Low (from raw data).
- Validates strategies with Fees and Slippage.
- Multiple Positions support (Scale-in).
- No Lookahead bias.

Inputs:
- data/processed/test.parquet (Features, Targets, Timestamps)
- data/raw/QQQ_1m.parquet (OHLC for execution simulation)
- models/feed_forward/multihorizon_nn.pt (Model)
- models/scaler.pkl (Scaler)

Config:
- Automatically loads defaults from conf/trading.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path
import importlib.util
import datetime
import pytz

import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

# --- Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(PROJECT_ROOT))

# Helper to load model class
try:
    model_script_path = PROJECT_ROOT / "scripts" / "07_modeling" / "07_feed_forward.py"
    spec = importlib.util.spec_from_file_location("feed_forward_module", model_script_path)
    ff_module = importlib.util.module_from_spec(spec)
    sys.modules["feed_forward_module"] = ff_module
    spec.loader.exec_module(ff_module)
    MultiHorizonMLP = ff_module.MultiHorizonMLP
except Exception as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

# --- Classes ---

class Position:
    def __init__(self, entry_time, entry_price, qty, stop_loss, take_profit, reason="Signal"):
        self.entry_time = entry_time
        self.entry_price = float(entry_price)
        self.qty = int(qty)
        self.stop_loss = float(stop_loss)
        self.take_profit = float(take_profit)
        self.reason = reason
        self.highest_price = float(entry_price)
        self.status = "OPEN"
        self.exit_time = None
        self.exit_price = 0.0
        self.exit_reason = ""
        
    def check_exit(self, timestamp, low, high, close, trading_cfg):
        """
        Check if position should close based on Intrabar High/Low.
        Conservative assumption: Only one bound is hit per bar.
        Rules:
        - If Low <= SL: Stopped Out
        - If High >= TP: Take Profit
        - If both: Worst case first (Stop Loss) unless High happens to be Open (unlikely to know).
          Conservative: Assume SL hit if both hit.
        """
        # Update High Watermark
        self.highest_price = max(self.highest_price, high)
        
        # Trailing Logic
        trail_en = trading_cfg.get("TRAILING_STOP_ENABLED", False)
        trail_pct = trading_cfg.get("TRAILING_STOP_PCT", 0.0)
        
        effective_sl = self.stop_loss
        if trail_en and trail_pct > 0:
            trail_price = self.highest_price * (1.0 - trail_pct)
            effective_sl = max(self.stop_loss, trail_price)

        # 1. Stop Loss (Fixed or Trailing)
        if low <= effective_sl:
            self.exit(timestamp, effective_sl, "StopLoss" if effective_sl == self.stop_loss else "TrailingStop")
            return True
            
        # 2. Take Profit
        if high >= self.take_profit:
            self.exit(timestamp, self.take_profit, "TakeProfit")
            return True
            
        # 3. Time Exit will be handled by caller (Max Hold)
        return False

    def exit(self, timestamp, price, reason):
        self.status = "CLOSED"
        self.exit_time = timestamp
        self.exit_price = float(price)
        self.exit_reason = reason

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def run_backtest(args, df_merged, model, scaler, device='cpu'):
    print(f"Running simulation on {len(df_merged)} bars...")
    # Not used in this version
    pass 

def main():
    # Load config from YAML
    TRADING_CONFIG_PATH = PROJECT_ROOT / "conf" / "trading.yaml"
    config = {}
    if TRADING_CONFIG_PATH.exists():
        with open(TRADING_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f) or {}

    trading = config.get("TRADING", {})
    risk = config.get("RISK_MANAGEMENT", {})
    execution = config.get("EXECUTION", {})

    parser = argparse.ArgumentParser()
    # Defaults from YAML
    parser.add_argument("--threshold", type=float, default=trading.get("PROB_THRESHOLD", 0.55))
    parser.add_argument("--max-positions", type=int, default=trading.get("MAX_POSITIONS", 3))
    parser.add_argument("--hold-bars", type=int, default=trading.get("MAX_HOLD_MINUTES", 15))
    parser.add_argument("--start-capital", type=float, default=100000.0)
    parser.add_argument("--risk-pct", type=float, default=risk.get("RISK_PER_TRADE_PCT", 0.005)) 
    parser.add_argument("--sl-pct", type=float, default=risk.get("STOP_LOSS_VALUE", 0.004))
    parser.add_argument("--tp-rr", type=float, default=risk.get("TAKE_PROFIT_VALUE", 1.5))
    parser.add_argument("--slippage-bps", type=float, default=execution.get("SLIPPAGE_BPS", 2.0))
    parser.add_argument("--fee", type=float, default=execution.get("FEE_PER_ORDER", 0.0))
    parser.add_argument("--cooldown", type=int, default=trading.get("COOLDOWN_MINUTES", 5))
    
    # Trailing args
    parser.add_argument("--trailing-enabled", action='store_true', default=risk.get("TRAILING_STOP_ENABLED", False))
    parser.add_argument("--trailing-pct", type=float, default=risk.get("TRAILING_STOP_PCT", 0.0))
    
    args = parser.parse_args()

    # Paths
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    MODEL_PATH = PROJECT_ROOT / "models" / "feed_forward" / "multihorizon_nn.pt"
    SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
    X_TEST_PATH = DATA_DIR / "X_test_scaled.npy"
    TEST_PQ_PATH = DATA_DIR / "test.parquet"
    RAW_QQQ_PATH = RAW_DIR / "QQQ_1m.parquet"
    
    # 1. Load Data
    print("Loading data...")
    if not TEST_PQ_PATH.exists() or not RAW_QQQ_PATH.exists():
        print("Missing data files.")
        sys.exit(1)
        
    df_test = pd.read_parquet(TEST_PQ_PATH)
    df_raw = pd.read_parquet(RAW_QQQ_PATH)
    
    # Align Timestamps
    if "timestamp" not in df_test.columns:
        df_test = df_test.reset_index()
        if df_test.columns[0] in ["index", "Date", "Datetime"]:
             df_test.rename(columns={df_test.columns[0]: "timestamp"}, inplace=True)
    df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], utc=True)
    
    if "timestamp" not in df_raw.columns:
         df_raw = df_raw.reset_index()
         col0 = df_raw.columns[0]
         if col0 in ["index", "Date", "Datetime"]:
             df_raw.rename(columns={col0: "timestamp"}, inplace=True)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    
    # Load X_test
    X_test = np.load(X_TEST_PATH)
    df_test["orig_idx"] = np.arange(len(df_test))
    
    # Calculate EMA 200 on raw data (Trend Filter)
    df_raw["ema200"] = df_raw["close"].ewm(span=200, adjust=False).mean()
    
    # Merge Cleanly
    cols_to_merge = ["timestamp", "open", "high", "low", "close", "ema200"]
    df_merged = pd.merge(df_test, df_raw[cols_to_merge], on="timestamp", how="inner", suffixes=("", "_raw"))
    df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    
    valid_indices = df_merged["orig_idx"].values
    X_valid = X_test[valid_indices]
    
    print(f"Aligned Data: {len(df_merged)} bars.")
    print(f"Config: Th={args.threshold:.2f}, Hold={args.hold_bars}m, SL={args.sl_pct:.1%}, TP={args.tp_rr:.1f}x")
    print(f"Trailing Stop: {args.trailing_enabled} ({args.trailing_pct:.1%})")
    print("Trend Filter: EMA 200 enabled")
    
    # 2. Model Inference (Batch)
    print("Running Inference...")
    in_dim = X_valid.shape[1]
    model = MultiHorizonMLP(in_dim=in_dim, out_dim=3)
    state = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    
    probs = []
    batch_size = 4096
    with torch.no_grad():
        for i in range(0, len(X_valid), batch_size):
            xb = torch.FloatTensor(X_valid[i:i+batch_size])
            logits = model(xb)
            p = sigmoid(logits[:, 1].numpy()) # 15m up
            probs.append(p)
    probs = np.concatenate(probs)
    
    df_merged["prob"] = probs
    
    # 3. Simulation Loop
    print("Simulating Trades...")
    realized_pnl = 0.0
    positions = []
    closed_trades = []
    equity_curve = []
    
    last_trade_time = None
    
    # Config dict for trailing
    trading_cfg = {
        "TRAILING_STOP_ENABLED": args.trailing_enabled,
        "TRAILING_STOP_PCT": args.trailing_pct
    }
    
    for i in range(len(df_merged) - 1):
        row = df_merged.iloc[i]
        next_row = df_merged.iloc[i+1] # The bar we trade inside
        
        timestamp = row["timestamp"]
        prob = row["prob"]
        
        # Decide which close to use for mark-to-market.
        close_price = row.get("close_raw", row.get("close", 0.0))
        ema_200 = row.get("ema200", 0.0)

        # Check Exits using NEXT BAR candles (Realistic)
        open_next = next_row["open"]
        high_next = next_row["high"]
        low_next = next_row["low"]
        close_next = next_row.get("close_raw", next_row.get("close", 0.0))
        time_next = next_row["timestamp"]
        
        # Update Equity (Realized + Open PnL of active positions)
        open_pnl = sum([p.qty * (close_price - p.entry_price) for p in positions])
        current_equity = args.start_capital + realized_pnl + open_pnl
        equity_curve.append(current_equity)
        
        active_positions = []
        for p in positions:
            is_closed = False
            
            # Check Max Hold
            elapsed = (time_next - p.entry_time).total_seconds() / 60
            if elapsed >= args.hold_bars:
                p.exit(time_next, close_next, "MaxHold")
                is_closed = True
                
            # Check SL/TP (Intrabar)
            elif p.check_exit(time_next, low_next, high_next, close_next, trading_cfg):
                is_closed = True
            
            if is_closed:
                # Add to Realized PnL immediately
                slip_loss = p.exit_price * (args.slippage_bps / 10000.0)
                net_exit = p.exit_price - slip_loss
                pnl = (net_exit - p.entry_price) * p.qty - (args.fee * 2)
                realized_pnl += pnl
                closed_trades.append(p)
            else:
                active_positions.append(p)
        positions = active_positions
        
        # 2. Check Entry Signal (at Close i) -> Enters at Open i+1
        if len(positions) < args.max_positions:
            # ENTRY CONDITION: Prob > Th AND Price > EMA200
            if prob >= args.threshold and close_price > ema_200:
                # Check Cooldown
                is_cooldown = False
                if last_trade_time:
                    if (timestamp - last_trade_time).total_seconds() / 60 < args.cooldown:
                        is_cooldown = True
                
                if not is_cooldown:
                    # ENTRY
                    base_equity = args.start_capital + realized_pnl
                    risk_amt = base_equity * args.risk_pct
                    
                    entry_price = open_next # Execution Price (realistic)
                    
                    # Slippage penalty on entry
                    slippage = entry_price * (args.slippage_bps / 10000.0)
                    entry_price_slip = entry_price + slippage
                    
                    stop_dist = entry_price_slip * args.sl_pct
                    stop_price = entry_price_slip - stop_dist
                    take_profit_price = entry_price_slip + (stop_dist * args.tp_rr)
                    
                    qty = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                    if qty > 0:
                        pos = Position(time_next, entry_price_slip, qty, stop_price, take_profit_price)
                        positions.append(pos)
                        last_trade_time = timestamp
                        
    # End Loop
    
    # Close remaining
    final_price = df_merged.iloc[-1].get("close_raw", df_merged.iloc[-1].get("close", 0.0))
    final_time = df_merged.iloc[-1]["timestamp"]
    for p in positions:
        p.exit(final_time, final_price, "EndSim")
        
        slip_loss = p.exit_price * (args.slippage_bps / 10000.0)
        net_exit = p.exit_price - slip_loss
        pnl = (net_exit - p.entry_price) * p.qty - (args.fee * 2)
        realized_pnl += pnl
        
        closed_trades.append(p)
        
    # Calculate Stats
    print("\n--- Results ---")
    wins = 0
    trade_dicts = []
    
    for t in closed_trades:
        # PnL logic repeated just for CSV log
        slip_loss = t.exit_price * (args.slippage_bps / 10000.0)
        net_exit = t.exit_price - slip_loss
        net = (net_exit - t.entry_price) * t.qty - (args.fee * 2)
        
        if net > 0: wins += 1
        
        trade_dicts.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "qty": t.qty,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "reason": t.exit_reason,
            "pnl": net
        })
        
    final_equity = args.start_capital + realized_pnl
    win_rate = wins / len(closed_trades) if closed_trades else 0
    
    print(f"Total Trades: {len(closed_trades)}")
    print(f"Final Equity: {final_equity:,.2f} (Start: {args.start_capital})")
    print(f"Total Return: {(final_equity/args.start_capital - 1):.2%}")
    print(f"Win Rate:     {win_rate:.2%}")
    print(f"Total PnL:    {realized_pnl:,.2f}")
    
    # Save
    if trade_dicts:
        df_tr = pd.DataFrame(trade_dicts)
        out_csv = PROJECT_ROOT / "reports" / "backtest_trades.csv"
        df_tr.to_csv(out_csv, index=False)
        print(f"Saved trades to {out_csv}")
        
        # ADDED PRINT OF LAST 10 TRADES
        print("\n--- Last 10 trades ---")
        print(df_tr[["entry_time", "qty", "pnl", "reason"]].tail(10))
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Strategy Equity", linewidth=1.5)
    plt.axhline(y=args.start_capital, color='r', linestyle='--', alpha=0.3, label="Start Capital")
    plt.title(f"Backtest Equity Curve (Return: {(final_equity/args.start_capital - 1):.2%})")
    plt.xlabel("Bars (Minutes)")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    # OUTPUT PATH CHANGED to images/backtesting
    out_png = PROJECT_ROOT / "images" / "backtesting" / "ff_trade_backtest_equity.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")

if __name__ == "__main__":
    main()
