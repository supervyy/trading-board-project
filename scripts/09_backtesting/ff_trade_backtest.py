"""
FF Trade Backtest (Direction) – Simple Strategy Simulation
==========================================================

Trading-Regel (Deployment-like):
- Nutzt Prob(UP_15m)
- Wenn Prob >= threshold => Enter LONG
- Halten hold_bars (default 15 Minuten)
- Keine Overlaps: nach Entry springt der Index um hold_bars

PnL:
- Nutzt target_15m (Return) aus test.parquet
- Optional nutzt close (falls vorhanden) nur für Buy&Hold Kurve
- Startkapital default: 100000 (wie viele README-Beispiele)
- qty default: 1 (wie dein Deployment typischerweise)

Run:
  python scripts/09_backtesting/ff_trade_backtest.py --threshold 0.6 --hold-bars 15 --start-capital 100000 --qty 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def find_project_root(start: Path) -> Path:
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "data" / "processed").exists() and (p / "models").exists() and (p / "scripts").exists():
            return p
    return start.parents[1]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def import_multihorizon_mlp(project_root: Path) -> type:
    model_script_path = project_root / "scripts" / "07_modeling" / "07_feed_forward.py"
    if not model_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {model_script_path}")

    spec = importlib.util.spec_from_file_location("feed_forward_module", model_script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["feed_forward_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, "MultiHorizonMLP"):
        raise AttributeError("MultiHorizonMLP not found in 07_feed_forward.py")

    return module.MultiHorizonMLP


def load_test_data(data_path: Path, hold_bars: int) -> tuple[np.ndarray, pd.DataFrame]:
    x_path = data_path / "X_test_scaled.npy"
    pq_path = data_path / "test.parquet"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing: {pq_path}")

    X = np.load(x_path).astype(np.float32)
    df = pd.read_parquet(pq_path)

    if "target_15m" not in df.columns:
        raise KeyError(f"Missing 'target_15m' in {pq_path}")

    n = min(len(df), X.shape[0])
    if n <= hold_bars + 1:
        raise RuntimeError("Not enough rows for trade simulation after alignment.")
    if n != X.shape[0] or n != len(df):
        print(f"[WARN] Length mismatch -> aligned to n={n} (X={X.shape[0]}, df={len(df)})")

    df = df.iloc[:n]
    # if timestamp is stored in index, keep it as column
    if "timestamp" not in df.columns and df.index.name == "timestamp":
        df = df.reset_index()          # creates 'timestamp' column
    else:
        df = df.reset_index(drop=True) # keep old behavior

    return X[:n], df



def infer_probs(model: torch.nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    N = X.shape[0]
    probs = np.zeros((N, 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float()
            logits = model(xb).cpu().numpy()
            probs[i:i + batch_size] = sigmoid(logits).astype(np.float32)
    return probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.55, help="Enter long if Prob(UP_15m) >= threshold")
    parser.add_argument("--hold-bars", type=int, default=15, help="Hold duration in bars (1 bar = 1 minute)")
    parser.add_argument("--start-capital", type=float, default=100000.0, help="Start capital for equity curve")
    parser.add_argument("--qty", type=int, default=1, help="Shares per trade (like deployment; default 1)")
    parser.add_argument("--fee", type=float, default=0.0, help="Flat fee per trade (entry+exit each)")
    parser.add_argument("--save-trades", action="store_true", help="Save trades CSV to images/backtesting/")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    data_path = project_root / "data" / "processed"
    model_path = project_root / "models" / "feed_forward" / "multihorizon_nn.pt"
    images_dir = project_root / "images" / "backtesting"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FF TRADE BACKTEST (Direction) – Deployment-like Simulation on Test Set")
    print(f"PROJECT_ROOT: {project_root}")
    print("=" * 90)

    # Load
    X_test, df_test = load_test_data(data_path, args.hold_bars)
    x_dim = X_test.shape[1]
    print(f"[DATA] X_test_scaled: {X_test.shape}")
    print(f"[DATA] test.parquet:  {df_test.shape}")

    # Model
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    MultiHorizonMLP = import_multihorizon_mlp(project_root)
    model = MultiHorizonMLP(in_dim=x_dim, out_dim=3)
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    print(f"[MODEL] Loaded: {model_path.name} | in_dim={x_dim} | out_dim=3")

    # Inference
    probs = infer_probs(model, X_test, batch_size=4096)
    p15 = probs[:, 1]  # Prob(UP_15m)
    r15 = df_test["target_15m"].astype(float).values  # realized return if holding 15m

    # Buy&Hold baseline (optional if close exists)
    close_col = None
    for c in ["close", "Close"]:
        if c in df_test.columns:
            close_col = c
            break
    # Optional timestamp column for nicer trade logs (no impact on strategy)
    time_col = None
    for c in ["timestamp", "datetime", "date", "time", "Datetime", "Date", "Timestamp", "Time"]:
        if c in df_test.columns:
            time_col = c
            break

    # Simulation: cash-based with qty
    start_cap = float(args.start_capital)
    cash = start_cap
    equity_curve = np.full(len(df_test), np.nan, dtype=np.float64)  # mark equity per bar
    trades = []

    i = 0
    N = len(df_test)

    while i < N - args.hold_bars:
        equity_curve[i] = cash

        if p15[i] >= args.threshold:
            # Entry
            # If we have close, use it as entry price; otherwise simulate via returns on cash (fallback)
            if close_col is not None:
                entry_price = float(df_test.loc[i, close_col])
                cost = args.qty * entry_price + args.fee
                if cost > cash:
                    # not enough cash -> skip signal
                    i += 1
                    continue

                cash -= cost

                # Exit after hold
                trade_ret = float(r15[i])  # return from i -> i+hold
                exit_price = entry_price * (1.0 + trade_ret)
                proceeds = args.qty * exit_price - args.fee
                cash += proceeds

                pnl = proceeds - (args.qty * entry_price) - args.fee  # approx (double fee already included)
            else:
                # Fallback: apply return to a "position fraction" = all-in of current cash (not qty based)
                trade_ret = float(r15[i])
                pnl = cash * trade_ret
                cash = cash * (1.0 + trade_ret) - 2.0 * args.fee
            def get_time(pos: int):
                if time_col is None:
                 return None
                v = df_test.iloc[pos][time_col]  # iloc = positionsbasiert (robust)
                if time_col == "timestamp":      # ms -> datetime
                    return pd.to_datetime(v, unit="ms", utc=True)
                return v
            entry_time = get_time(i)
            exit_time  = get_time(i + args.hold_bars)
            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "qty": int(args.qty),
                    "reason": "MaxHold",
                    "i_entry": i,
                    "i_exit": i + args.hold_bars,
                    "prob_15m": float(p15[i]),
                    "ret_15m": trade_ret,
                    "pnl": float(pnl),
                    "cash_after": float(cash),
                    "win": int(trade_ret > 0.0),
                }
            )

            # Fill equity between entry and exit as flat (simple step curve)
            j_end = min(N, i + args.hold_bars)
            equity_curve[i:j_end] = cash
            i += args.hold_bars
        else:
            i += 1

    # Fill remaining NaNs with last known cash
    last = cash
    for k in range(len(equity_curve)):
        if np.isnan(equity_curve[k]):
            equity_curve[k] = last
        else:
            last = equity_curve[k]

    trades_df = pd.DataFrame(trades)
    total_return = (cash / start_cap) - 1.0

    # Stats
    print("\n" + "-" * 90)
    print(f"Config: threshold={args.threshold:.2f} | hold_bars={args.hold_bars} | start_capital={start_cap:.2f} | qty={args.qty}")
    print("-" * 90)
    print(f"Trades:       {len(trades_df)}")
    print(f"Final equity: {cash:,.2f}")
    print(f"Total return: {total_return:.2%}")

    if len(trades_df) > 0:
        win_rate = float(trades_df["win"].mean())
        avg_ret = float(trades_df["ret_15m"].mean())
        med_ret = float(trades_df["ret_15m"].median())
        avg_pnl = float(trades_df["pnl"].mean())
        print(f"Win rate:     {win_rate:.2%}")
        print(f"Avg ret:      {avg_ret:.4%}")
        print(f"Med ret:      {med_ret:.4%}")
        print(f"Avg PnL:      {avg_pnl:,.2f}")
    cols = [c for c in ["entry_time", "qty", "pnl", "reason"] if c in trades_df.columns]
    if cols:
        print("\n--- Last 10 trades ---")
        print(trades_df[cols].tail(10).to_string(index=True))

    # Plot equity curve (+ buy&hold)
    plt.figure(figsize=(12, 5))
    plt.plot(equity_curve, label="Strategy Equity", alpha=0.9)

    if close_col is not None:
        close = df_test[close_col].astype(float).values
        bh = start_cap * (close / close[0])
        plt.plot(bh, label="Buy & Hold (same start capital)", alpha=0.7)

    plt.title("FF Trade Backtest (Direction) – Equity Curve")
    plt.xlabel("Test index")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True, alpha=0.2)

    out = images_dir / "ff_trade_backtest_equity.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\n[PLOT] {out}")

    # Save trades
    if args.save_trades and len(trades_df) > 0:
        trades_out = images_dir / "ff_trade_backtest_trades.csv"
        trades_df.to_csv(trades_out, index=False)
        print(f"[SAVE] Trades CSV: {trades_out}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
