"""
FF Backtest (Direction) – Evaluation auf Test-Set
=================================================

- Lädt: data/processed/X_test_scaled.npy  (already scaled)
- Lädt: data/processed/test.parquet      (targets: target_5m/target_15m/target_30m als Returns)
- True-Labels: (target > 0) => 1 sonst 0
- Model: models/feed_forward/multihorizon_nn.pt (out_dim=3 -> [5m,15m,30m])
- Output:
  - Metriken (Accuracy pro Horizon)
  - Plot: Prob(15m) vs True(15m) für letzte N Punkte
  - Plot: Balkenplot Accuracy pro Horizon
  - Optional: CSV mit probs/preds/truth

Run (am besten aus PROJECT_ROOT):
  python scripts/09_backtesting/ff_backtest.py --last-n 600 --threshold 0.5 --save-csv
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


# -----------------------------
# Robust project root finder
# -----------------------------
def find_project_root(start: Path) -> Path:
    """
    Sucht nach Projekt-Root, indem typische Ordner gefunden werden.
    Fix für Windows-Pfade, wenn man das Script aus Unterordnern ausführt.
    """
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "data" / "processed").exists() and (p / "models").exists() and (p / "scripts").exists():
            return p
    # fallback: 2 levels up (wie üblich scripts/09_backtesting -> scripts -> root)
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


def load_test_data(data_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    x_path = data_path / "X_test_scaled.npy"
    pq_path = data_path / "test.parquet"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing: {x_path}")
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing: {pq_path}")

    X = np.load(x_path).astype(np.float32)
    df = pd.read_parquet(pq_path)

    target_cols = ["target_5m", "target_15m", "target_30m"]
    for c in target_cols:
        if c not in df.columns:
            raise KeyError(f"Missing target column '{c}' in {pq_path}")

    # Align length
    n = min(len(df), X.shape[0])
    if n <= 0:
        raise RuntimeError("No rows after alignment.")
    if n != X.shape[0] or n != len(df):
        print(f"[WARN] Length mismatch -> aligned to n={n} (X={X.shape[0]}, df={len(df)})")

    return X[:n], df.iloc[:n].reset_index(drop=True)


def infer_probs(model: torch.nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    N = X.shape[0]
    probs = np.zeros((N, 3), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float()
            logits = model(xb).cpu().numpy()  # (B,3)
            probs[i:i + batch_size] = sigmoid(logits).astype(np.float32)

    return probs


def accuracy(y_true: np.ndarray, p: np.ndarray, thr: float) -> float:
    y_pred = (p >= thr).astype(int)
    return float((y_pred == y_true).mean())


def add_bar_labels(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.55, help="Sigmoid threshold for UP (default 0.55)")
    parser.add_argument("--last-n", type=int, default=400, help="How many last points for the 15m plot")
    parser.add_argument("--save-csv", action="store_true", help="Save detailed CSV in images/backtesting/")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    data_path = project_root / "data" / "processed"
    model_path = project_root / "models" / "feed_forward" / "multihorizon_nn.pt"
    images_dir = project_root / "images" / "backtesting"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FF BACKTEST (Direction) – Evaluation auf X_test_scaled.npy")
    print(f"PROJECT_ROOT: {project_root}")
    print("=" * 90)

    # Load
    X_test, df_test = load_test_data(data_path)
    x_dim = X_test.shape[1]
    print(f"[DATA] X_test_scaled: {X_test.shape}")
    print(f"[DATA] test.parquet:  {df_test.shape}")

    # True labels
    target_cols = ["target_5m", "target_15m", "target_30m"]
    y_true = np.column_stack([(df_test[c].values > 0).astype(int) for c in target_cols])  # (N,3)

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
    probs = infer_probs(model, X_test, batch_size=4096)  # (N,3)
    p5, p15, p30 = probs[:, 0], probs[:, 1], probs[:, 2]

    # Metrics
    acc5 = accuracy(y_true[:, 0], p5, args.threshold)
    acc15 = accuracy(y_true[:, 1], p15, args.threshold)
    acc30 = accuracy(y_true[:, 2], p30, args.threshold)

    print("\nMETRICS (Direction Accuracy)")
    print(f"  threshold = {args.threshold:.2f}")
    print(f"  5m : {acc5:.4f}")
    print(f"  15m: {acc15:.4f}   <-- (Deployment nutzt i.d.R. 15m)")
    print(f"  30m: {acc30:.4f}")

    # Save CSV (optional)
    if args.save_csv:
        out_csv = images_dir / "ff_backtest_probs.csv"
        out_df = pd.DataFrame(
            {
                "p_5m": p5,
                "p_15m": p15,
                "p_30m": p30,
                "y_5m": y_true[:, 0],
                "y_15m": y_true[:, 1],
                "y_30m": y_true[:, 2],
            }
        )
        out_df.to_csv(out_csv, index=False)
        print(f"[SAVE] CSV: {out_csv}")

    # Plot 1: p15 vs true (last N)
    n = min(args.last_n, len(p15))
    idx0 = len(p15) - n
    xs = np.arange(idx0, idx0 + n)

    plt.figure(figsize=(12, 4))
    plt.plot(xs, p15[idx0:idx0 + n], label="Prob UP (15m)", alpha=0.9)
    plt.plot(xs, y_true[idx0:idx0 + n, 1], label="True Direction (15m)", alpha=0.6)
    plt.axhline(args.threshold, linestyle="--", alpha=0.6, label=f"Threshold {args.threshold:.2f}")
    plt.ylim(-0.05, 1.05)
    plt.title(f"FF Backtest – 15m: Predicted Probability vs True Direction (last {n})")
    plt.xlabel(f"Test index (last {n})")

    plt.ylabel("Probability / Label")
    plt.grid(True, alpha=0.2)
    plt.legend()
    out1 = images_dir / "ff_backtest_15m_prob_vs_true.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close()
    print(f"[PLOT] {out1}")

    # Plot 2: bar accuracies
    plt.figure(figsize=(7, 4))
    horizons = ["5m", "15m", "30m"]
    vals = [acc5, acc15, acc30]
    bars = plt.bar(horizons, vals)
    add_bar_labels(plt.gca(), bars)
    plt.ylim(0.0, 1.0)
    plt.title("FF Backtest – Accuracy per Horizon")
    plt.ylabel("Accuracy")
    plt.grid(axis="y", alpha=0.2)
    out2 = images_dir / "ff_backtest_accuracy_bar.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()
    print(f"[PLOT] {out2}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
