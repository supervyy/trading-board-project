"""
Step 07 – ONE Comparison Plot: Dummy Baseline vs FeedForward vs LSTM (Direction)
===============================================================================
Ziel:
- EIN Bild erzeugen, das den Vergleich zeigt: Dummy vs FeedForward vs LSTM
- Metrik: Accuracy pro Horizont (5m/15m/30m) + Overall in Titel
- Fairer Vergleich: alles auf die LSTM-Testlänge ausrichten (drop first SEQ_LEN)

Voraussetzungen:
- data/processed/X_*_scaled.npy existiert
- data/processed/{train,val,test}.parquet existiert
- FeedForward Modell gespeichert unter: models/feed_forward/multihorizon_nn.pt
- LSTM Modell gespeichert unter: models/lstm/lstm_direction.pt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier

import torch
import importlib.util


# -----------------------------
# Config
# -----------------------------
TARGET_COLS = ["target_5m", "target_15m", "target_30m"]
HORIZON_NAMES = ["5m", "15m", "30m"]

SEQ_LEN = 20  # muss zu 07_lstm.py passen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"

FF_MODEL_PATH = PROJECT_ROOT / "models" / "feed_forward" / "multihorizon_nn.pt"
LSTM_MODEL_PATH = PROJECT_ROOT / "models" / "lstm" / "lstm_direction.pt"

OUT_DIR = PROJECT_ROOT / "images" / "modeling" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def import_attr(py_path: Path, module_name: str, attr_name: str):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, attr_name)


def load_X(split: str) -> np.ndarray:
    path = DATA_PATH / f"X_{split}_scaled.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return np.nan_to_num(np.load(path)).astype(np.float32)


def load_y_dir(split: str) -> np.ndarray:
    path = DATA_PATH / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_parquet(path)
    y = (df[TARGET_COLS].values > 0).astype(np.int32)
    return y


def align_to_lstm_length(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # LSTM create_sequences nutzt y[i + SEQ_LEN] als Label -> Länge N-SEQ_LEN
    return X[SEQ_LEN:], y[SEQ_LEN:]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def acc_per_horizon(pred: np.ndarray, y: np.ndarray):
    per = (pred == y).mean(axis=0)
    overall = (pred == y).mean()
    return per, overall


def predict_ff(ff_model, X_np: np.ndarray) -> np.ndarray:
    X_t = torch.from_numpy(X_np).float().to(DEVICE)
    ff_model.eval()
    with torch.no_grad():
        logits = ff_model(X_t).cpu().numpy()
    return (sigmoid(logits) > 0.5).astype(np.int32)


def predict_lstm(lstm_model, X_seq_np: np.ndarray, batch_size: int = 512) -> np.ndarray:
    lstm_model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_seq_np), batch_size):
            xb = torch.from_numpy(X_seq_np[i:i + batch_size]).float().to(DEVICE)
            logits = lstm_model(xb).cpu().numpy()
            out.append((sigmoid(logits) > 0.5).astype(np.int32))
    return np.vstack(out)


def label_bars(bar_container, dy: float = 0.002):
    for b in bar_container:
        h = float(b.get_height())
        plt.text(
            b.get_x() + b.get_width() / 2,
            h + dy,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_PATH:    {DATA_PATH}")
    print(f"DEVICE:       {DEVICE}")

    # Load data
    X_train = load_X("train")
    X_test = load_X("test")
    y_train = load_y_dir("train")
    y_test = load_y_dir("test")

    # Align TEST for fair comparison vs LSTM
    X_test_a, y_test_a = align_to_lstm_length(X_test, y_test)

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test : X={X_test.shape}, y={y_test.shape}")
    print(f"Test aligned (drop first {SEQ_LEN}): X={X_test_a.shape}, y={y_test_a.shape}")

    # -----------------------------
    # 1) Dummy baseline (most_frequent)
    # -----------------------------
    dummy = OneVsRestClassifier(DummyClassifier(strategy="most_frequent", random_state=42))
    dummy.fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test_a)
    acc_dummy, acc_dummy_overall = acc_per_horizon(pred_dummy, y_test_a)

    # -----------------------------
    # 2) FeedForward (load + predict)
    # -----------------------------
    if not FF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing FF model: {FF_MODEL_PATH}\nRun: python scripts/07_modeling/07_feed_forward.py")

    ff_script = Path(__file__).resolve().parent / "07_feed_forward.py"
    MultiHorizonMLP = import_attr(ff_script, "ff_module_compare", "MultiHorizonMLP")

    ff_model = MultiHorizonMLP(in_dim=X_train.shape[1], out_dim=len(TARGET_COLS)).to(DEVICE)
    ff_model.load_state_dict(torch.load(FF_MODEL_PATH, map_location=DEVICE))
    pred_ff = predict_ff(ff_model, X_test_a)
    acc_ff, acc_ff_overall = acc_per_horizon(pred_ff, y_test_a)

    # -----------------------------
    # 3) LSTM (load + sequences + predict)
    # -----------------------------
    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing LSTM model: {LSTM_MODEL_PATH}\nRun: python scripts/07_modeling/07_lstm.py")

    lstm_script = Path(__file__).resolve().parent / "07_lstm.py"
    LSTMClassifier = import_attr(lstm_script, "lstm_module_compare", "LSTMClassifier")
    create_sequences = import_attr(lstm_script, "lstm_module_compare2", "create_sequences")

    # sequences like training
    # NOTE: y muss float32 sein (wie in deinem LSTM-Training)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test.astype(np.float32), sequence_length=SEQ_LEN)

    # instantiate EXACTLY like in 07_lstm.py
    lstm_model = LSTMClassifier(
        input_size=X_train.shape[1],
        hidden_size=128,
        num_layers=1,
        output_size=len(TARGET_COLS),
        bidirectional=False,
        dropout=0.1,
    ).to(DEVICE)

    lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=DEVICE))
    pred_lstm = predict_lstm(lstm_model, X_test_seq)
    acc_lstm, acc_lstm_overall = acc_per_horizon(pred_lstm, y_test_seq.astype(np.int32))

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n" + "=" * 72)
    print("TEST ACCURACY – Dummy vs FeedForward vs LSTM (aligned to LSTM)")
    print("=" * 72)
    for i, col in enumerate(TARGET_COLS):
        print(
            f"{col:10} | Dummy={acc_dummy[i]:.4f} | FF={acc_ff[i]:.4f} | LSTM={acc_lstm[i]:.4f}"
        )
    print("-" * 72)
    print(
        f"OVERALL   | Dummy={acc_dummy_overall:.4f} | FF={acc_ff_overall:.4f} | LSTM={acc_lstm_overall:.4f}"
    )
    print("=" * 72)

    # -----------------------------
    # ONE plot (with values above bars)
    # -----------------------------
    x = np.arange(len(HORIZON_NAMES))
    w = 0.25

    plt.figure(figsize=(11, 5))
    b_dummy = plt.bar(x - w, acc_dummy, width=w, label="Dummy (most_frequent)")
    b_ff    = plt.bar(x,     acc_ff,    width=w, label="FeedForward")
    b_lstm  = plt.bar(x + w, acc_lstm,  width=w, label="LSTM")

    label_bars(b_dummy)
    label_bars(b_ff)
    label_bars(b_lstm)

    plt.xticks(x, HORIZON_NAMES)
    plt.ylabel("Accuracy")
    plt.ylim(0.45, 0.60)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.title(
        f"Step 07 – Dummy vs Models (TEST, aligned seq_len={SEQ_LEN})\n"
        f"Overall: Dummy={acc_dummy_overall:.3f} | FF={acc_ff_overall:.3f} | LSTM={acc_lstm_overall:.3f}"
    )
    plt.legend()
    plt.tight_layout()

    out = OUT_DIR / "07_dummy_vs_models_test.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"\n✅ Saved plot: {out}")


if __name__ == "__main__":
    main()
