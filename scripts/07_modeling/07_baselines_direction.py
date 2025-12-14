"""
Baselines (Direction) for Step 07 Modeling
=========================================
Passt zu euren 07_feed_forward.py und 07_lstm.py:
- y = (target_return > 0)  -> 0/1 Richtung
- Metrik: Accuracy pro Horizont + Overall

Baselines:
1) Market: always UP
2) Dummy: most_frequent (Prof-"Dummy")
3) Logistic Regression: linear baseline

Optional: ALIGN_TO_LSTM=True droppt die ersten (SEQ_LEN-1) Samples in Val/Test,
damit es exakt zu LSTM-Sequenzen (seq_len=20) passt.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

TARGET_COLS = ["target_5m", "target_15m", "target_30m"]

ALIGN_TO_LSTM = True
SEQ_LEN = 20  # wie in eurem LSTM


# Pfade analog zu euren 07_* Dateien
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "processed"
IMAGES_PATH  = PROJECT_ROOT / "images" / "modeling" / "baselines"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def load_X(split: str) -> np.ndarray:
    # du sagst: ihr nutzt X_*_scaled
    return np.load(DATA_PATH / f"X_{split}_scaled.npy").astype(np.float32)


def load_y_dir(split: str) -> np.ndarray:
    df = pd.read_parquet(DATA_PATH / f"{split}.parquet")
    y = (df[TARGET_COLS].values > 0).astype(np.int32)
    return y


def maybe_align_to_lstm(X, y):
    if not ALIGN_TO_LSTM:
        return X, y
    cut = SEQ_LEN
    return X[cut:], y[cut:]


def acc_per_horizon(pred: np.ndarray, y: np.ndarray):
    per = (pred == y).mean(axis=0)
    overall = (pred == y).mean()
    return per, overall


def main():
    print(f"DATA_PATH: {DATA_PATH}")

    X_train = np.nan_to_num(load_X("train"))
    X_val   = np.nan_to_num(load_X("val"))
    X_test  = np.nan_to_num(load_X("test"))

    y_train = load_y_dir("train")
    y_val   = load_y_dir("val")
    y_test  = load_y_dir("test")

    # Alignment nur für Val/Test relevant (Train nutzt LSTM auch sequenziell, aber Baseline soll nur Vergleich sein)
    X_val_a, y_val_a   = maybe_align_to_lstm(X_val, y_val)
    X_test_a, y_test_a = maybe_align_to_lstm(X_test, y_test)

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val_a.shape}, y={y_val_a.shape} (aligned={ALIGN_TO_LSTM})")
    print(f"Test:  X={X_test_a.shape}, y={y_test_a.shape} (aligned={ALIGN_TO_LSTM})")

    # 1) Market baseline: always UP
    pred_up = np.ones_like(y_test_a)
    acc_up, acc_up_overall = acc_per_horizon(pred_up, y_test_a)

    # 2) Dummy most_frequent (Prof-"Dummy")
    dummy = OneVsRestClassifier(DummyClassifier(strategy="most_frequent", random_state=42))
    dummy.fit(X_train, y_train)
    pred_dummy = dummy.predict(X_test_a)
    acc_dummy, acc_dummy_overall = acc_per_horizon(pred_dummy, y_test_a)

    # 3) Logistic Regression baseline (linear)
    logreg = LogisticRegression(solver="saga", max_iter=300, n_jobs=-1)
    clf = OneVsRestClassifier(logreg, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred_lr = clf.predict(X_test_a)
    acc_lr, acc_lr_overall = acc_per_horizon(pred_lr, y_test_a)

    # p(up) fürs Verständnis
    p_up = y_test_a.mean(axis=0)

    print("\n" + "="*70)
    print("BASELINE RESULTS (TEST) – Direction Accuracy")
    print("="*70)
    for i, col in enumerate(TARGET_COLS):
        print(f"{col:12} | p_up={p_up[i]:.3f} | always_up={acc_up[i]:.4f} | dummy_mf={acc_dummy[i]:.4f} | logreg={acc_lr[i]:.4f}")
    print("-"*70)
    print(f"OVERALL      | always_up={acc_up_overall:.4f} | dummy_mf={acc_dummy_overall:.4f} | logreg={acc_lr_overall:.4f}")
    print("="*70)

    # Plot
    horizons = ["5m", "15m", "30m"]
    x = np.arange(len(horizons))
    w = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, acc_up,    width=w, label="Market: Always UP")
    plt.bar(x,     acc_dummy, width=w, label="Dummy: most_frequent")
    plt.bar(x + w, acc_lr,    width=w, label="LogReg (linear)")

    plt.xticks(x, horizons)
    plt.ylabel("Accuracy")
    plt.title(f"Baselines vs Market (Test) | aligned_to_lstm={ALIGN_TO_LSTM}")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    out = IMAGES_PATH / "07_baselines_direction_accuracy.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"\n✅ Plot saved: {out}")


if __name__ == "__main__":
    main()
