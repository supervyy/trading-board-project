import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------------------------------------
# Pfade / Setup
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "processed"
MODELS_PATH  = PROJECT_ROOT / "models" / "lstm"
IMAGES_PATH  = PROJECT_ROOT / "images" / "modeling" / "lstm"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# gleiche Targets wie im Feedforward
TARGET_COLS = ["target_5m", "target_15m", "target_30m"]


# -------------------------------------------------
# Dataset & Sequenz-Helfer
# -------------------------------------------------
class SequenceDataset(Dataset):
    """
    Dataset für Sequenzen:
    X_seq: [num_seq, seq_len, num_features]
    y_seq: [num_seq, num_targets]
    """
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        assert X_seq.shape[0] == y_seq.shape[0]
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y_seq).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 20,
):
    """
    Sliding-Window-Sequenzen über die Zeit:

    Input:
        X: [N, num_features]
        y: [N, num_targets]
    Output:
        X_seq: [N - seq_len, seq_len, num_features]
        y_seq: [N - seq_len, num_targets]
            (Ziel ist immer der nächste Zeitschritt nach dem Fenster)
    """
    X_seq, y_seq = [], []
    N = X.shape[0]
    for i in range(N - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


# -------------------------------------------------
# LSTM-Modell (Klassifikation Richtung)
# -------------------------------------------------
class LSTMClassifier(nn.Module):
    """
    Einfaches LSTM für Richtungs-Klassifikation
    (3 Outputs: 5m / 15m / 30m).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        output_size: int = 3,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, (h_n, c_n) = self.lstm(x)
        # h_n: [num_layers * num_directions, batch_size, hidden_size]
        last_layer_h = h_n[-self.num_directions:, :, :]   # [num_directions, B, H]
        last_layer_h = last_layer_h.transpose(0, 1).reshape(x.size(0), -1)
        logits = self.fc(last_layer_h)
        return logits


# -------------------------------------------------
# Daten laden (X_train.npy + train.parquet)
# -------------------------------------------------
def load_data_for_lstm(sequence_length: int = 20):
    print(f"📂 Lade Feature-Matrizen aus {DATA_PATH} ...")
    X_train = np.load(DATA_PATH / "X_train_scaled.npy")
    X_val   = np.load(DATA_PATH / "X_val_scaled.npy")
    X_test  = np.load(DATA_PATH / "X_test_scaled.npy")

    print("📂 Lade train/val/test Parquet für Targets ...")
    train_df = pd.read_parquet(DATA_PATH / "train.parquet")
    val_df   = pd.read_parquet(DATA_PATH / "val.parquet")
    test_df  = pd.read_parquet(DATA_PATH / "test.parquet")

    # Regression-Targets
    y_train_reg = train_df[TARGET_COLS].values.astype(np.float32)
    y_val_reg   = val_df[TARGET_COLS].values.astype(np.float32)
    y_test_reg  = test_df[TARGET_COLS].values.astype(np.float32)

    # Klassifikation: Richtung (0/1)
    y_train = (y_train_reg > 0).astype(np.float32)
    y_val   = (y_val_reg > 0).astype(np.float32)
    y_test  = (y_test_reg > 0).astype(np.float32)

    print(f"   Train X: {X_train.shape}, y: {y_train.shape}")
    print(f"   Val   X: {X_val.shape},   y: {y_val.shape}")
    print(f"   Test  X: {X_test.shape},  y: {y_test.shape}")

    # Sequenzen bauen
    print(f"🧩 Baue Sequenzen mit Länge {sequence_length} ...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   sequence_length)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  sequence_length)

    print("   Sequenz-Shapes:")
    print(f"     X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    print(f"     X_val_seq:   {X_val_seq.shape},   y_val_seq:   {y_val_seq.shape}")
    print(f"     X_test_seq:  {X_test_seq.shape},  y_test_seq:  {y_test_seq.shape}")

    train_ds = SequenceDataset(X_train_seq, y_train_seq)
    val_ds   = SequenceDataset(X_val_seq,   y_val_seq)
    test_ds  = SequenceDataset(X_test_seq,  y_test_seq)

    # input_size = Anzahl Features
    input_size = X_train_seq.shape[2]
    return train_ds, val_ds, test_ds, input_size


# -------------------------------------------------
# Training / Evaluation
# -------------------------------------------------
def train_lstm(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    input_size: int,
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=1,
        output_size=len(TARGET_COLS),
        bidirectional=False,
        dropout=0.1,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    no_improve = 0
    train_losses, val_losses = [], []

    print(f"\n🚀 Training LSTM (direction 5m/15m/30m) auf {DEVICE} ...")
    for epoch in range(1, epochs + 1):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ----- Validation -----
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += loss.item() * xb.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch:03d} | Train: {epoch_train_loss:.5f} | Val: {epoch_val_loss:.5f}")

        # Early Stopping
        if epoch_val_loss < best_val_loss - 1e-5:
            best_val_loss = epoch_val_loss
            no_improve = 0
            torch.save(model.state_dict(), MODELS_PATH / "lstm_direction.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

    # bestes Modell laden
    model.load_state_dict(
        torch.load(MODELS_PATH / "lstm_direction.pt", map_location=DEVICE)
    )

    # -------------------------------------------------
    # 1) Loss-Plot
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("LSTM – Train vs Val Loss (direction targets)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    loss_path = IMAGES_PATH / "07_lstm_loss.png"
    plt.savefig(loss_path)
    plt.close()
    print("📉 LSTM Loss-Kurve gespeichert unter:", loss_path)

    # -------------------------------------------------
    # 2) Evaluation auf Test-Sequenzen
    # -------------------------------------------------
    def eval_loader(loader, split_name: str):
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        logits = np.vstack(all_logits)
        targets = np.vstack(all_targets)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(np.float32)

        acc_per_horizon = (preds == targets).mean(axis=0)
        acc_overall = (preds == targets).mean()

        print(f"\n🔍 {split_name} Accuracy (LSTM):")
        for i, col in enumerate(TARGET_COLS):
            print(f"   {col:12}: {acc_per_horizon[i]:.4f}")
        print(f"   Overall     : {acc_overall:.4f}")

        return acc_per_horizon, acc_overall, probs, targets

    val_accs, val_overall, _, _ = eval_loader(val_loader, "Validation")
    test_accs, test_overall, test_probs, test_targets = eval_loader(test_loader, "Test")

    # -------------------------------------------------
    # 3) Test-Accuracy-Barplot
    # -------------------------------------------------
    horizons = ["5m", "15m", "30m"]
    plt.figure(figsize=(8, 5))
    plt.bar(horizons, test_accs)
    plt.ylim(0.4, 0.7)
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Horizon – LSTM")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    acc_path = IMAGES_PATH / "07_lstm_test_accuracy.png"
    plt.savefig(acc_path)
    plt.close()
    print("📊 LSTM Test-Accuracy-Barplot gespeichert unter:", acc_path)

    # -------------------------------------------------
    # 4) Actual vs Predicted (wie beim Feedforward)
    # -------------------------------------------------
    n_plot = min(200, test_probs.shape[0])
    x = np.arange(n_plot)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(TARGET_COLS):
        ax = axes[i]
        actual = test_targets[:n_plot, i]
        prob   = test_probs[:n_plot, i]

        ax.step(
            x,
            actual,
            where="post",
            label="Actual Direction (0/1)",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.plot(
            x,
            prob,
            label="Predicted Prob (up)",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_title(f"{col} – erste {n_plot} Test-Sequenzen")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value / Probability")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.text(
            0.01,
            0.95,
            f"Test-Acc: {test_accs[i]:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.8, linewidth=0
            ),
        )
        ax.legend()

    # 4. Subplot: Validation-Loss
    ax4 = axes[3]
    ax4.plot(val_losses, label="Val Loss", linewidth=1.2)
    ax4.set_title("Validation Loss pro Epoch (LSTM)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("BCE Loss")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend()

    plt.suptitle("LSTM – Actual vs Predicted (Test) & Validation Loss", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    curves_path = IMAGES_PATH / "07_lstm_actual_vs_predicted_test.png"
    plt.savefig(curves_path)
    plt.close()
    print("📈 LSTM Actual-vs-Predicted-Plots gespeichert unter:", curves_path)

    return model


def main():
    sequence_length = 20  # wie in deiner LSTM-Beschreibung
    train_ds, val_ds, test_ds, input_size = load_data_for_lstm(
        sequence_length=sequence_length
    )
    train_lstm(train_ds, val_ds, test_ds, input_size=input_size)


if __name__ == "__main__":
    main()
