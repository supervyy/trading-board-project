import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import os

# -------------------------------------------------
# Pfade / Setup
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH    = PROJECT_ROOT / "data" / "processed"
MODELS_PATH  = PROJECT_ROOT / "models" / "feed_forward"
IMAGES_PATH  = PROJECT_ROOT / "images" / "modeling" / "feed_forward"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

TARGET_COLS = ["target_5m", "target_15m", "target_30m"]


# -------------------------------------------------
# Daten laden
# -------------------------------------------------
def load_data():
    print(f"📂 Loading X matrices from {DATA_PATH} ...")
    X_train = np.load(DATA_PATH / "X_train_scaled.npy")
    X_val   = np.load(DATA_PATH / "X_val_scaled.npy")
    X_test  = np.load(DATA_PATH / "X_test_scaled.npy")

    print("📂 Loading parquet splits for targets ...")
    train_df = pd.read_parquet(DATA_PATH / "train.parquet")
    val_df   = pd.read_parquet(DATA_PATH / "val.parquet")
    test_df  = pd.read_parquet(DATA_PATH / "test.parquet")

    # Regression-Targets (Returns)
    y_train_reg = train_df[TARGET_COLS].values.astype(np.float32)
    y_val_reg   = val_df[TARGET_COLS].values.astype(np.float32)
    y_test_reg  = test_df[TARGET_COLS].values.astype(np.float32)

    # Klassifikation: Richtung (0/1)
    y_train = (y_train_reg > 0).astype(np.float32)
    y_val   = (y_val_reg > 0).astype(np.float32)
    y_test  = (y_test_reg > 0).astype(np.float32)

    # Konsistenz-Check
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0]   == y_val.shape[0]
    assert X_test.shape[0]  == y_test.shape[0]

    X_train_t = torch.from_numpy(X_train).float()
    X_val_t   = torch.from_numpy(X_val).float()
    X_test_t  = torch.from_numpy(X_test).float()

    y_train_t = torch.from_numpy(y_train).float()
    y_val_t   = torch.from_numpy(y_val).float()
    y_test_t  = torch.from_numpy(y_test).float()

    print(f"✅ Shapes -> X_train: {X_train_t.shape}, y_train: {y_train_t.shape}")
    print(f"             X_val:   {X_val_t.shape},   y_val:   {y_val_t.shape}")
    print(f"             X_test:  {X_test_t.shape},  y_test:  {y_test_t.shape}")

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t


# -------------------------------------------------
# Modell
# -------------------------------------------------
class MultiHorizonMLP(nn.Module):
    """
    3 Outputs (Klassifikation):
    - Output[:,0] -> Richtung target_5m
    - Output[:,1] -> Richtung target_15m
    - Output[:,2] -> Richtung target_30m

    Architektur (angelehnt an das andere Projekt):
    in_dim -> 1024 -> 1024 -> 512 -> 512 -> 256 -> 3
    """
    def __init__(self, in_dim: int, out_dim: int = 3, dropout_p: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            # Hidden 1: 1024
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # Hidden 2: 1024
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # Hidden 3: 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # Hidden 4: 512
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # Hidden 5: 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p),

            # Output: 3 Logits (für BCEWithLogitsLoss)
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# -------------------------------------------------
# Training + Evaluation
# -------------------------------------------------
def train_model(
    X_train, y_train, X_val, y_val, X_test, y_test,
    batch_size=2048, epochs=50, lr=1e-3, weight_decay=1e-4, patience=5
):
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    test_ds  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_dim  = X_train.shape[1]
    out_dim = y_train.shape[1]   # 3 Targets (5m, 15m, 30m)

    model = MultiHorizonMLP(in_dim, out_dim=out_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Multi-Label-Binär
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    no_improve = 0
    train_losses, val_losses = [], []

    print("\n🚀 Training Multi-Horizon Feed-Forward (direction 5m/15m/30m)...")
    for epoch in range(1, epochs + 1):
        # --- Train ---
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

        # --- Validation ---
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
            torch.save(model.state_dict(), MODELS_PATH / "multihorizon_nn.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch}")
                break

    # Beste Version laden
    model.load_state_dict(torch.load(MODELS_PATH / "multihorizon_nn.pt", map_location=DEVICE))

    # -------------------------------------------------
    # 1) Loss-Plot (hast du schon)
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Multi-Horizon MLP – Train vs Val Loss")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    out_path = IMAGES_PATH / "06_multihorizon_mlp_loss.png"
    plt.savefig(out_path)
    plt.close()
    print("📉 Loss-Kurve gespeichert unter:", out_path)

    # -------------------------------------------------
    # 2) Evaluation: Accuracy pro Horizont
    # -------------------------------------------------
    def eval_loader(loader, split_name):
        model.eval()
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                all_logits.append(logits.cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        logits = np.vstack(all_logits)
        targets = np.vstack(all_targets)

        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        preds = (probs > 0.5).astype(np.float32)

        acc_per_horizon = (preds == targets).mean(axis=0)
        acc_overall = (preds == targets).mean()

        print(f"\n🔍 {split_name} Accuracy:")
        for i, col in enumerate(TARGET_COLS):
            print(f"   {col:12}: {acc_per_horizon[i]:.4f}")
        print(f"   Overall     : {acc_overall:.4f}")

        return acc_per_horizon, acc_overall, probs, targets

    # Val/Test Accuracy berechnen
    val_accs, val_overall, _, _ = eval_loader(val_loader, "Validation")
    test_accs, test_overall, test_probs, test_targets = eval_loader(test_loader, "Test")

    # -------------------------------------------------
    # 3) Test-Accuracy-Barplot (hast du schon)
    # -------------------------------------------------
    horizons = ["5m", "15m", "30m"]
    plt.figure(figsize=(8, 5))
    plt.bar(horizons, test_accs)
    plt.ylim(0.4, 0.7)
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy per Horizon – Feedforward NN")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    out_path2 = IMAGES_PATH / "06_multihorizon_mlp_test_accuracy.png"
    plt.savefig(out_path2)
    plt.close()
    print("📊 Test-Accuracy-Barplot gespeichert unter:", out_path2)

    # -------------------------------------------------
    # 4) NEU: Linienplots – Actual Direction vs Predicted Probability
    # -------------------------------------------------
    # Wir nehmen die ersten n_plot Test-Samples für anschauliche Plots
        # -------------------------------------------------
    # 4) Schöne Linienplots – Actual Direction vs Predicted Probability
    # -------------------------------------------------
    # Wir nehmen die ersten n_plot Test-Samples für anschauliche Plots
    n_plot = min(200, test_probs.shape[0])   # statt 500 -> 200, übersichtlicher
    x = np.arange(n_plot)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(TARGET_COLS):
        ax = axes[i]
        actual = test_targets[:n_plot, i]
        prob   = test_probs[:n_plot, i]

        # Optional: gleitender Mittelwert der tatsächlichen Richtung (macht das Muster „weicher“)
        # window = 20
        # rolling_actual = pd.Series(actual).rolling(window, center=True).mean()

        # 0/1 als Step-Plot (ruhiger als normale Linie)
        ax.step(x, actual, where="post",
                label="Actual Direction (0/1)",
                linewidth=1.0, alpha=0.8)

        # Vorhersage-Wahrscheinlichkeit als Linie
        ax.plot(x, prob,
                label="Predicted Prob (up)",
                linestyle="--", linewidth=1.5)

        # Optional: gleitender Mittelwert einzeichnen
        # ax.plot(x, rolling_actual, label=f"Rolling mean ({window})", linewidth=1.2, alpha=0.9)

        ax.set_title(f"{col} – erste {n_plot} Test-Samples")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value / Probability")
        ax.set_ylim(-0.05, 1.05)  # sauberer Rahmen
        ax.grid(True, linestyle=":", alpha=0.6)

        # Accuracy in die Ecke schreiben (macht den Plot aussagekräftiger)
        ax.text(
            0.01, 0.95,
            f"Test-Acc: {test_accs[i]:.2f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=0)
        )

        ax.legend()

    # 4. Subplot: Validation-Loss
    ax4 = axes[3]
    ax4.plot(val_losses, label="Val Loss", linewidth=1.2)
    ax4.set_title("Validation Loss pro Epoch")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("BCE Loss")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend()

    plt.suptitle(
        "Multi-Horizon MLP – Actual vs Predicted (Test) & Validation Loss",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path3 = IMAGES_PATH / "06_multihorizon_mlp_actual_vs_predicted_test.png"
    plt.savefig(out_path3)
    plt.close()
    print("📈 Actual-vs-Predicted-Plots gespeichert unter:", out_path3)


    return model



def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    train_model(X_train.to(DEVICE), y_train.to(DEVICE),
                X_val.to(DEVICE),   y_val.to(DEVICE),
                X_test.to(DEVICE),  y_test.to(DEVICE))


if __name__ == "__main__":
    main()
