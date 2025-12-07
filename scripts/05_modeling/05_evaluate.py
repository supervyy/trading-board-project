
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

def load_data():
    print("⏳ Loading validation data from", DATA_PATH)
    try:
        X_train = np.load(DATA_PATH / "X_train.npy")
        y_train = np.load(DATA_PATH / "y_train.npy")
        X_val = np.load(DATA_PATH / "X_val.npy")
        y_val = np.load(DATA_PATH / "y_val.npy")
        
        # Binarize targets
        y_train = (y_train > 0).astype(np.float32)
        y_val = (y_val > 0).astype(np.float32)
        
        # Convert to Tensor
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).view(-1, 1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).view(-1, 1)
        
        return X_train_t, y_train_t, X_val_t, y_val_t
    except FileNotFoundError:
        print("❌ Error: Data files not found.")
        sys.exit(1)

def evaluate_models(X_train, y_train, X_val, y_val):
    print("\n📊 Evaluating Models...")
    
    # 1. Evaluate Neural Network
    nn_path = MODELS_PATH / "nn_model.pt"
    if nn_path.exists():
        print(f"   Loading NN from {nn_path}...")
        
        input_dim = X_train.shape[1]
        model = SimpleNN(input_dim)
        model.load_state_dict(torch.load(nn_path))
        model.eval()
        
        # Evaluate Train
        with torch.no_grad():
            train_preds = model(X_train)
            train_preds_cls = (train_preds > 0.5).float()
            train_acc = (train_preds_cls.eq(y_train).sum() / y_train.shape[0]).item()
            
            val_preds = model(X_val)
            val_preds_cls = (val_preds > 0.5).float()
            val_acc = (val_preds_cls.eq(y_val).sum() / y_val.shape[0]).item()
        
        print("\n   🧠 Neural Network Results:")
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Val Accuracy:   {val_acc:.4f}")
    else:
        print("   ⚠️ NN model not found.")

    # 2. Evaluate Decision Tree
    tree_path = MODELS_PATH / "tree_entry.pkl"
    if tree_path.exists():
        print(f"\n   Loading Decision Tree from {tree_path}...")
        tree_model = joblib.load(tree_path)
        
        # Convert back to numpy for sklearn
        X_train_np = X_train.numpy()
        y_train_np = y_train.numpy().ravel()
        X_val_np = X_val.numpy()
        y_val_np = y_val.numpy().ravel()
        
        train_acc = tree_model.score(X_train_np, y_train_np)
        val_acc = tree_model.score(X_val_np, y_val_np)
        
        print("\n   🌳 Decision Tree Results:")
        print(f"      Train Accuracy: {train_acc:.4f}")
        print(f"      Val Accuracy:   {val_acc:.4f}")
    else:
        print("   ⚠️ Tree model not found.")

def main():
    X_train, y_train, X_val, y_val = load_data()
    evaluate_models(X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()
