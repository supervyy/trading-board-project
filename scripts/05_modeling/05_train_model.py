
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os
import shutil

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
IMAGES_PATH = PROJECT_ROOT / "images" / "modeling"

MODELS_PATH.mkdir(parents=True, exist_ok=True)
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

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
    print("⏳ Loading data from", DATA_PATH)
    try:
        X_train = np.load(DATA_PATH / "X_train.npy")
        y_train = np.load(DATA_PATH / "y_train.npy")
        X_val = np.load(DATA_PATH / "X_val.npy")
        y_val = np.load(DATA_PATH / "y_val.npy")
        
        # Binarize targets: 1 if > 0, else 0
        y_train = (y_train > 0).astype(np.float32)
        y_val = (y_val > 0).astype(np.float32)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
        
        print(f"   ✅ Data Loaded & Binarized: X_train: {X_train.shape}, y_train: {y_train.shape}")
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_train, y_train, X_val, y_val
        
    except FileNotFoundError:
        print("❌ Error: Data files not found at", DATA_PATH)
        print("Please run 04_main_post_split.py first.")
        sys.exit(1)

def train_neural_network(X_train, y_train, X_val, y_val):
    print("\n🧠 Training Neural Network (PyTorch MLP)...")
    
    input_dim = X_train.shape[1]
    model = SimpleNN(input_dim)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    epochs = 20
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            
        epoch_loss = running_loss / len(loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss_val = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss_val)
            
        print(f"   Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss_val:.4f}")
    
    # Save Model state dict (safer than full model pickling if class is missing)
    # BUT user asked for "Save models/nn_model.pt (PyTorch model)"
    # I'll save the state_dict for robustness, and we load it by instantiating the class.
    
    model_path = MODELS_PATH / "nn_model.pt"
    # Saving full model state. 
    # To facilitate easy loading in evaluate script, I will save state_dict 
    # because that is standard practice.
    torch.save(model.state_dict(), model_path)
    print(f"   ✅ Model saved to {model_path}")
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Model Loss (PyTorch)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = IMAGES_PATH / "loss_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"   ✅ Loss curve saved to {plot_path}")
    
    return model

def train_decision_tree(X_train, y_train, X_val, y_val):
    print("\nbf Training Decision Tree...")
    
    # Sklearn needs numpy
    # X_train is tensor, convert back or use original numpy args
    # Function receives converted tensors AND original numpy arrays conceptually
    # but my loader returned mixed. Let's rely on sklearn handling tensors (it usually converts)
    # OR better, pass numpy versions explicitly.
    
    # I updated load_data to return numpy versions too.
    # Wait, function signature has fixed args. 
    # I will just call .numpy() on tensors.
    
    X_train_np = X_train.numpy()
    y_train_np = y_train.numpy().ravel() # Flatten to 1D
    X_val_np = X_val.numpy()
    y_val_np = y_val.numpy().ravel()
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_np, y_train_np)
    
    depth = clf.get_depth()
    leaves = clf.get_n_leaves()
    
    print(f"   Tree Depth: {depth}")
    print(f"   Number of Leaves: {leaves}")
    
    val_acc = clf.score(X_val_np, y_val_np)
    print(f"   Validation Accuracy: {val_acc:.4f}")
    
    model_path = MODELS_PATH / "tree_entry.pkl"
    joblib.dump(clf, model_path)
    print(f"   ✅ Tree model saved to {model_path}")
    
    return clf

def calculate_baselines(y_train, y_val):
    print("\n📏 Calculating Baselines...")
    
    y_train_np = y_train.numpy().ravel()
    y_val_np = y_val.numpy().ravel()
    
    # Baseline 1: Always predict 1
    # Check class balance in train
    prop_1 = np.mean(y_train_np)
    print(f"   Proportion of 1s in Train: {prop_1:.4f}")
    
    # Always 1 Accuracy on Val
    # If we predict 1, accuracy is simple sum(y)/len(y)
    acc_always_1 = np.mean(y_val_np)
    # (If we predict 1, hits are where y_val is 1)
    print(f"   Baseline (Always 1): {acc_always_1:.4f}")
    
    # Baseline 2: Random Prediction
    # 50/50 Random
    random_preds = np.random.randint(0, 2, size=len(y_val_np))
    acc_random = accuracy_score(y_val_np, random_preds)
    print(f"   Baseline (Random 50/50): {acc_random:.4f}")

def main():
    # 1. Load Data
    X_train_t, y_train_t, X_val_t, y_val_t, _, _, _, _ = load_data()
    
    # 2. Train and Save NN
    train_neural_network(X_train_t, y_train_t, X_val_t, y_val_t)
    
    # 3. Train and Save Decision Tree
    train_decision_tree(X_train_t, y_train_t, X_val_t, y_val_t)
    
    # 4. Baselines
    calculate_baselines(y_train_t, y_val_t)
    
    print("\n✅ Modeling pipeline complete.")

if __name__ == "__main__":
    main()
