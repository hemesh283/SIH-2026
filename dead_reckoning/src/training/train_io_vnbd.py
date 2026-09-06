"""Vehicle-Domain Training & Fine-Tuning for IO-VNBD Native 10 Hz GRU.

Initializes weights from models/gru_local_best.pt (OxIOD pre-trained),
adapts to automotive kinematic scales using IO-VNBD train split (45,742 windows),
evaluates validation loss on held-out Driver A & Driver E sequences (33,262 windows),
and saves the best model checkpoint to models/gru_io_vnbd_best.pt.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path("D:/dead_reckoning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_model import GRUDeadReckoning

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "io_vnbd"


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Load Data
    print("\nLoading IO-VNBD processed datasets...")
    train_data = np.load(DATA_DIR / "io_vnbd_train_local.npz")
    val_data = np.load(DATA_DIR / "io_vnbd_val_local.npz")

    X_train_raw = train_data["X"]  # (45742, 20, 6)
    y_train_raw = train_data["y"]  # (45742, 3)
    X_val_raw = val_data["X"]      # (33262, 20, 6)
    y_val_raw = val_data["y"]      # (33262, 3)

    print(f"Train shapes: X={X_train_raw.shape}, y={y_train_raw.shape}")
    print(f"Val shapes  : X={X_val_raw.shape}, y={y_val_raw.shape}")

    # 2. Load IO-VNBD Normalization Statistics
    with open(MODELS_DIR / "io_vnbd_normalization.json", "r", encoding="utf-8") as f:
        norm_info = json.load(f)

    X_mean = np.array(norm_info["mean"], dtype=np.float32)
    X_std = np.array(norm_info["std"], dtype=np.float32)
    y_mean = np.array(norm_info["target_mean"], dtype=np.float32)
    y_std = np.array(norm_info["target_std"], dtype=np.float32)

    # Normalize inputs and targets
    X_train = (X_train_raw - X_mean) / X_std
    y_train = (y_train_raw - y_mean) / y_std
    X_val = (X_val_raw - X_mean) / X_std
    y_val = (y_val_raw - y_mean) / y_std

    # Create DataLoaders
    batch_size = 128
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    # 3. Initialize Model and Load Pretrained Weights
    model = GRUDeadReckoning(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout=0.2,
    ).to(device)

    pretrained_path = MODELS_DIR / "gru_local_best.pt"
    if pretrained_path.is_file():
        print(f"Initializing from pretrained OxIOD weights: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # Load compatible recurrent layers
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully transferred {len(pretrained_dict)} weight tensors.")

    # 4. Optimizer, Scheduler, Loss
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    # 5. Training Loop
    best_val_loss = float("inf")
    best_checkpoint_path = MODELS_DIR / "gru_io_vnbd_best.pt"
    patience = 5
    patience_counter = 0

    train_losses = []
    val_losses = []

    print("\nStarting Training & Domain Adaptation...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_preds = model(vx)
                v_loss = criterion(v_preds, vy)
                epoch_val_loss += v_loss.item()
                val_batches += 1

        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Checkpoint Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "train_loss": avg_train_loss,
                "architecture": "GRUDeadReckoning",
                "input_shape": [None, 20, 6],
                "output_shape": [None, 3],
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path.name} (Val Loss: {best_val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Patience: {patience})")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds. Best Val Loss: {best_val_loss:.5f}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss (Smooth L1)", linewidth=2.0)
    plt.plot(val_losses, label="Val Loss (Smooth L1)", linewidth=2.0)
    plt.title("IO-VNBD GRU Vehicle Domain Fine-Tuning Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    curve_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {curve_path}")


if __name__ == "__main__":
    train()
