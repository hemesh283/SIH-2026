"""Train the fixed GRU on local-frame OxIOD displacement targets.

Normalization statistics are calculated exclusively from the local training NPZ.
Validation is used only for checkpoint selection; test data are intentionally
not loaded or evaluated by this experiment script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.gru_model import GRUDeadReckoning


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    """Set NumPy, Python, and Torch seeds for reproducible CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # The small fixed dataset is faster and more reproducible without CPU
    # thread oversubscription on the desktop host.
    torch.set_num_threads(1)


def load_local_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite local-target arrays, without using sequence IDs for fitting."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"local dataset does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"X", "y", "sequence_ids"}:
            raise ValueError(f"{path}: expected X, y, sequence_ids")
        X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    if X.ndim != 3 or X.shape[1:] != (200, 6) or y.ndim != 2 or y.shape[1:] != (3,):
        raise ValueError(f"{path}: unexpected X/y shapes {X.shape}, {y.shape}")
    if len(X) != len(y) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{path}: invalid non-finite or unmatched arrays")
    return X, y


def calculate_normalization(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return feature and target statistics computed from train data only."""
    X_mean, X_std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1))
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    if np.any(X_std == 0) or np.any(y_std == 0):
        raise ValueError("training normalization has a zero standard deviation")
    return X_mean, X_std, y_mean, y_std


def normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply caller-provided normalization statistics without recomputing them."""
    return (values - mean) / std


def evaluate_normalized(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray]:
    """Return normalized MSE and normalized predictions for a non-shuffled loader."""
    model.eval()
    losses, predictions = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            prediction = model(X_batch.to(DEVICE))
            losses.append(criterion(prediction, y_batch.to(DEVICE)).item() * len(X_batch))
            predictions.append(prediction.cpu().numpy())
    return sum(losses) / len(loader.dataset), np.concatenate(predictions)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute aggregate and per-local-component MSE/RMSE in original units."""
    squared_error = (y_true - y_pred) ** 2
    metrics = {"overall_mse": float(squared_error.mean()), "rmse": float(np.sqrt(squared_error.mean()))}
    for index, axis in enumerate(("x", "y", "z")):
        mse = float(squared_error[:, index].mean())
        metrics[f"local_{axis}_mse"] = mse
        metrics[f"local_{axis}_rmse"] = float(np.sqrt(mse))
    return metrics


def save_plots(train_losses: list[float], validation_losses: list[float], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save training curve and validation true-vs-predicted local-component plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Normalized target MSE"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gru_local_loss.png", dpi=150); plt.close()
    for index, axis in enumerate(("dx", "dy", "dz")):
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true[:, index], y_pred[:, index], s=12, alpha=0.65)
        low, high = min(y_true[:, index].min(), y_pred[:, index].min()), max(y_true[:, index].max(), y_pred[:, index].max())
        plt.plot([low, high], [low, high], "k--", linewidth=1)
        plt.xlabel(f"True local Δ{axis[-1].upper()}"); plt.ylabel(f"Predicted local Δ{axis[-1].upper()}")
        plt.tight_layout(); plt.savefig(RESULTS_DIR / f"gru_local_validation_{axis}.png", dpi=150); plt.close()


def print_comparison(rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print original-unit validation metrics for baselines and the GRU."""
    print("\nVALIDATION COMPARISON (original local-target units)")
    print("Model | Overall MSE | RMSE | Local X RMSE | Local Y RMSE | Local Z RMSE")
    for name, metric in rows:
        print(f"{name} | {metric['overall_mse']:.6f} | {metric['rmse']:.6f} | "
              f"{metric['local_x_rmse']:.6f} | {metric['local_y_rmse']:.6f} | {metric['local_z_rmse']:.6f}")


def main() -> None:
    """Run the fixed, train/validation-only local-target GRU experiment."""
    set_seed()
    X_train, y_train = load_local_npz(PROCESSED_DIR / "handheld_train_local.npz")
    X_val, y_val = load_local_npz(PROCESSED_DIR / "handheld_val_local.npz")
    X_mean, X_std, y_mean, y_std = calculate_normalization(X_train, y_train)
    X_train_norm, X_val_norm = normalize(X_train, X_mean, X_std), normalize(X_val, X_mean, X_std)
    y_train_norm, y_val_norm = normalize(y_train, y_mean, y_std), normalize(y_val, y_mean, y_std)

    validation_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_norm), torch.from_numpy(y_val_norm)), batch_size=BATCH_SIZE, shuffle=False)
    model = GRUDeadReckoning().to(DEVICE)
    criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / "gru_local_best.pt"
    state_path = MODEL_DIR / "gru_local_training_state.pt"
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError("cannot resume: training state does not exist")
        state = torch.load(state_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        train_losses, validation_losses = state["train_losses"], state["validation_losses"]
        best_loss, best_epoch, start_epoch = state["best_loss"], state["best_epoch"], state["epoch"]
    else:
        train_losses, validation_losses, best_loss, best_epoch, start_epoch = [], [], float("inf"), 0, 0

    end_epoch = min(start_epoch + args.epochs_per_run, EPOCHS)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        generator = torch.Generator().manual_seed(SEED + epoch)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_norm), torch.from_numpy(y_train_norm)), batch_size=BATCH_SIZE, shuffle=True, generator=generator)
        model.train(); total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch.to(DEVICE)), y_batch.to(DEVICE))
            loss.backward(); optimizer.step()
            total_loss += loss.item() * len(X_batch)
        train_loss = total_loss / len(train_loader.dataset)
        validation_loss, _ = evaluate_normalized(model, validation_loader, criterion)
        train_losses.append(train_loss); validation_losses.append(validation_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} - train loss: {train_loss:.6f} - validation loss: {validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            torch.save({"epoch": epoch, "validation_loss": validation_loss, "model_state_dict": model.state_dict()}, checkpoint_path)
        torch.save({"epoch": epoch, "best_epoch": best_epoch, "best_loss": best_loss,
                    "train_losses": train_losses, "validation_losses": validation_losses,
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state()}, state_path)

    if end_epoch < EPOCHS:
        print(f"Training progress saved at epoch {end_epoch}/{EPOCHS}. Resume to continue.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, y_val_pred_norm = evaluate_normalized(model, validation_loader, criterion)
    y_val_pred = y_val_pred_norm * y_std + y_mean
    np.savez(MODEL_DIR / "gru_local_normalization.npz", X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)

    zero_metrics = regression_metrics(y_val, np.zeros_like(y_val))
    mean_metrics = regression_metrics(y_val, np.broadcast_to(y_mean, y_val.shape))
    model_metrics = regression_metrics(y_val, y_val_pred)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss (normalized targets): {best_loss:.6f}")
    print_comparison([("Zero baseline", zero_metrics), ("Mean baseline", mean_metrics), ("Local-target GRU", model_metrics)])
    save_plots(train_losses, validation_losses, y_val, y_val_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fixed local-target GRU experiment.")
    parser.add_argument("--resume", action="store_true", help="resume from the saved training state")
    parser.add_argument("--epochs-per-run", type=int, default=EPOCHS, help="epochs to run before saving progress")
    args = parser.parse_args()
    if args.epochs_per_run <= 0:
        parser.error("--epochs-per-run must be positive")
    main()
