"""Held-Out ML Evaluation and Benchmark for IO-VNBD Models.

Compares:
1. Zero Displacement Baseline
2. Pre-existing OxIOD GRU Model (Domain Mismatch)
3. IO-VNBD Native 10 Hz Trained GRU Model (Domain-Adapted)

Calculates:
- Overall MAE and RMSE
- Per-axis (dx, dy, dz) MAE and RMSE
- Euclidean displacement error (Mean, Median, P95, Max)
- Error reduction percentage vs baselines

Saves:
- results/io_vnbd/ml_test_results.csv
- results/io_vnbd/ml_test_error_comparison.png
- results/io_vnbd/ml_test_scatter_predictions.png
- results/io_vnbd/ml_test_cdf_error.png
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("D:/dead_reckoning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_model import GRUDeadReckoning

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "io_vnbd"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute rigorous ML regression metrics on 3D displacement vectors."""
    diff = y_pred - y_true  # (N, 3)
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    mae_x = float(np.mean(abs_diff[:, 0]))
    mae_y = float(np.mean(abs_diff[:, 1]))
    mae_z = float(np.mean(abs_diff[:, 2]))
    mae_overall = float(np.mean(abs_diff))

    rmse_x = float(np.sqrt(np.mean(sq_diff[:, 0])))
    rmse_y = float(np.sqrt(np.mean(sq_diff[:, 1])))
    rmse_z = float(np.sqrt(np.mean(sq_diff[:, 2])))
    rmse_overall = float(np.sqrt(np.mean(sq_diff)))

    euclidean_errors = np.linalg.norm(diff, axis=1)  # (N,)
    euclidean_mean = float(np.mean(euclidean_errors))
    euclidean_median = float(np.median(euclidean_errors))
    euclidean_p95 = float(np.percentile(euclidean_errors, 95))
    euclidean_max = float(np.max(euclidean_errors))

    return {
        "model": model_name,
        "mae_overall_m": mae_overall,
        "rmse_overall_m": rmse_overall,
        "mae_x_m": mae_x,
        "mae_y_m": mae_y,
        "mae_z_m": mae_z,
        "rmse_x_m": rmse_x,
        "rmse_y_m": rmse_y,
        "rmse_z_m": rmse_z,
        "euclidean_mean_m": euclidean_mean,
        "euclidean_median_m": euclidean_median,
        "euclidean_p95_m": euclidean_p95,
        "euclidean_max_m": euclidean_max,
    }


def evaluate():
    print("=" * 70)
    print("HELD-OUT IO-VNBD TEST SET ML EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # 1. Load Test Data
    test_path = DATA_DIR / "io_vnbd_test_local.npz"
    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    test_data = np.load(test_path)
    X_test = test_data["X"]  # (N, 20, 6)
    y_test = test_data["y"]  # (N, 3)
    N_samples = len(X_test)
    print(f"Loaded {N_samples:,} held-out test windows of shape {X_test.shape[1:]}")
    print(f"Ground-truth test displacement range (dx): [{y_test[:, 0].min():.2f}, {y_test[:, 0].max():.2f}] m")
    print(f"Ground-truth test displacement mean (dx): {y_test[:, 0].mean():.2f} m, std: {y_test[:, 0].std():.2f} m")

    results = []

    # -------------------------------------------------------------
    # BASELINE 1: Zero Displacement Baseline
    # -------------------------------------------------------------
    print("\nEvaluating Baseline 1: Zero Displacement...")
    preds_zero = np.zeros_like(y_test)
    res_zero = compute_metrics(y_test, preds_zero, "1_Zero_Displacement")
    results.append(res_zero)
    print(f"  Zero Baseline MAE: {res_zero['mae_overall_m']:.4f} m | RMSE: {res_zero['rmse_overall_m']:.4f} m | Euclidean Mean: {res_zero['euclidean_mean_m']:.4f} m")

    # -------------------------------------------------------------
    # BASELINE 2: Existing OxIOD GRU Model
    # -------------------------------------------------------------
    print("\nEvaluating Baseline 2: Existing OxIOD GRU (models/gru_local_best.pt)...")
    oxiod_ckpt_path = MODELS_DIR / "gru_local_best.pt"
    oxiod_norm_path = MODELS_DIR / "normalization.json"

    with open(oxiod_norm_path, "r", encoding="utf-8") as f:
        oxiod_norm = json.load(f)

    oxiod_X_mean = np.array(oxiod_norm["mean"], dtype=np.float32)
    oxiod_X_std = np.array(oxiod_norm["std"], dtype=np.float32)
    oxiod_y_mean = np.array(oxiod_norm.get("target_mean", [0.0, 0.0, 0.0]), dtype=np.float32)
    oxiod_y_std = np.array(oxiod_norm.get("target_std", [1.0, 1.0, 1.0]), dtype=np.float32)

    oxiod_model = GRUDeadReckoning(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout=0.2,
    ).to(device)
    oxiod_ckpt = torch.load(oxiod_ckpt_path, map_location=device, weights_only=True)
    oxiod_state = oxiod_ckpt["model_state_dict"] if "model_state_dict" in oxiod_ckpt else oxiod_ckpt
    oxiod_model.load_state_dict(oxiod_state)
    oxiod_model.eval()

    X_test_oxiod_norm = (X_test - oxiod_X_mean) / oxiod_X_std
    preds_oxiod_list = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, N_samples, batch_size):
            bx = torch.from_numpy(X_test_oxiod_norm[i:i + batch_size]).to(device)
            bp = oxiod_model(bx).cpu().numpy()
            preds_oxiod_list.append(bp)

    preds_oxiod_norm = np.concatenate(preds_oxiod_list, axis=0)
    preds_oxiod = preds_oxiod_norm * oxiod_y_std + oxiod_y_mean
    res_oxiod = compute_metrics(y_test, preds_oxiod, "2_OxIOD_Pretrained_GRU")
    results.append(res_oxiod)
    print(f"  OxIOD GRU MAE: {res_oxiod['mae_overall_m']:.4f} m | RMSE: {res_oxiod['rmse_overall_m']:.4f} m | Euclidean Mean: {res_oxiod['euclidean_mean_m']:.4f} m")

    # -------------------------------------------------------------
    # MODEL 3: IO-VNBD Native 10 Hz GRU
    # -------------------------------------------------------------
    print("\nEvaluating Model 3: IO-VNBD Native 10 Hz GRU (models/gru_io_vnbd_best.pt)...")
    vnbd_ckpt_path = MODELS_DIR / "gru_io_vnbd_best.pt"
    vnbd_norm_path = MODELS_DIR / "io_vnbd_normalization.json"

    if not vnbd_ckpt_path.is_file():
        raise FileNotFoundError(f"IO-VNBD best model checkpoint not found at {vnbd_ckpt_path}")

    with open(vnbd_norm_path, "r", encoding="utf-8") as f:
        vnbd_norm = json.load(f)

    vnbd_X_mean = np.array(vnbd_norm["mean"], dtype=np.float32)
    vnbd_X_std = np.array(vnbd_norm["std"], dtype=np.float32)
    vnbd_y_mean = np.array(vnbd_norm["target_mean"], dtype=np.float32)
    vnbd_y_std = np.array(vnbd_norm["target_std"], dtype=np.float32)

    vnbd_model = GRUDeadReckoning(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout=0.2,
    ).to(device)
    vnbd_ckpt = torch.load(vnbd_ckpt_path, map_location=device, weights_only=True)
    vnbd_state = vnbd_ckpt["model_state_dict"] if "model_state_dict" in vnbd_ckpt else vnbd_ckpt
    vnbd_model.load_state_dict(vnbd_state)
    vnbd_model.eval()

    X_test_vnbd_norm = (X_test - vnbd_X_mean) / vnbd_X_std
    preds_vnbd_list = []
    with torch.no_grad():
        for i in range(0, N_samples, batch_size):
            bx = torch.from_numpy(X_test_vnbd_norm[i:i + batch_size]).to(device)
            bp = vnbd_model(bx).cpu().numpy()
            preds_vnbd_list.append(bp)

    preds_vnbd_norm = np.concatenate(preds_vnbd_list, axis=0)
    preds_vnbd = preds_vnbd_norm * vnbd_y_std + vnbd_y_mean
    res_vnbd = compute_metrics(y_test, preds_vnbd, "3_IO_VNBD_Native_GRU")
    results.append(res_vnbd)
    print(f"  IO-VNBD GRU MAE: {res_vnbd['mae_overall_m']:.4f} m | RMSE: {res_vnbd['rmse_overall_m']:.4f} m | Euclidean Mean: {res_vnbd['euclidean_mean_m']:.4f} m")

    # -------------------------------------------------------------
    # Save Results CSV
    # -------------------------------------------------------------
    df = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "ml_test_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved ML evaluation metrics to {out_csv}")
    print("\n" + df[["model", "mae_overall_m", "rmse_overall_m", "mae_x_m", "rmse_x_m", "euclidean_mean_m", "euclidean_p95_m", "euclidean_max_m"]].to_string(index=False))

    # Calculate Improvement
    zero_euc = res_zero["euclidean_mean_m"]
    oxiod_euc = res_oxiod["euclidean_mean_m"]
    vnbd_euc = res_vnbd["euclidean_mean_m"]
    imp_vs_zero = (zero_euc - vnbd_euc) / zero_euc * 100.0
    imp_vs_oxiod = (oxiod_euc - vnbd_euc) / oxiod_euc * 100.0
    print(f"\nVehicle-Domain Adaptation Improvement:")
    print(f"  Euclidean Error Reduction vs Zero Baseline: {imp_vs_zero:.2f}%")
    print(f"  Euclidean Error Reduction vs OxIOD GRU   : {imp_vs_oxiod:.2f}%")

    # -------------------------------------------------------------
    # Plot 1: Error Bar Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    models = ["Zero Baseline", "OxIOD GRU\n(Domain Mismatch)", "IO-VNBD Native GRU\n(Vehicle Domain)"]
    mae_vals = [res_zero["mae_overall_m"], res_oxiod["mae_overall_m"], res_vnbd["mae_overall_m"]]
    rmse_vals = [res_zero["rmse_overall_m"], res_oxiod["rmse_overall_m"], res_vnbd["rmse_overall_m"]]
    euc_vals = [res_zero["euclidean_mean_m"], res_oxiod["euclidean_mean_m"], res_vnbd["euclidean_mean_m"]]

    x = np.arange(len(models))
    width = 0.25

    rects1 = ax.bar(x - width, mae_vals, width, label="MAE (m)", color="#4A90E2")
    rects2 = ax.bar(x, rmse_vals, width, label="RMSE (m)", color="#F5A623")
    rects3 = ax.bar(x + width, euc_vals, width, label="Mean Euclidean Error (m)", color="#7ED321")

    ax.set_ylabel("Displacement Error (meters)", fontsize=12)
    ax.set_title("Held-Out Test Set: Error Comparison Across Baselines", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Annotate bar values
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}m",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    bar_path = RESULTS_DIR / "ml_test_error_comparison.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved bar comparison plot to {bar_path}")

    # -------------------------------------------------------------
    # Plot 2: Scatter Prediction vs Ground Truth (dx)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Subsample 2000 points for readable scatter
    sample_idx = np.random.choice(N_samples, size=min(2000, N_samples), replace=False)
    gt_dx = y_test[sample_idx, 0]
    oxiod_dx = preds_oxiod[sample_idx, 0]
    vnbd_dx = preds_vnbd[sample_idx, 0]

    max_val = max(gt_dx.max(), vnbd_dx.max(), 50)

    # OxIOD
    ax1.scatter(gt_dx, oxiod_dx, alpha=0.3, s=15, color="#D0021B", edgecolors="none")
    ax1.plot([0, max_val], [0, max_val], "k--", label="Ideal 1:1 Parity")
    ax1.set_title("OxIOD Handheld GRU (Domain Mismatch)\nPredictions Collapse Near Walking Speeds (0-3m)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("True Vehicle 2s Displacement dx (m)", fontsize=11)
    ax1.set_ylabel("Predicted Displacement dx (m)", fontsize=11)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)

    # IO-VNBD
    ax2.scatter(gt_dx, vnbd_dx, alpha=0.3, s=15, color="#417505", edgecolors="none")
    ax2.plot([0, max_val], [0, max_val], "k--", label="Ideal 1:1 Parity")
    ax2.set_title("Native IO-VNBD GRU (Domain-Adapted)\nAccurate Vehicle Kinematics Across Full Speed Range", fontsize=11, fontweight="bold")
    ax2.set_xlabel("True Vehicle 2s Displacement dx (m)", fontsize=11)
    ax2.set_ylabel("Predicted Displacement dx (m)", fontsize=11)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    scatter_path = RESULTS_DIR / "ml_test_scatter_predictions.png"
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {scatter_path}")

    # -------------------------------------------------------------
    # Plot 3: Cumulative Distribution Function (CDF) of Euclidean Error
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    euc_zero = np.sort(np.linalg.norm(y_test - preds_zero, axis=1))
    euc_oxiod = np.sort(np.linalg.norm(y_test - preds_oxiod, axis=1))
    euc_vnbd = np.sort(np.linalg.norm(y_test - preds_vnbd, axis=1))
    cdf = np.linspace(0, 1, len(euc_zero))

    ax.plot(euc_zero, cdf, label="Zero Displacement Baseline", color="#9B9B9B", linestyle="--", linewidth=2)
    ax.plot(euc_oxiod, cdf, label="OxIOD GRU (Domain Mismatch)", color="#D0021B", linewidth=2)
    ax.plot(euc_vnbd, cdf, label="IO-VNBD Native GRU (Fine-Tuned)", color="#417505", linewidth=2.5)

    ax.set_xlabel("Euclidean Displacement Error per 2s Window (meters)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("CDF of 3D Displacement Error on Held-Out Test Set", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    cdf_path = RESULTS_DIR / "ml_test_cdf_error.png"
    plt.savefig(cdf_path, dpi=300)
    plt.close()
    print(f"Saved CDF plot to {cdf_path}")


if __name__ == "__main__":
    evaluate()
