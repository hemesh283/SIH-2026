"""Audit script to rigorously distinguish PyTorch vs ONNX numerical parity
from model prediction vs ground truth error.
"""

from pathlib import Path
import sys
import numpy as np
import torch
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gru_inference import DeadReckoningInference

MODEL_PT = PROJECT_ROOT / "models" / "gru_local_best.pt"
MODEL_ONNX = PROJECT_ROOT / "models" / "gru_local.onnx"
ASSET_ONNX = PROJECT_ROOT / "FE" / "gudumap" / "gudumap" / "app" / "src" / "main" / "assets" / "gru_local.onnx"
NORM_NPZ = PROJECT_ROOT / "models" / "gru_local_normalization.npz"
NORM_JSON = PROJECT_ROOT / "models" / "normalization.json"
VAL_DATA = PROJECT_ROOT / "data" / "processed" / "handheld_val_local.npz"


def main():
    print("=" * 70)
    print("AUDIT: PYTORCH VS ONNX NUMERICAL PARITY & GROUND TRUTH ERROR")
    print("=" * 70)

    # Load validation data
    val_data = np.load(VAL_DATA)
    X_val = val_data["X"].astype(np.float32)  # (N, 200, 6)
    y_val = val_data["y"].astype(np.float32)  # (N, 3)
    num_samples = len(X_val)
    print(f"Validation dataset loaded: {num_samples} windows of shape {X_val.shape[1:]}")

    # Load PyTorch inference
    infer_pt = DeadReckoningInference(model_path=MODEL_PT, normalization_path=NORM_NPZ)

    # Load ONNX inference (repo model)
    infer_onnx = DeadReckoningInference(model_path=MODEL_ONNX, normalization_path=NORM_JSON)

    # Load ONNX inference (Android asset model)
    infer_onnx_asset = DeadReckoningInference(model_path=ASSET_ONNX, normalization_path=NORM_JSON)

    # --- PART 1: SINGLE SAMPLE (X[0]) DETAILED AUDIT ---
    print("\n" + "-" * 70)
    print("PART 1: DETAILED AUDIT ON FIRST VALIDATION SAMPLE (X[0])")
    print("-" * 70)
    x0 = X_val[0]  # (200, 6)
    gt0 = y_val[0] # (3,)

    pred_pt_0 = infer_pt.predict(x0)
    pred_onnx_0 = infer_onnx.predict(x0)
    pred_asset_0 = infer_onnx_asset.predict(x0)

    print(f"Sample 0 Ground Truth [dx, dy, dz]: {gt0.tolist()}")
    print(f"Sample 0 PyTorch Pred [dx, dy, dz] : {pred_pt_0.tolist()}")
    print(f"Sample 0 ONNX Pred    [dx, dy, dz] : {pred_onnx_0.tolist()}")
    print(f"Sample 0 Asset ONNX   [dx, dy, dz] : {pred_asset_0.tolist()}")

    # Parity metrics on sample 0
    diff_0 = np.abs(pred_pt_0 - pred_onnx_0)
    max_abs_diff_0 = float(np.max(diff_0))
    mean_abs_diff_0 = float(np.mean(diff_0))
    rmse_parity_0 = float(np.sqrt(np.mean((pred_pt_0 - pred_onnx_0) ** 2)))

    print("\n[Sample 0 - PyTorch vs ONNX Parity]")
    print(f"  Max Absolute Difference : {max_abs_diff_0:.10e} m")
    print(f"  Mean Absolute Difference: {mean_abs_diff_0:.10e} m")
    print(f"  RMSE                    : {rmse_parity_0:.10e} m")

    # Error vs ground truth on sample 0
    err_pt_gt_0 = np.abs(pred_pt_0 - gt0)
    euclidean_err_0 = float(np.linalg.norm(pred_pt_0 - gt0))
    print("\n[Sample 0 - Model Prediction vs Ground Truth]")
    print(f"  Per-axis Absolute Error [dx, dy, dz]: {err_pt_gt_0.tolist()} m")
    print(f"  Max Absolute Error                  : {float(np.max(err_pt_gt_0)):.6f} m")
    print(f"  Mean Absolute Error                 : {float(np.mean(err_pt_gt_0)):.6f} m")
    print(f"  RMSE against GT                     : {float(np.sqrt(np.mean((pred_pt_0 - gt0) ** 2))):.6f} m")
    print(f"  Euclidean Error (norm of error)     : {euclidean_err_0:.6f} m ({euclidean_err_0:.4f} m)")

    # --- PART 2: FULL VALIDATION SET (ALL SAMPLES) AUDIT ---
    print("\n" + "-" * 70)
    print(f"PART 2: DATASET-WIDE AUDIT ACROSS ALL {num_samples} VALIDATION SAMPLES")
    print("-" * 70)

    # Batch prediction for efficiency
    preds_pt = infer_pt.predict_batch(X_val)       # (N, 3)
    preds_onnx = infer_onnx.predict_batch(X_val)   # (N, 3)

    # Parity: PyTorch vs ONNX across all samples
    diff_all = np.abs(preds_pt - preds_onnx)
    max_abs_diff_all = float(np.max(diff_all))
    mean_abs_diff_all = float(np.mean(diff_all))
    rmse_parity_all = float(np.sqrt(np.mean((preds_pt - preds_onnx) ** 2)))

    print("\n[Full Dataset - PyTorch vs ONNX Numerical Parity]")
    print(f"  Max Absolute Difference : {max_abs_diff_all:.10e} m")
    print(f"  Mean Absolute Difference: {mean_abs_diff_all:.10e} m")
    print(f"  RMSE                    : {rmse_parity_all:.10e} m")

    # Model accuracy: Predictions vs Ground Truth across all samples
    err_gt_all = np.abs(preds_onnx - y_val) # (N, 3)
    max_err_gt_all = float(np.max(err_gt_all))
    mean_err_gt_all = float(np.mean(err_gt_all))
    rmse_gt_all = float(np.sqrt(np.mean((preds_onnx - y_val) ** 2)))
    euclidean_err_all = np.linalg.norm(preds_onnx - y_val, axis=1) # (N,)
    mean_euclidean_err_all = float(np.mean(euclidean_err_all))

    print("\n[Full Dataset - Model Prediction vs Ground Truth Error]")
    print(f"  Max Absolute Error       : {max_err_gt_all:.6f} m")
    print(f"  Mean Absolute Error (MAE): {mean_err_gt_all:.6f} m")
    print(f"  RMSE against Ground Truth: {rmse_gt_all:.6f} m")
    print(f"  Mean Euclidean Error     : {mean_euclidean_err_all:.6f} m")

    # Verification of models/gru_local.onnx vs Android asset gru_local.onnx
    preds_asset = infer_onnx_asset.predict_batch(X_val)
    asset_diff = float(np.max(np.abs(preds_onnx - preds_asset)))
    print(f"\n[Asset Check] models/gru_local.onnx vs Android asset gru_local.onnx max diff: {asset_diff:.10e} m")

    print("\n" + "=" * 70)
    print("AUDIT CONCLUSION")
    print("=" * 70)
    if abs(euclidean_err_0 - 0.0128) < 0.0001:
        print("CONFIRMED: The reported '0.0128 m' is EXACTLY the Sample 0 Euclidean Error")
        print("between model prediction and ground truth (np.linalg.norm(prediction - ground_truth)).")
        print("It is NOT the numerical difference between PyTorch and ONNX Runtime!")
        print(f"The actual PyTorch vs ONNX parity difference is ~{mean_abs_diff_all:.2e} m (floating-point precision).")
    print("=" * 70)


if __name__ == "__main__":
    main()
