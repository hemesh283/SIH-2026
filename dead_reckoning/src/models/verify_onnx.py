"""Verify numerical parity between PyTorch checkpoint and ONNX Runtime model.

Uses deterministic samples from data/processed/handheld_val_local.npz.
Computes:
- Maximum absolute difference
- Mean absolute difference
- Root Mean Squared Error (RMSE)
Enforces strict floating-point tolerance thresholds (e.g. max diff < 1e-4).
Exits with error if verification fails.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.gru_model import GRUDeadReckoning

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "gru_local_best.pt"
DEFAULT_ONNX = PROJECT_ROOT / "models" / "gru_local.onnx"
DEFAULT_NORM = PROJECT_ROOT / "models" / "gru_local_normalization.npz"
DEFAULT_VAL_DATA = PROJECT_ROOT / "data" / "processed" / "handheld_val_local.npz"
TOLERANCE_MAX_DIFF = 1e-4
TOLERANCE_RMSE = 1e-5


def verify_models(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    onnx_path: str | Path = DEFAULT_ONNX,
    norm_path: str | Path = DEFAULT_NORM,
    val_data_path: str | Path = DEFAULT_VAL_DATA,
    max_tolerance: float = TOLERANCE_MAX_DIFF,
    rmse_tolerance: float = TOLERANCE_RMSE,
) -> dict[str, float]:
    """Verify that PyTorch and ONNX Runtime produce equivalent predictions."""
    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path)
    norm_path = Path(norm_path)
    val_data_path = Path(val_data_path)

    for p in [checkpoint_path, onnx_path, norm_path, val_data_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Required file not found: {p}")

    # 1. Load normalization statistics (from training split only)
    norm = np.load(norm_path)
    X_mean = norm["X_mean"].astype(np.float32)
    X_std = norm["X_std"].astype(np.float32)
    y_mean = norm["y_mean"].astype(np.float32)
    y_std = norm["y_std"].astype(np.float32)

    # 2. Load deterministic validation dataset
    val_data = np.load(val_data_path)
    X_val = val_data["X"].astype(np.float32)
    y_val = val_data["y"].astype(np.float32)
    num_samples = len(X_val)

    # Normalize validation inputs with training statistics
    X_val_norm = (X_val - X_mean) / X_std

    # 3. PyTorch Model Evaluation
    model = GRUDeadReckoning()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pytorch_pred_norm = model(torch.from_numpy(X_val_norm)).numpy()
    pytorch_pred = pytorch_pred_norm * y_std + y_mean

    # 4. ONNX Runtime Model Evaluation
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_pred_norm = session.run(None, {input_name: X_val_norm})[0]
    onnx_pred = onnx_pred_norm * y_std + y_mean

    # 5. Compute Numerical Discrepancies
    # Normalized space differences
    norm_abs_diff = np.abs(pytorch_pred_norm - onnx_pred_norm)
    norm_max_diff = float(norm_abs_diff.max())
    norm_mean_diff = float(norm_abs_diff.mean())
    norm_rmse = float(np.sqrt(np.mean(norm_abs_diff**2)))

    # Original physical unit (meters) differences
    abs_diff = np.abs(pytorch_pred - onnx_pred)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    rmse = float(np.sqrt(np.mean(abs_diff**2)))

    results = {
        "num_samples": num_samples,
        "norm_max_diff": norm_max_diff,
        "norm_mean_diff": norm_mean_diff,
        "norm_rmse": norm_rmse,
        "max_diff_meters": max_diff,
        "mean_diff_meters": mean_diff,
        "rmse_meters": rmse,
    }

    print("=" * 60)
    print("PYTORCH VS ONNX VERIFICATION REPORT")
    print("=" * 60)
    print(f"Validation Samples Evaluated : {num_samples}")
    print(f"Input Tensor Shape           : {X_val_norm.shape}")
    print(f"Output Tensor Shape          : {pytorch_pred.shape}")
    print("-" * 60)
    print("Normalized Output Space Differences:")
    print(f"  Maximum Absolute Difference: {norm_max_diff:.8e}")
    print(f"  Mean Absolute Difference   : {norm_mean_diff:.8e}")
    print(f"  RMSE                       : {norm_rmse:.8e}")
    print("-" * 60)
    print("Physical Displacement Space Differences (meters):")
    print(f"  Maximum Absolute Difference: {max_diff:.8e} m")
    print(f"  Mean Absolute Difference   : {mean_diff:.8e} m")
    print(f"  RMSE                       : {rmse:.8e} m")
    print("-" * 60)

    # First 3 sample comparison
    print("First 3 Sample Predictions [dx, dy, dz] (meters):")
    for i in range(min(3, num_samples)):
        print(f"  Sample {i:02d} | Ground Truth: {y_val[i].tolist()}")
        print(f"            | PyTorch     : {pytorch_pred[i].tolist()}")
        print(f"            | ONNX        : {onnx_pred[i].tolist()}")
        print(f"            | Diff        : {abs_diff[i].tolist()}")

    # 6. Strict Verification Assertion
    if norm_max_diff > max_tolerance:
        raise AssertionError(
            f"VERIFICATION FAILED: Normalized max diff {norm_max_diff:.4e} exceeds threshold {max_tolerance:.4e}"
        )
    if norm_rmse > rmse_tolerance:
        raise AssertionError(
            f"VERIFICATION FAILED: Normalized RMSE {norm_rmse:.4e} exceeds threshold {rmse_tolerance:.4e}"
        )

    print("=" * 60)
    print("[SUCCESS] PyTorch and ONNX Runtime models are numerically equivalent within tolerance!")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PyTorch vs ONNX models.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX))
    parser.add_argument("--norm", type=str, default=str(DEFAULT_NORM))
    parser.add_argument("--val-data", type=str, default=str(DEFAULT_VAL_DATA))
    args = parser.parse_args()
    verify_models(args.checkpoint, args.onnx, args.norm, args.val_data)
