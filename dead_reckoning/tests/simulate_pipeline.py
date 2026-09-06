"""Realistic pipeline simulation demonstrating end-to-end deployment inference.

Pipeline steps:
1. Raw feature window acquisition [200, 6]
2. Strict shape & finite-value validation
3. Training statistics normalization
4. Trained GRU forward inference
5. 3-value physical displacement output [dx, dy, dz] in meters
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gru_inference import DeadReckoningInference

VAL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "handheld_val_local.npz"


def run_pipeline_simulation() -> None:
    print("=" * 65)
    print("DEAD RECKONING END-TO-END PIPELINE SIMULATION")
    print("=" * 65)

    # 1. Initialize Inference Engine
    print("\n[Step 1] Initializing DeadReckoningInference Engine...")
    inference_engine = DeadReckoningInference()
    print(f"  Backend loaded: {inference_engine.backend.upper()}")
    print(f"  Model path    : {inference_engine.model_path}")
    print(f"  Norm path     : {inference_engine.norm_path}")

    # 2. Acquire a real held-out validation window (simulating raw sensor stream buffer)
    print("\n[Step 2] Acquiring raw 200-sample window from held-out validation set...")
    val_data = np.load(VAL_DATA_PATH)
    raw_window = val_data["X"][0].copy()  # Shape: (200, 6)
    ground_truth = val_data["y"][0]       # Shape: (3,)

    print(f"  INPUT SHAPE: {raw_window.shape}")
    print(f"  Raw input min/max: [{raw_window.min():.4f}, {raw_window.max():.4f}]")

    # 3. Shape & Finite-Value Validation
    print("\n[Step 3] Validating input shape and numerical integrity...")
    validated_tensor = inference_engine.validate_input(raw_window)
    print("  Validation PASSED: (200, 6) finite float32 window verified.")

    # 4. Normalization inspection
    normalized_window = (validated_tensor - inference_engine.X_mean) / inference_engine.X_std
    norm_min = float(normalized_window.min())
    norm_max = float(normalized_window.max())
    print(f"  NORMALIZED INPUT RANGE: [{norm_min:.4f}, {norm_max:.4f}]")

    # 5. Run Inference
    print("\n[Step 4] Running GRU inference...")
    prediction = inference_engine.predict(raw_window)

    # 6. Output Verification
    print("\n[Step 5] Processing output displacement...")
    print(f"  OUTPUT SHAPE: {prediction.shape}")
    print(f"  PREDICTION (dx, dy, dz): {prediction.tolist()} meters")
    print(f"  GROUND TRUTH           : {ground_truth.tolist()} meters")
    error = np.abs(prediction - ground_truth)
    print(f"  Absolute Error (m)     : {error.tolist()}")
    print(f"  Euclidean Error (m)    : {np.linalg.norm(error):.4f} meters")

    # Assert contract criteria
    assert prediction.shape == (3,), f"Output shape mismatch: {prediction.shape}"
    assert np.isfinite(prediction).all(), "Prediction contains non-finite values"

    print("\n" + "=" * 65)
    print("[SUCCESS] Pipeline simulation executed successfully without error!")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline_simulation()
