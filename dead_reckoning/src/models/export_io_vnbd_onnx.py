"""Export IO-VNBD Native 10 Hz GRU Model to ONNX and Benchmark Parity/Latency.

Contract:
- Input: 'imu_window', shape: [batch, 20, 6]
- Output: 'local_displacement', shape: [batch, 3]
- Dynamic batch dimension
- Window size: 20 samples (2.0s at 10 Hz)
- Feature dim: 6 [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

PROJECT_ROOT = Path("D:/dead_reckoning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_model import GRUDeadReckoning

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "gru_io_vnbd_best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "gru_io_vnbd.onnx"
WINDOW_SIZE = 20
FEATURE_DIM = 6
OUTPUT_DIM = 3
OPSET_VERSION = 17


def export_and_benchmark(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    output_path: Path = DEFAULT_OUTPUT,
    opset_version: int = OPSET_VERSION,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IO-VNBD NATIVE 10 Hz GRU ONNX EXPORT & BENCHMARK")
    print("=" * 70)

    # 1. Load PyTorch model
    model = GRUDeadReckoning(
        input_size=FEATURE_DIM,
        hidden_size=64,
        num_layers=2,
        output_size=OUTPUT_DIM,
        dropout=0.2,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded PyTorch checkpoint from {checkpoint_path}")

    # 2. Export to ONNX
    dummy_input = torch.zeros((1, WINDOW_SIZE, FEATURE_DIM), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["imu_window"],
        output_names=["local_displacement"],
        dynamic_axes={
            "imu_window": {0: "batch"},
            "local_displacement": {0: "batch"},
        },
        dynamo=False,
    )
    print(f"Exported ONNX model to {output_path}")

    # 3. Verify ONNX Model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure checked and verified.")

    # 4. Numerical Parity Verification using Real Test Windows
    test_path = PROJECT_ROOT / "data" / "processed" / "io_vnbd_test_local.npz"
    if test_path.is_file():
        test_data = np.load(test_path)
        X_test = test_data["X"][:1000].astype(np.float32)  # Use 1,000 real test samples
        with open(PROJECT_ROOT / "models" / "io_vnbd_normalization.json", "r") as f:
            norm = json.load(f)
        X_norm = (X_test - np.array(norm["mean"], dtype=np.float32)) / np.array(norm["std"], dtype=np.float32)
        test_tensor = torch.from_numpy(X_norm)
    else:
        # Fallback to random tensor if test dataset not found
        test_tensor = torch.randn((1000, WINDOW_SIZE, FEATURE_DIM), dtype=torch.float32)

    # PyTorch inference
    with torch.no_grad():
        pt_preds = model(test_tensor).numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: test_tensor.numpy()}
    ort_preds = session.run(None, ort_inputs)[0]

    # Parity metrics
    diff = np.abs(pt_preds - ort_preds)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    rmse_diff = float(np.sqrt(np.mean((pt_preds - ort_preds) ** 2)))

    print("\n--- Numerical Parity (PyTorch vs ONNX Runtime) ---")
    print(f"Evaluated on {len(test_tensor)} test windows:")
    print(f"  Maximum Absolute Difference : {max_diff:.8e}")
    print(f"  Mean Absolute Difference    : {mean_diff:.8e}")
    print(f"  Root Mean Squared Error (RMSE): {rmse_diff:.8e}")

    assert max_diff < 1e-4, f"Parity check failed: max diff {max_diff} >= 1e-4"

    # 5. Latency Benchmarking on CPU (Single-Window Execution for Edge Devices)
    single_window = test_tensor[0:1].numpy()
    ort_single_input = {session.get_inputs()[0].name: single_window}

    # Warmup
    for _ in range(50):
        _ = session.run(None, ort_single_input)

    latencies_ms = []
    num_runs = 500
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, ort_single_input)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies_ms))
    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    print("\n--- ONNX Runtime CPU Inference Latency (Window = [1, 20, 6]) ---")
    print(f"Benchmarked over {num_runs} consecutive single-window runs:")
    print(f"  Mean Latency   : {mean_lat:.3f} ms")
    print(f"  Median Latency : {median_lat:.3f} ms")
    print(f"  95th Percentile: {p95_lat:.3f} ms")
    print(f"  Min / Max      : {min_lat:.3f} ms / {max_lat:.3f} ms")
    print(f"  Effective Rate : {1000.0 / mean_lat:.1f} inferences/second (Required: 1.0 Hz step rate)")

    results = {
        "onnx_path": str(output_path),
        "input_shape": list(session.get_inputs()[0].shape),
        "output_shape": list(session.get_outputs()[0].shape),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "rmse_diff": rmse_diff,
        "mean_latency_ms": mean_lat,
        "median_latency_ms": median_lat,
        "p95_latency_ms": p95_lat,
        "inferences_per_sec": 1000.0 / mean_lat,
    }

    results_file = PROJECT_ROOT / "results" / "io_vnbd" / "onnx_parity_latency.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ONNX benchmark results to {results_file}")

    return results


if __name__ == "__main__":
    export_and_benchmark()
