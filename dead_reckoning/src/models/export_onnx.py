"""Export the trained PyTorch GRU model to ONNX format.

Contract:
- Input: 'imu_window', shape: [batch, 200, 6]
- Output: 'local_displacement', shape: [batch, 3]
- Batch dimension is dynamic.
- Sequence length is fixed at 200.
- Feature dimension is fixed at 6.
- The model represents the trained neural network only; normalization is handled externally.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import onnx
import onnxruntime as ort
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.gru_model import GRUDeadReckoning

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "gru_local_best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "gru_local.onnx"
WINDOW_SIZE = 200
FEATURE_DIM = 6
OUTPUT_DIM = 3
OPSET_VERSION = 17


def export_gru_to_onnx(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_path: str | Path = DEFAULT_OUTPUT,
    opset_version: int = OPSET_VERSION,
) -> Path:
    """Export the trained PyTorch GRU checkpoint to ONNX format."""
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate model and load trained state dictionary
    model = GRUDeadReckoning(
        input_size=FEATURE_DIM,
        hidden_size=64,
        num_layers=2,
        output_size=OUTPUT_DIM,
        dropout=0.2,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Create dummy input with shape [batch=1, 200, 6]
    dummy_input = torch.zeros((1, WINDOW_SIZE, FEATURE_DIM), dtype=torch.float32)

    # 3. Export to ONNX with dynamic batch dimension
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

    # 4. Verify exported model with ONNX checker
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # 5. Sanity check with ONNX Runtime session
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    diff = float(abs(torch_output - ort_outputs[0]).max())
    if diff > 1e-5:
        raise RuntimeError(
            f"Sanity check failed: ONNX max absolute diff {diff} exceeds tolerance 1e-5"
        )

    print(f"Exported ONNX model successfully to: {output_path}")
    print(f"Input shape: {session.get_inputs()[0].shape}")
    print(f"Output shape: {session.get_outputs()[0].shape}")
    print(f"Sanity check max diff: {diff:.2e}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained GRU to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path for exported ONNX model",
    )
    args = parser.parse_args()
    export_gru_to_onnx(args.checkpoint, args.output)
