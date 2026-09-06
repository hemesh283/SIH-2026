"""Reusable, deployment-ready inference engine for GRU Dead Reckoning.

This module encapsulates:
- Input shape and finite-value validation
- Feature normalization using fixed training statistics
- Model execution (PyTorch or ONNX Runtime backend)
- Output denormalization to physical displacement in meters: [dx, dy, dz]

Does not expose internal PyTorch or ONNX details to calling application code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gru_local_best.pt"
DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "gru_local.onnx"
DEFAULT_NORM_NPZ = PROJECT_ROOT / "models" / "gru_local_normalization.npz"
DEFAULT_NORM_JSON = PROJECT_ROOT / "models" / "normalization.json"

EXPECTED_WINDOW_SIZE = 200
EXPECTED_FEATURE_DIM = 6
EXPECTED_OUTPUT_DIM = 3
FEATURE_NAMES: list[str] = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]
OUTPUT_NAMES: list[str] = ["dx", "dy", "dz"]


class DeadReckoningInference:
    """Production inference wrapper for local 3D displacement estimation.

    Parameters:
        model_path: Path to PyTorch checkpoint (.pt) or ONNX model (.onnx).
                    Defaults to models/gru_local_best.pt (or gru_local.onnx if specified).
        normalization_path: Path to normalization stats (.npz or .json).
                            Defaults to models/gru_local_normalization.npz.
        device: Device for PyTorch inference ("cpu" or "cuda"). Defaults to "cpu".
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        normalization_path: str | Path | None = None,
        backend: str = "auto",
        device: str = "cpu",
        window_size: int = EXPECTED_WINDOW_SIZE,
    ) -> None:
        self.device_str = device
        self.window_size = window_size
        self.feature_dim = EXPECTED_FEATURE_DIM
        self.feature_names = list(FEATURE_NAMES)
        self.output_names = list(OUTPUT_NAMES)

        # 1. Resolve paths
        if model_path is None:
            if backend.lower() == "onnx" and DEFAULT_ONNX_PATH.is_file():
                self.model_path = Path(DEFAULT_ONNX_PATH)
            elif DEFAULT_MODEL_PATH.is_file():
                self.model_path = Path(DEFAULT_MODEL_PATH)
            elif DEFAULT_ONNX_PATH.is_file():
                self.model_path = Path(DEFAULT_ONNX_PATH)
            else:
                raise FileNotFoundError(
                    f"No model found at default locations: {DEFAULT_MODEL_PATH} or {DEFAULT_ONNX_PATH}"
                )
        else:
            self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        if normalization_path is None:
            if DEFAULT_NORM_NPZ.is_file():
                self.norm_path = Path(DEFAULT_NORM_NPZ)
            elif DEFAULT_NORM_JSON.is_file():
                self.norm_path = Path(DEFAULT_NORM_JSON)
            else:
                raise FileNotFoundError(
                    f"No normalization file found at {DEFAULT_NORM_NPZ} or {DEFAULT_NORM_JSON}"
                )
        else:
            self.norm_path = Path(normalization_path)

        if not self.norm_path.is_file():
            raise FileNotFoundError(f"Normalization file does not exist: {self.norm_path}")

        # 2. Load normalization parameters
        self._load_normalization(self.norm_path)

        # 3. Detect backend and load model
        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            self.backend = "onnx"
            self._init_onnx_backend()
        elif suffix in (".pt", ".pth", ".bin"):
            self.backend = "pytorch"
            self._init_pytorch_backend()
        else:
            raise ValueError(
                f"Unsupported model file extension '{suffix}'. Expected .pt, .pth, or .onnx"
            )

    def _load_normalization(self, path: Path) -> None:
        """Load normalization statistics strictly computed on training data."""
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.X_mean = np.array(data["mean"], dtype=np.float32)
            self.X_std = np.array(data["std"], dtype=np.float32)
            self.y_mean = np.array(data["target_mean"], dtype=np.float32)
            self.y_std = np.array(data["target_std"], dtype=np.float32)
        else:
            norm = np.load(path, allow_pickle=False)
            self.X_mean = norm["X_mean"].astype(np.float32)
            self.X_std = norm["X_std"].astype(np.float32)
            self.y_mean = norm["y_mean"].astype(np.float32)
            self.y_std = norm["y_std"].astype(np.float32)

        if self.X_mean.shape != (EXPECTED_FEATURE_DIM,):
            raise ValueError(f"X_mean must have shape ({EXPECTED_FEATURE_DIM},), got {self.X_mean.shape}")
        if self.X_std.shape != (EXPECTED_FEATURE_DIM,):
            raise ValueError(f"X_std must have shape ({EXPECTED_FEATURE_DIM},), got {self.X_std.shape}")
        if np.any(self.X_std == 0):
            raise ValueError("Normalization X_std contains zero standard deviation")
        if np.any(self.y_std == 0):
            raise ValueError("Normalization y_std contains zero standard deviation")

    def _init_pytorch_backend(self) -> None:
        """Initialize PyTorch model in eval mode."""
        import torch
        from src.models.gru_model import GRUDeadReckoning

        self.torch = torch
        self.device = torch.device(self.device_str)
        self.model = GRUDeadReckoning(
            input_size=EXPECTED_FEATURE_DIM,
            hidden_size=64,
            num_layers=2,
            output_size=EXPECTED_OUTPUT_DIM,
            dropout=0.2,
        )
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _init_onnx_backend(self) -> None:
        """Initialize ONNX Runtime inference session."""
        import onnxruntime as ort

        self.ort_session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.ort_session.get_inputs()[0].name

    def validate_input(self, window: Any) -> np.ndarray:
        """Validate input type, shape, and numerical validity.

        Accepts:
            window: array-like of shape (200, 6) or (1, 200, 6)
        Returns:
            Normalized 3D float32 numpy array of shape (1, 200, 6)
        """
        # Convert to numpy float32
        try:
            arr = np.asarray(window, dtype=np.float32)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Input window could not be converted to float32 array: {err}") from err

        # Check finiteness (no NaN or Inf)
        if not np.isfinite(arr).all():
            raise ValueError("Input window contains NaN or infinite values")

        # Standardize shape to (1, window_size, 6)
        if arr.ndim == 2:
            if arr.shape != (self.window_size, self.feature_dim):
                raise ValueError(
                    f"Expected 2D window of shape ({self.window_size}, {self.feature_dim}), "
                    f"got {arr.shape}"
                )
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            if arr.shape[1:] != (self.window_size, self.feature_dim):
                raise ValueError(
                    f"Expected 3D batch with window shape (*, {self.window_size}, {self.feature_dim}), "
                    f"got {arr.shape}"
                )
        else:
            raise ValueError(
                f"Input window must be 2D ({self.window_size}, {self.feature_dim}) or 3D (batch, {self.window_size}, {self.feature_dim}), got {arr.ndim}D array with shape {arr.shape}"
            )

        return arr

    def predict(self, window: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Run inference on a single 200x6 sequential IMU window.

        Args:
            window: Sequential IMU samples with shape (200, 6) or (1, 200, 6).
                    Columns MUST strictly follow: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
                    Acceleration units: g (linear acceleration, gravity removed).
                    Gyroscope units: rad/s.

        Returns:
            Displacement array [dx, dy, dz] in meters, shape (3,).
        """
        batch_arr = self.validate_input(window)
        if batch_arr.shape[0] != 1:
            raise ValueError(
                f"predict() expects a single window. For batch predictions, use predict_batch(). Got batch_size={batch_arr.shape[0]}"
            )

        # 1. Normalize input features using training statistics
        normalized_window = (batch_arr - self.X_mean) / self.X_std

        # 2. Run model forward pass
        if self.backend == "pytorch":
            with self.torch.no_grad():
                tensor_in = self.torch.from_numpy(normalized_window).to(self.device)
                norm_pred = self.model(tensor_in).cpu().numpy()
        else:
            norm_pred = self.ort_session.run(None, {self.input_name: normalized_window})[0]

        # 3. Denormalize target prediction back to original units (meters)
        pred_meters = norm_pred * self.y_std + self.y_mean

        # Return flat 1D array of shape (3,)
        return pred_meters[0]

    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """Run batch inference on multiple 200x6 sequential IMU windows.

        Args:
            windows: Batch of sequential IMU windows with shape (batch_size, 200, 6).

        Returns:
            Displacement array with shape (batch_size, 3) in meters.
        """
        batch_arr = self.validate_input(windows)

        # 1. Normalize input features
        normalized_windows = (batch_arr - self.X_mean) / self.X_std

        # 2. Run model
        if self.backend == "pytorch":
            with self.torch.no_grad():
                tensor_in = self.torch.from_numpy(normalized_windows).to(self.device)
                norm_pred = self.model(tensor_in).cpu().numpy()
        else:
            norm_pred = self.ort_session.run(None, {self.input_name: normalized_windows})[0]

        # 3. Denormalize to meters
        pred_meters = norm_pred * self.y_std + self.y_mean
        return pred_meters

    def __repr__(self) -> str:
        return (
            f"DeadReckoningInference(backend='{self.backend}', "
            f"model='{self.model_path.name}', "
            f"window_size={self.window_size}, "
            f"features={self.feature_dim})"
        )
