"""Comprehensive deterministic tests for DeadReckoningInference module."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gru_inference import DeadReckoningInference

MODEL_PT = PROJECT_ROOT / "models" / "gru_local_best.pt"
MODEL_ONNX = PROJECT_ROOT / "models" / "gru_local.onnx"
NORM_NPZ = PROJECT_ROOT / "models" / "gru_local_normalization.npz"
NORM_JSON = PROJECT_ROOT / "models" / "normalization.json"
VAL_DATA = PROJECT_ROOT / "data" / "processed" / "handheld_val_local.npz"


class TestDeadReckoningInference(unittest.TestCase):
    """Test suite for DeadReckoningInference contract and error handling."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.infer_pt = DeadReckoningInference(
            model_path=MODEL_PT,
            normalization_path=NORM_NPZ,
        )
        cls.infer_onnx = DeadReckoningInference(
            model_path=MODEL_ONNX,
            normalization_path=NORM_JSON,
        )

    def test_model_loading_and_attributes(self) -> None:
        """Verify model backends and contract metadata."""
        self.assertEqual(self.infer_pt.backend, "pytorch")
        self.assertEqual(self.infer_onnx.backend, "onnx")
        self.assertEqual(self.infer_pt.window_size, 200)
        self.assertEqual(self.infer_pt.feature_dim, 6)
        self.assertEqual(
            self.infer_pt.feature_names,
            ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        )
        self.assertEqual(self.infer_pt.output_names, ["dx", "dy", "dz"])

    def test_correct_2d_input(self) -> None:
        """Verify 2D input (200, 6) produces exactly 1D array of shape (3,)."""
        window = np.zeros((200, 6), dtype=np.float32)
        pred = self.infer_pt.predict(window)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.isfinite(pred).all())

    def test_correct_3d_input(self) -> None:
        """Verify 3D input (1, 200, 6) produces exactly 1D array of shape (3,)."""
        window = np.zeros((1, 200, 6), dtype=np.float32)
        pred = self.infer_pt.predict(window)

        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.isfinite(pred).all())

    def test_predict_batch(self) -> None:
        """Verify predict_batch handles multiple windows."""
        batch = np.zeros((4, 200, 6), dtype=np.float32)
        preds = self.infer_pt.predict_batch(batch)

        self.assertEqual(preds.shape, (4, 3))
        self.assertTrue(np.isfinite(preds).all())

    def test_incorrect_shape_rejections(self) -> None:
        """Verify explicit ValueError on mismatched window sizes or feature dimensions."""
        invalid_shapes = [
            (100, 6),       # Wrong timesteps
            (250, 6),       # Wrong timesteps
            (200, 5),       # Missing feature
            (200, 7),       # Extra feature
            (200,),         # 1D vector
            (1, 100, 6),    # Wrong 3D timesteps
            (2, 200, 6),    # Multiple windows passed to predict() instead of predict_batch()
            (1, 1, 200, 6), # 4D tensor
        ]
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                dummy = np.zeros(shape, dtype=np.float32)
                with self.assertRaises(ValueError):
                    self.infer_pt.predict(dummy)

    def test_nan_and_inf_rejections(self) -> None:
        """Verify explicit rejection of non-finite inputs."""
        # Test NaN
        window_nan = np.zeros((200, 6), dtype=np.float32)
        window_nan[50, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            self.infer_pt.predict(window_nan)

        # Test +Inf
        window_inf = np.zeros((200, 6), dtype=np.float32)
        window_inf[10, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            self.infer_pt.predict(window_inf)

        # Test -Inf
        window_ninf = np.zeros((200, 6), dtype=np.float32)
        window_ninf[199, 5] = -np.inf
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            self.infer_pt.predict(window_ninf)

    def test_deterministic_reproducibility(self) -> None:
        """Verify that identical inputs produce identical outputs across calls."""
        rng = np.random.default_rng(42)
        window = rng.standard_normal((200, 6)).astype(np.float32)

        pred1 = self.infer_pt.predict(window)
        pred2 = self.infer_pt.predict(window)
        np.testing.assert_array_equal(pred1, pred2)

    def test_pytorch_vs_onnx_backend_agreement(self) -> None:
        """Verify PyTorch and ONNX inference produce equivalent outputs."""
        rng = np.random.default_rng(123)
        window = rng.standard_normal((200, 6)).astype(np.float32)

        pred_pt = self.infer_pt.predict(window)
        pred_onnx = self.infer_onnx.predict(window)

        max_diff = float(np.abs(pred_pt - pred_onnx).max())
        self.assertLess(max_diff, 1e-5, f"Backend discrepancy {max_diff} exceeded 1e-5")

    def test_real_validation_data_sample(self) -> None:
        """Verify prediction on an actual held-out validation sample."""
        val_data = np.load(VAL_DATA)
        X_val = val_data["X"].astype(np.float32)
        y_val = val_data["y"].astype(np.float32)

        # Predict first sample
        sample_window = X_val[0]
        ground_truth = y_val[0]
        prediction = self.infer_pt.predict(sample_window)

        self.assertEqual(prediction.shape, (3,))
        # Error on single 2s window should be realistic (< 0.5 meters)
        error = np.linalg.norm(prediction - ground_truth)
        self.assertLess(error, 0.5, f"Displacement error {error:.4f}m exceeds sanity threshold")


if __name__ == "__main__":
    unittest.main(verbosity=2)
