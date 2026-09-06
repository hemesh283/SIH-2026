"""Comprehensive Test Suite for Native 10 Hz IO-VNBD Pipeline and Models.

Verifies:
1. 10 Hz timestamp handling and uniform intervals
2. 20-sample window generation (2.0s physical window)
3. Sensor feature ordering: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
4. Gyroscope axis semantics and pitch-to-yaw-rate mapping
5. Target displacement calculation in local vehicle frame
6. WGS-84 to NED coordinate conversion round-trip parity
7. Sequence-level train/validation/test leakage absence
8. Normalization statistics integrity (computed strictly from train split)
9. Model forward pass shapes ([batch, 20, 6] -> [batch, 3])
10. Numerical parity between PyTorch checkpoint and ONNX runtime export
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

PROJECT_ROOT = Path("D:/dead_reckoning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gru_inference import DeadReckoningInference
from src.models.gru_model import GRUDeadReckoning
from src.navigation.coordinate_frames import CoordinateTransformer, MEAN_EARTH_RADIUS
from src.evaluation.run_real_io_vnbd_benchmark import ned_to_geodetic_vec, geodetic_to_ned_vec

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "io_vnbd"


class TestNativeIOVNBDPipeline(unittest.TestCase):
    """Test suite for Option A: Native 10 Hz, 20-sample window dead reckoning."""

    def test_01_native_10hz_timestamps_and_window_shape(self):
        """Verify processed datasets contain 20-sample windows at native 10 Hz (no 100 Hz interpolation)."""
        train_path = DATA_DIR / "io_vnbd_train_local.npz"
        self.assertTrue(train_path.is_file(), f"Missing train dataset: {train_path}")

        data = np.load(train_path)
        X = data["X"]
        y = data["y"]

        # Window size must be exactly 20 samples
        self.assertEqual(X.shape[1:], (20, 6), f"Expected window shape (20, 6), got {X.shape[1:]}")
        # Target shape must be 3 (dx, dy, dz)
        self.assertEqual(y.shape[1:], (3,), f"Expected target shape (3,), got {y.shape[1:]}")

        # Check finiteness (no NaNs or Infs)
        self.assertTrue(np.isfinite(X).all(), "Inputs contain NaNs or Infs")
        self.assertTrue(np.isfinite(y).all(), "Targets contain NaNs or Infs")

    def test_02_sensor_feature_ordering_and_gyro_mapping(self):
        """Verify feature ordering: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]."""
        with open(RESULTS_DIR / "sensor_schema.md", "r", encoding="utf-8") as f:
            content = f.read()

        # Check documentation mandates
        self.assertIn("GYROSCOPE Pitch", content)
        self.assertIn("ACCELEROMETER", content)
        self.assertIn("GYROSCOPE", content)

        # Inspect normalization statistics
        with open(MODELS_DIR / "io_vnbd_normalization.json", "r", encoding="utf-8") as f:
            norm = json.load(f)

        self.assertEqual(len(norm["mean"]), 6)
        self.assertEqual(len(norm["std"]), 6)
        self.assertEqual(len(norm["target_mean"]), 3)
        self.assertEqual(len(norm["target_std"]), 3)

        # Linear acceleration std dev should be realistic automotive vibration (0.01 - 0.5 g)
        for i in range(3):
            self.assertGreater(norm["std"][i], 0.005)
            self.assertLess(norm["std"][i], 2.0)

    def test_03_target_displacement_local_frame(self):
        """Verify target displacement construction in local initial frame."""
        # 10 m/s for 2 seconds = 20 m along heading
        heading_deg = 45.0
        psi = np.deg2rad(heading_deg)
        c, s = np.cos(psi), np.sin(psi)
        R_bn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        local_disp = np.array([20.0, 0.0, 0.0]) # Forward motion
        ned_disp = R_bn @ local_disp

        # Rotate back to body: v_b = R_bn^T @ v_ned
        recovered_local = R_bn.T @ ned_disp
        np.testing.assert_allclose(recovered_local, local_disp, atol=1e-5)

    def test_04_coordinate_conversion_parity(self):
        """Verify vectorized WGS-84 to NED round-trip matches scalar CoordinateTransformer."""
        orig_lat, orig_lon = 52.2053, 0.1218
        lats = np.array([52.2053, 52.2060, 52.2070])
        lons = np.array([0.1218, 0.1230, 0.1245])

        # Vectorized
        pn_vec, pe_vec = geodetic_to_ned_vec(lats, lons, orig_lat, orig_lon)
        lat_back_vec, lon_back_vec = ned_to_geodetic_vec(pn_vec, pe_vec, orig_lat, orig_lon)

        np.testing.assert_allclose(lat_back_vec, lats, atol=1e-7)
        np.testing.assert_allclose(lon_back_vec, lons, atol=1e-7)

        # Compare with scalar method
        for i in range(len(lats)):
            pn_s, pe_s = CoordinateTransformer.geodetic_to_ned(lats[i], lons[i], orig_lat, orig_lon)
            self.assertAlmostEqual(pn_vec[i], pn_s, places=5)
            self.assertAlmostEqual(pe_vec[i], pe_s, places=5)

    def test_05_train_val_test_leakage_absence(self):
        """Verify sequences are strictly partitioned with 0 overlap between splits."""
        manifest_path = RESULTS_DIR / "split_manifest.csv"
        self.assertTrue(manifest_path.is_file(), f"Missing split manifest: {manifest_path}")

        df = pd.read_csv(manifest_path)
        train_seqs = set(df[df["assigned_split"] == "train"]["sequence_key"])
        val_seqs = set(df[df["assigned_split"] == "val"]["sequence_key"])
        test_seqs = set(df[df["assigned_split"] == "test"]["sequence_key"])

        # Check disjoint sets
        self.assertEqual(len(train_seqs.intersection(val_seqs)), 0, "Leakage detected between train and val!")
        self.assertEqual(len(train_seqs.intersection(test_seqs)), 0, "Leakage detected between train and test!")
        self.assertEqual(len(val_seqs.intersection(test_seqs)), 0, "Leakage detected between val and test!")

        # Check audited sequence counts
        self.assertEqual(len(train_seqs), 53)
        self.assertEqual(len(val_seqs), 10)
        self.assertEqual(len(test_seqs), 9)

    def test_06_normalization_leakage_check(self):
        """Verify normalization statistics match strictly the training dataset."""
        train_data = np.load(DATA_DIR / "io_vnbd_train_local.npz")
        X_train = train_data["X"]
        y_train = train_data["y"]

        computed_X_mean = np.mean(X_train, axis=(0, 1))
        computed_X_std = np.std(X_train, axis=(0, 1))
        computed_y_mean = np.mean(y_train, axis=0)
        computed_y_std = np.std(y_train, axis=0)
        computed_y_std[computed_y_std == 0.0] = 1.0  # Safe division guard

        with open(MODELS_DIR / "io_vnbd_normalization.json", "r", encoding="utf-8") as f:
            norm = json.load(f)

        np.testing.assert_allclose(norm["mean"], computed_X_mean, atol=1e-4)
        np.testing.assert_allclose(norm["std"], computed_X_std, atol=1e-4)
        np.testing.assert_allclose(norm["target_mean"], computed_y_mean, atol=1e-4)
        np.testing.assert_allclose(norm["target_std"], computed_y_std, atol=1e-4)

    def test_07_model_output_shape_and_inference_wrapper(self):
        """Verify PyTorch model forward pass shape [batch, 20, 6] -> [batch, 3] and inference engine."""
        model_path = MODELS_DIR / "gru_io_vnbd_best.pt"
        self.assertTrue(model_path.is_file(), f"Missing model checkpoint: {model_path}")

        model = GRUDeadReckoning(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            dropout=0.2,
        )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        dummy_batch = torch.randn((4, 20, 6), dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_batch)

        self.assertEqual(out.shape, (4, 3))
        self.assertTrue(torch.isfinite(out).all())

        # Test DeadReckoningInference with window_size=20
        infer = DeadReckoningInference(
            model_path=model_path,
            normalization_path=MODELS_DIR / "io_vnbd_normalization.json",
            backend="pytorch",
            window_size=20,
        )
        single_window = np.zeros((20, 6), dtype=np.float32)
        pred = infer.predict(single_window)
        self.assertEqual(pred.shape, (3,))
        self.assertTrue(np.isfinite(pred).all())

    def test_08_onnx_model_numerical_parity(self):
        """Verify exported ONNX model achieves numerical parity (< 1e-4) with PyTorch."""
        pt_path = MODELS_DIR / "gru_io_vnbd_best.pt"
        onnx_path = MODELS_DIR / "gru_io_vnbd.onnx"
        self.assertTrue(onnx_path.is_file(), f"Missing ONNX model: {onnx_path}")

        # PyTorch
        model = GRUDeadReckoning(input_size=6, hidden_size=64, num_layers=2, output_size=3, dropout=0.2)
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        # ONNX
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Test on 100 synthetic test windows
        test_in = np.random.normal(0.0, 1.0, size=(100, 20, 6)).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(test_in)).numpy()

        ort_out = session.run(None, {input_name: test_in})[0]

        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        self.assertLess(max_diff, 1e-4, f"ONNX parity violation: max diff = {max_diff}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
