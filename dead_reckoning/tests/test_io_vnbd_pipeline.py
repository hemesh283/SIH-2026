"""Automated Test Suite for IO-VNBD Pipeline, Navigation, Filter & Benchmark."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gru_inference import DeadReckoningInference
from src.navigation.coordinate_frames import CoordinateTransformer
from src.navigation.ekf import NavigationEKF
from src.navigation.nhc import NhcConstraint
from src.navigation.zupt import ZuptDetector
from src.preprocessing.io_vnbd_loader import IOVNBDLoader, generate_mock_io_vnbd_sequence
from src.preprocessing.resampler import StreamResampler
from src.evaluation.outage_simulator import OutageSimulator
from src.evaluation.metrics import NavigationMetrics
from src.evaluation.baseline_ladder import BaselineLadderEvaluator


class TestIOVNBDPipeline(unittest.TestCase):
    """Test suite covering IO-VNBD pipeline components, coordinate frames, EKF, and baselines."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.infer = DeadReckoningInference(backend="onnx")

    def test_mock_sequence_generation(self) -> None:
        """Verify synthetic IO-VNBD fixture adheres to schema and physics."""
        seq = generate_mock_io_vnbd_sequence(duration_s=60.0, sampling_rate_hz=10.0)
        self.assertEqual(len(seq.timestamps_s), 600)
        self.assertEqual(seq.acc_g.shape, (600, 3))
        self.assertEqual(seq.gyro_rads.shape, (600, 3))
        self.assertEqual(len(seq.gnss_lat), 600)
        self.assertEqual(len(seq.gt_lat), 600)
        self.assertTrue(np.isfinite(seq.acc_g).all())
        self.assertTrue(np.isfinite(seq.gyro_rads).all())
        self.assertTrue(seq.is_mock)

    def test_timestamp_resampling_and_window_creation(self) -> None:
        """Verify 100 Hz linear interpolation and exact 200-sample window generation."""
        seq = generate_mock_io_vnbd_sequence(duration_s=25.0, sampling_rate_hz=10.0)
        resampler = StreamResampler(target_rate_hz=100.0)
        stream = resampler.resample_and_window(
            t_raw=seq.timestamps_s,
            acc_g=seq.acc_g,
            gyro_rads=seq.gyro_rads,
            gnss_lat=seq.gnss_lat,
            gnss_lon=seq.gnss_lon,
            gnss_spd=seq.gnss_speed_mps,
            gnss_hdg=seq.gnss_heading_deg,
            gt_lat=seq.gt_lat,
            gt_lon=seq.gt_lon,
            gt_spd=seq.gt_speed_mps,
            gt_hdg=seq.gt_heading_deg,
            window_size=200,
            stride=100,
        )

        # 25 seconds at 100 Hz = ~2500 samples
        self.assertGreater(len(stream.t_100hz), 2400)
        # Verify uniform sampling interval dt = 0.01 s
        dt = np.diff(stream.t_100hz)
        np.testing.assert_allclose(dt, 0.01, atol=1e-5)

        # Check window dimensions
        self.assertEqual(stream.windows.shape[1:], (200, 6))
        self.assertGreater(len(stream.windows), 20)
        self.assertTrue(np.isfinite(stream.windows).all())

    def test_coordinate_frames_rotations_and_distances(self) -> None:
        """Verify heading DCM, phone alignment, and WGS-84 metric displacement conversions."""
        # 1. Heading 90 degrees (East)
        dcm_east = CoordinateTransformer.heading_to_dcm(90.0)
        forward_body = np.array([1.0, 0.0, 0.0])
        ned_vec = dcm_east @ forward_body
        # In NED: [North=0, East=1, Down=0]
        np.testing.assert_allclose(ned_vec, [0.0, 1.0, 0.0], atol=1e-6)

        # 2. Geodetic to NED and back
        orig_lat, orig_lon = 52.0, 0.0
        pn, pe = CoordinateTransformer.geodetic_to_ned(52.001, 0.001, orig_lat, orig_lon)
        lat_back, lon_back = CoordinateTransformer.ned_to_geodetic(pn, pe, orig_lat, orig_lon)
        self.assertAlmostEqual(lat_back, 52.001, places=6)
        self.assertAlmostEqual(lon_back, 0.001, places=6)

        # 3. Haversine distance
        dist = CoordinateTransformer.haversine_distance(52.0, 0.0, 52.001, 0.0)
        # 0.001 degree of latitude ~ 111.19 m
        self.assertAlmostEqual(dist, 111.19, delta=1.0)

    def test_ekf_simplified_filter_propagation_and_joseph_update(self) -> None:
        """Verify 6-state EKF prediction, Joseph form covariance update, and GNSS correction."""
        ekf = NavigationEKF()
        ekf.initialize(0.0, 0.0, 0.0, 5.0, 0.0, 0.0)
        self.assertTrue(ekf.is_initialized)

        # Predict with 1s displacement [5.0, 0.0, 0.0]
        ekf.predict(np.array([5.0, 0.0, 0.0]), dt=1.0)
        self.assertAlmostEqual(ekf.position_ned[0], 5.0, places=4)
        self.assertAlmostEqual(ekf.velocity_ned[0], 5.0, places=4)

        # Update with GNSS measurement [5.2, 0.0, 0.0]
        cov_prior = ekf.P[0, 0]
        ekf.update_gnss_position(5.2, 0.0, 0.0)
        self.assertLess(ekf.P[0, 0], cov_prior, "Covariance should decrease after measurement update")
        self.assertTrue(ekf.position_ned[0] > 5.0 and ekf.position_ned[0] <= 5.2)

        # Ensure covariance remains symmetric
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-10)

    def test_zupt_detector_conditions(self) -> None:
        """Verify multi-signal stationary detection logic."""
        zupt = ZuptDetector()
        
        # Stationary: linear accel ~ 0, gyro ~ 0, gnss speed ~ 0
        acc_stat = np.random.normal(0.0, 0.01, size=(20, 3)).astype(np.float32)
        gyro_stat = np.random.normal(0.0, 0.005, size=(20, 3)).astype(np.float32)
        self.assertTrue(zupt.is_stationary(acc_stat, gyro_stat, gnss_speed_mps=0.1))

        # Dynamic motion: speed = 10 m/s
        self.assertFalse(zupt.is_stationary(acc_stat, gyro_stat, gnss_speed_mps=10.0))

        # High vibration/gyro
        gyro_moving = np.ones((20, 3), dtype=np.float32) * 0.2
        self.assertFalse(zupt.is_stationary(acc_stat, gyro_moving, gnss_speed_mps=0.1))

    def test_nhc_pseudo_measurement_matrices(self) -> None:
        """Verify NHC measurement matrix rotates properly with heading."""
        nhc = NhcConstraint()
        # Heading = 0 (North) -> Lateral is East (vE = 0), Vertical is Down (vD = 0)
        z, H, R = nhc.get_measurement_matrix(heading_deg=0.0)
        self.assertEqual(z.shape, (2,))
        self.assertEqual(H.shape, (2, 6))
        self.assertAlmostEqual(H[0, 3], 0.0, places=5) # -sin(0)
        self.assertAlmostEqual(H[0, 4], 1.0, places=5) # cos(0)
        self.assertAlmostEqual(H[1, 5], 1.0, places=5) # vD

    def test_outage_simulator_deterministic_intervals(self) -> None:
        """Verify outage simulation creates valid deterministic masks without leakage."""
        t = np.arange(0.0, 100.0, 0.01)
        sim = OutageSimulator(outage_durations_s=[30.0], random_seed=42)
        intervals = sim.plan_outages(t, duration_s=30.0)
        self.assertEqual(len(intervals), 1)

        iv = intervals[0]
        self.assertAlmostEqual(iv.duration_s, 30.0)
        self.assertGreater(iv.start_time_s, 10.0)
        self.assertLess(iv.end_time_s, 95.0)

        mask = sim.create_gnss_mask(t, iv)
        self.assertEqual(len(mask), len(t))
        # Inside outage, mask must be False
        self.assertFalse(mask[iv.start_idx])
        self.assertFalse(mask[iv.end_idx - 1])
        # Outside outage, mask must be True
        self.assertTrue(mask[iv.start_idx - 1])
        self.assertTrue(mask[iv.end_idx + 1])

    def test_navigation_metrics_and_drift_calculation(self) -> None:
        """Verify position RMSE, endpoint error, and drift percentage calculations."""
        t = np.arange(0.0, 50.0, 0.01) # 5000 samples
        orig_lat, orig_lon = 52.0, 0.0

        # Ground truth: constant speed 10 m/s heading North (0 deg)
        # Distance = 500 m
        gt_dist_m = 10.0 * 50.0
        gt_pn = np.linspace(0.0, gt_dist_m, len(t))
        gt_pe = np.zeros(len(t))
        gt_lat = np.zeros(len(t))
        gt_lon = np.zeros(len(t))
        for i in range(len(t)):
            gt_lat[i], gt_lon[i] = CoordinateTransformer.ned_to_geodetic(gt_pn[i], gt_pe[i], orig_lat, orig_lon)

        # Estimated: drifting by 10 meters East at the end
        est_pn = gt_pn.copy()
        est_pe = np.linspace(0.0, 10.0, len(t))
        est_lat = np.zeros(len(t))
        est_lon = np.zeros(len(t))
        for i in range(len(t)):
            est_lat[i], est_lon[i] = CoordinateTransformer.ned_to_geodetic(est_pn[i], est_pe[i], orig_lat, orig_lon)

        result = NavigationMetrics.evaluate_outage(
            est_lat=est_lat,
            est_lon=est_lon,
            gt_lat=gt_lat,
            gt_lon=gt_lon,
            outage_start_idx=0,
            outage_end_idx=len(t),
            outage_duration_s=50.0,
            baseline_name="TEST_BASELINE",
            timestamps_s=t,
        )

        self.assertAlmostEqual(result.endpoint_error_m, 10.0, delta=0.5)
        # Distance ~ 500 m, endpoint error ~ 10 m -> Drift % ~ 2.0%
        self.assertAlmostEqual(result.drift_percent_endpoint, 2.0, delta=0.2)
        self.assertAlmostEqual(result.distance_travelled_m, 500.0, delta=5.0)

    def test_baseline_ladder_integration(self) -> None:
        """Verify baseline ladder executes all 7 configurations without error."""
        seq = generate_mock_io_vnbd_sequence(duration_s=30.0, sampling_rate_hz=10.0)
        resampler = StreamResampler(target_rate_hz=100.0)
        stream = resampler.resample_and_window(
            t_raw=seq.timestamps_s,
            acc_g=seq.acc_g,
            gyro_rads=seq.gyro_rads,
            gnss_lat=seq.gnss_lat,
            gnss_lon=seq.gnss_lon,
            gnss_spd=seq.gnss_speed_mps,
            gnss_hdg=seq.gnss_heading_deg,
            gt_lat=seq.gt_lat,
            gt_lon=seq.gt_lon,
            gt_spd=seq.gt_speed_mps,
            gt_hdg=seq.gt_heading_deg,
            window_size=200,
            stride=100,
        )

        outage_sim = OutageSimulator(outage_durations_s=[10.0])
        intervals = outage_sim.plan_outages(stream.t_100hz, duration_s=10.0)
        self.assertTrue(len(intervals) > 0)
        mask = outage_sim.create_gnss_mask(stream.t_100hz, intervals[0])

        ladder = BaselineLadderEvaluator(inference_engine=self.infer)
        solutions = ladder.run_all_baselines(stream, mask)

        expected_baselines = [
            "BASELINE 1: Pure INS",
            "BASELINE 2: INS + EKF",
            "BASELINE 3: ML Only",
            "BASELINE 4: ML + INS",
            "BASELINE 5: ML + INS + EKF",
            "BASELINE 6: ML + INS + EKF + NHC",
            "BASELINE 7: ML + INS + EKF + NHC + ZUPT",
        ]

        self.assertEqual(len(solutions), 7)
        for b in expected_baselines:
            self.assertIn(b, solutions)
            sol = solutions[b]
            self.assertEqual(len(sol.lat), len(stream.t_100hz))
            self.assertTrue(np.isfinite(sol.lat).all())
            self.assertTrue(np.isfinite(sol.lon).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
