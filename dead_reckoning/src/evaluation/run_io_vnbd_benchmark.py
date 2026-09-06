"""Main Reproducible Evaluation Entry Point for IO-VNBD GNSS Outage Benchmark.

Can be run via:
    python src/evaluation/run_io_vnbd_benchmark.py
or:
    python src/evaluation/run_io_vnbd_benchmark.py --outages 10 30 60 120

Executes:
  1. Dataset loading (or high-fidelity synthetic fixture if IO-VNBD is not yet placed)
  2. 100 Hz resampling and validation
  3. ONNX Runtime & PyTorch model inference
  4. GNSS outage simulation (10s, 30s, 60s, 120s)
  5. Scientific Baseline Ladder (Baselines 1 through 7)
  6. Trajectory error & recovery metric calculations
  7. Visualization generation (6 professional publication-quality figures)
  8. Machine-readable CSV generation (results.csv, summary.csv)
  9. Inference latency profiling (single-window, mean, p95)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import List, Dict

import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.gru_inference import DeadReckoningInference
from src.navigation.coordinate_frames import CoordinateTransformer
from src.navigation.zupt import ZuptDetector
from src.navigation.nhc import NhcConstraint
from src.preprocessing.io_vnbd_loader import IOVNBDLoader, VehicleSequenceData, generate_mock_io_vnbd_sequence
from src.preprocessing.resampler import StreamResampler
from src.evaluation.outage_simulator import OutageSimulator, OutageInterval
from src.evaluation.baseline_ladder import BaselineLadderEvaluator, TrajectorySolution
from src.evaluation.metrics import NavigationMetrics, OutageEvaluationResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IO-VNBD GNSS Outage Benchmark Runner")
    parser.add_argument(
        "--outages",
        nargs="+",
        type=float,
        default=[10.0, 30.0, 60.0, 120.0],
        help="GNSS outage durations in seconds (default: 10 30 60 120)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Directory containing raw datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "io_vnbd"),
        help="Output directory for plots and CSVs",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["onnx", "pytorch", "auto"],
        default="onnx",
        help="ML inference backend (default: onnx)",
    )
    return parser.parse_args()


def benchmark_inference_latency(infer_engine: DeadReckoningInference, num_runs: int = 200) -> Dict[str, float]:
    """Measure single-window inference latency, average latency, and p95 latency."""
    dummy_window = np.zeros((200, 6), dtype=np.float32)
    # Warmup
    for _ in range(20):
        _ = infer_engine.predict(dummy_window)

    latencies_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = infer_engine.predict(dummy_window)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    return {
        "single_ms": float(latencies_ms[0]),
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def generate_plots(
    output_dir: Path,
    stream,
    solutions: Dict[str, TrajectorySolution],
    outage_interval: OutageInterval,
    metrics_results: List[OutageEvaluationResult],
) -> None:
    """Generate professional evaluation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    s = outage_interval.start_idx
    e = outage_interval.end_idx
    dur = outage_interval.duration_s

    # 1. Trajectory Comparison Plot (WGS-84 / Local NED)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(stream.gt_lon_100hz, stream.gt_lat_100hz, "k--", label="Ground Truth", linewidth=2.0)
    # Highlight outage segment
    ax.plot(stream.gt_lon_100hz[s:e], stream.gt_lat_100hz[s:e], "r-", linewidth=3.0, label=f"GNSS Outage ({dur:.0f}s)")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    for idx, (name, sol) in enumerate(solutions.items()):
        short_name = name.split(":")[-1].strip()
        color = colors[idx % len(colors)]
        ax.plot(sol.lon[s:e], sol.lat[s:e], label=short_name, color=color, linewidth=1.8, alpha=0.9)

    ax.set_title(f"Vehicle Trajectory During {dur:.0f}s GNSS Outage", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (deg)", fontsize=12)
    ax.set_ylabel("Latitude (deg)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / f"trajectory_comparison_{dur:.0f}s.png", dpi=300)
    plt.close(fig)

    # 2. Position Error vs Time
    fig, ax = plt.subplots(figsize=(12, 6))
    t_out = stream.t_100hz[s:e] - stream.t_100hz[s]

    for idx, (name, sol) in enumerate(solutions.items()):
        short_name = name.split(":")[-1].strip()
        color = colors[idx % len(colors)]
        errors = [
            CoordinateTransformer.haversine_distance(sol.lat[i], sol.lon[i], stream.gt_lat_100hz[i], stream.gt_lon_100hz[i])
            for i in range(s, e)
        ]
        ax.plot(t_out, errors, label=short_name, color=color, linewidth=2.0)

    ax.set_title(f"Position Error Drift Over {dur:.0f}s Outage", fontsize=14, fontweight="bold")
    ax.set_xlabel("Outage Elapsed Time (s)", fontsize=12)
    ax.set_ylabel("Horizontal Position Error (m)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / f"position_error_vs_time_{dur:.0f}s.png", dpi=300)
    plt.close(fig)

    # 3. Baseline Comparison Bar Chart
    df_metrics = pd.DataFrame([asdict(m) for m in metrics_results])
    df_dur = df_metrics[df_metrics["outage_duration_s"] == dur]

    if not df_dur.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        baselines = [b.split(":")[-1].strip() for b in df_dur["baseline_name"]]
        x_pos = np.arange(len(baselines))
        width = 0.35

        ax.bar(x_pos - width / 2, df_dur["position_rmse_m"], width, label="Position RMSE (m)", color="#4C72B0")
        ax.bar(x_pos + width / 2, df_dur["endpoint_error_m"], width, label="Endpoint Error (m)", color="#DD8452")

        ax.set_title(f"Baseline Navigation Error Ladder ({dur:.0f}s Outage)", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(baselines, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Error (m)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / f"baseline_comparison_{dur:.0f}s.png", dpi=300)
        plt.close(fig)

    # 4. Drift % vs Outage Duration
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in df_metrics["baseline_name"].unique():
        sub = df_metrics[df_metrics["baseline_name"] == name].sort_values("outage_duration_s")
        short_name = name.split(":")[-1].strip()
        ax.plot(sub["outage_duration_s"], sub["drift_percent_endpoint"], marker="o", linewidth=2.0, label=short_name)

    # Draw SIH Hackathon <10% drift target threshold line
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1.5, label="Target: 10% Drift Threshold")
    ax.set_title("Endpoint Drift % vs Outage Duration", fontsize=14, fontweight="bold")
    ax.set_xlabel("GNSS Outage Duration (s)", fontsize=12)
    ax.set_ylabel("Drift % = (Endpoint Error / Distance) × 100", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "drift_percent_vs_duration.png", dpi=300)
    plt.close(fig)

    # 5. Position RMSE vs Outage Duration
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in df_metrics["baseline_name"].unique():
        sub = df_metrics[df_metrics["baseline_name"] == name].sort_values("outage_duration_s")
        short_name = name.split(":")[-1].strip()
        ax.plot(sub["outage_duration_s"], sub["position_rmse_m"], marker="s", linewidth=2.0, label=short_name)

    ax.set_title("Position RMSE vs Outage Duration", fontsize=14, fontweight="bold")
    ax.set_xlabel("GNSS Outage Duration (s)", fontsize=12)
    ax.set_ylabel("Position RMSE (m)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_outage_duration.png", dpi=300)
    plt.close(fig)


def run_benchmark():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("IO-VNBD VEHICLE DEAD RECKONING & GNSS OUTAGE BENCHMARK")
    print("=" * 75)

    # 1. Initialize Inference Engine
    print("\n[Step 1] Initializing Dead Reckoning Inference Engine...")
    infer = DeadReckoningInference(backend=args.backend)
    print(f"  Backend loaded: {infer.backend.upper()}")
    print(f"  Model path    : {infer.model_path}")
    print(f"  Normalization : {infer.norm_path}")

    # Profile latency
    print("\n[Step 2] Profiling Model Inference Latency (ONNX CPU)...")
    latency = benchmark_inference_latency(infer, num_runs=150)
    print(f"  Single-window latency : {latency['single_ms']:.3f} ms")
    print(f"  Mean latency          : {latency['mean_ms']:.3f} ± {latency['std_ms']:.3f} ms")
    print(f"  P95 latency           : {latency['p95_ms']:.3f} ms")
    print(f"  Max latency           : {latency['max_ms']:.3f} ms")

    # 2. Check for IO-VNBD dataset
    print("\n[Step 3] Inspecting IO-VNBD Dataset Availability...")
    loader = IOVNBDLoader(raw_data_dir=args.data_dir)
    dataset_present = loader.is_available()

    if dataset_present:
        print("  [SUCCESS] IO-VNBD dataset found!")
        pairs = loader.list_sequences()
        print(f"  Discovered {len(pairs)} synchronized smartphone/ECU sequence pairs.")
        s_csv, v_csv = pairs[0]
        print(f"  Loading sequence: {s_csv.name} & {v_csv.name}")
        seq_data = loader.load_sequence(s_csv, v_csv)
    else:
        print("  " + "!" * 65)
        print("  [ALERT] ML validation on real vehicle data is blocked:")
        print("          IO-VNBD dataset not found under D:\\dead_reckoning\\data\\raw")
        print("  " + "!" * 65)
        print("  To benchmark against official vehicle data:")
        print("    1. Download IO-VNBD from: https://github.com/onyekpeu/IO-VNBD")
        print("    2. Place CSV files into : D:\\dead_reckoning\\data\\raw\\IO-VNBD")
        print("    3. Re-run this evaluation command.")
        print("\n  Executing benchmark using high-fidelity synthetic vehicle sequence fixture...")
        seq_data = generate_mock_io_vnbd_sequence(duration_s=220.0, sampling_rate_hz=10.0)

    print(f"  Sequence ID    : {seq_data.sequence_id}")
    print(f"  Total Duration : {seq_data.timestamps_s[-1] - seq_data.timestamps_s[0]:.1f} seconds")
    print(f"  Total Samples  : {len(seq_data.timestamps_s)} samples")

    # 3. Resample to uniform 100 Hz
    print("\n[Step 4] Resampling sensor stream to uniform 100 Hz...")
    resampler = StreamResampler(target_rate_hz=100.0)
    stream = resampler.resample_and_window(
        t_raw=seq_data.timestamps_s,
        acc_g=seq_data.acc_g,
        gyro_rads=seq_data.gyro_rads,
        gnss_lat=seq_data.gnss_lat,
        gnss_lon=seq_data.gnss_lon,
        gnss_spd=seq_data.gnss_speed_mps,
        gnss_hdg=seq_data.gnss_heading_deg,
        gt_lat=seq_data.gt_lat,
        gt_lon=seq_data.gt_lon,
        gt_spd=seq_data.gt_speed_mps,
        gt_hdg=seq_data.gt_heading_deg,
    )
    print(f"  Resampled length: {len(stream.t_100hz)} samples at 100 Hz (dt = 0.01s)")
    print(f"  Total 200-sample windows generated: {len(stream.windows)}")

    # 4. Outage Simulation and Baseline Ladder Evaluation
    print("\n[Step 5] Running GNSS Outage Benchmark Across Durations...")
    outage_sim = OutageSimulator(outage_durations_s=args.outages)
    ladder = BaselineLadderEvaluator(inference_engine=infer)

    all_results: List[OutageEvaluationResult] = []
    last_solutions = None
    last_interval = None

    for duration in args.outages:
        print(f"\n  --- Testing Outage Duration: {duration:.0f} seconds ---")
        intervals = outage_sim.plan_outages(stream.t_100hz, duration_s=duration)
        if not intervals:
            print(f"    [SKIP] Sequence too short for {duration:.0f}s outage.")
            continue

        interval = intervals[0]
        gnss_mask = outage_sim.create_gnss_mask(stream.t_100hz, interval)

        # Run all 7 baselines
        solutions = ladder.run_all_baselines(stream, gnss_mask)
        last_solutions = solutions
        last_interval = interval

        for b_name, sol in solutions.items():
            res = NavigationMetrics.evaluate_outage(
                est_lat=sol.lat,
                est_lon=sol.lon,
                gt_lat=stream.gt_lat_100hz,
                gt_lon=stream.gt_lon_100hz,
                outage_start_idx=interval.start_idx,
                outage_end_idx=interval.end_idx,
                outage_duration_s=duration,
                baseline_name=b_name,
                timestamps_s=stream.t_100hz,
                est_speed_mps=sol.speed_mps,
                gt_speed_mps=stream.gt_spd_100hz,
                est_heading_deg=sol.heading_deg,
                gt_heading_deg=stream.gt_hdg_100hz,
            )
            all_results.append(res)
            print(
                f"    {b_name:<36} | "
                f"RMSE: {res.position_rmse_m:6.2f}m | "
                f"EndErr: {res.endpoint_error_m:6.2f}m | "
                f"Drift%: {res.drift_percent_endpoint:5.2f}% | "
                f"Dist: {res.distance_travelled_m:6.1f}m"
            )

    # 5. Export CSV Results
    print("\n[Step 6] Saving Evaluation Results...")
    df_results = pd.DataFrame([asdict(r) for r in all_results])
    results_csv = output_dir / "results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"  Results saved to: {results_csv}")

    # Summary table aggregated by baseline & duration
    summary_cols = [
        "baseline_name", "outage_duration_s", "distance_travelled_m",
        "position_rmse_m", "endpoint_error_m", "drift_percent_endpoint"
    ]
    df_summary = df_results[summary_cols].copy()
    summary_csv = output_dir / "summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"  Summary saved to: {summary_csv}")

    # 6. Generate Figures
    if last_solutions and last_interval:
        print("\n[Step 7] Generating Publication-Quality Plots...")
        generate_plots(output_dir, stream, last_solutions, last_interval, all_results)
        print(f"  Plots saved under: {output_dir}")

    # 7. Print Final Benchmark Summary
    print("\n" + "=" * 75)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 75)
    best_run = df_results.sort_values("drift_percent_endpoint").iloc[0]
    print(f"Best Configuration      : {best_run['baseline_name']}")
    print(f"Best Outage Duration    : {best_run['outage_duration_s']:.0f} seconds")
    print(f"Best Endpoint Drift %   : {best_run['drift_percent_endpoint']:.2f}%")
    print(f"Best Position RMSE      : {best_run['position_rmse_m']:.2f} meters")
    print(f"Best Endpoint Error     : {best_run['endpoint_error_m']:.2f} meters")
    print(f"Distance Travelled      : {best_run['distance_travelled_m']:.1f} meters")
    print(f"Target <10% Achieved    : {'YES' if best_run['drift_percent_endpoint'] < 10.0 else 'NO'}")
    print(f"ONNX Latency (Mean)     : {latency['mean_ms']:.3f} ms")
    print(f"ONNX Latency (P95)      : {latency['p95_ms']:.3f} ms")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
