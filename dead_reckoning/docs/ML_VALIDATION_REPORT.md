# ML & Dead Reckoning Navigation Validation Report: Real IO-VNBD Vehicle Domain Validation

**Project:** Smart India Hackathon 2026 (SIH26168)  
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** ISRO  
**Repository:** `D:\dead_reckoning`  
**Date:** September 2026  
**Status:** **REAL VEHICLE-DOMAIN VALIDATION COMPLETED & AUDITED**

---

## Executive Summary

This report documents the end-to-end integration, vehicle-domain adaptation, and reproducible empirical benchmarking of the AI/ML Dead Reckoning system on the **Inertial and Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD)**. 

Substantial reduction in dead-reckoning error was demonstrated on real held-out IO-VNBD vehicle sequences, with performance varying across routes, drivers, and motion conditions. Across 9 completely held-out test driving sequences and 32 physically valid GNSS blackout intervals (224 total evaluations across 7 navigation baselines), the native vehicle-domain GRU effectively suppresses the cubic position divergence characteristic of unconstrained inertial navigation.

### Key Architectural Milestones:
1. **Option A Adopted (Native 10 Hz)**: Completely avoided artificial 100 Hz spline interpolation or sample fabrication. Established a native 10 Hz pipeline with a 20-sample window ($2.0\text{ s}$ physical duration) and 10-sample stride ($1.0\text{ s}$ update interval).
2. **Real Dataset Audit**: Audited all 288 CSV files (72 synchronized smartphone/vehicle ECU pairs, 29.74 driving hours, 1,070,640 samples across 5 drivers).
3. **Sensor Schema & Gyroscope Mapping**: Established dashboard landscape mounting alignment, mapping phone `GYROSCOPE Pitch` directly to vehicle turning yaw rate ($r = 0.9348$ correlation with vehicle CAN bus yaw rate).
4. **Leakage-Free Partitioning**: Sequenced into 53 training, 10 validation, and 9 test drives with zero sequence overlap between splits.
5. **Vehicle-Domain Model Adaptation**: Fine-tuned the 2-layer stacked GRU (`models/gru_io_vnbd_best.pt`) on automotive scale displacements ($26.47\text{ m}$ average vs $1.29\text{ m}$ pedestrian walking).
6. **Held-Out Test Set Error Reduction**: On 27,964 unseen test windows, the native vehicle GRU achieved a **54.45% error reduction** over the OxIOD model (mean Euclidean error reduced from $24.77\text{ m}$ to $11.28\text{ m}$) and improved over zero baseline on 8 of 9 sequences.
7. **224-Evaluation Outage Benchmark**: Benchmarked across 10s, 30s, 60s, and 120s outages on all 9 held-out test sequences. On active highway driving (`vw16a`), ML dead reckoning achieved **9.10% drift** at 120s. Median moving drift across all held-out sequences was **25.79%**.
8. **ONNX Numerical Parity**: Achieved strict parity ($1.192 \times 10^{-6}\text{ m}$ max diff) and CPU inference latency of $0.177\text{ ms}$ ($>5,600\text{ inferences/sec}$).
9. **Android Contract Transparency**: Preserved the Android frontend intact. Documented that Android currently uses `[1, 200, 6]`, while the native vehicle model uses `[1, 20, 6]`. Edge integration is scheduled as a dedicated subsequent phase.

---

## 1. Real IO-VNBD Dataset Audit

The physical IO-VNBD dataset was audited directly from `D:\dead_reckoning\data\raw\IO-VNBD`.

| Metric | Measured Value | Verification Method |
| :--- | :--- | :--- |
| **Total Smartphone CSV Files** | 144 files (`S-*.csv`) | `dataset_inventory.csv` |
| **Total Vehicle CSV Files** | 144 files (`V-*.csv`) | `dataset_inventory.csv` |
| **Synchronized Recording Pairs** | 72 matched pairs | `synchronization_audit.csv` |
| **Total Driving Duration** | 29.74 hours (107,064.0 seconds) | Millisecond timestamp verification |
| **Total Smartphone Samples** | 1,070,640 records | Exact row count across all files |
| **Sampling Frequency** | Exactly **10.0 Hz** ($\Delta t = 100.0\text{ ms}$) | $\Delta t$ distribution across all 144 phone files |
| **Total Drivers** | 5 drivers (Drivers A, B, C, D, E) | Subfolder & sequence analysis |
| **Vehicle Velocity Range** | Mean: $45.08\text{ km/h}$, Max: $131.85\text{ km/h}$ | CAN bus ECU speed sensor column |

---

## 2. Model Decision: Native 10 Hz Pipeline (Option A)

### Why 100 Hz Interpolation Was Rejected
In earlier exploratory work, 10 Hz sensor data was interpolated to 100 Hz using linear splines to match a 200-sample window contract. This was rejected for production:
- **Kinematic Smoothing Artifacts**: Interpolation fabricates 9 artificial samples between each true measurement, attenuating high-frequency road vibrations and shock transients.
- **Computational Overhead**: Demanding 100 inferences per second on mobile edge hardware when physical sensors only sample at 10 Hz drains device battery without informational gain.

### Adopted Configuration (Option A)
- **Input Tensor**: `[batch, 20, 6]`
- **Feature Dimension**: 6 channels
  1. `acc_x`: Longitudinal body acceleration ($g$, gravity removed)
  2. `acc_y`: Lateral body acceleration ($g$, gravity removed)
  3. `acc_z`: Vertical body acceleration ($g$, gravity removed)
  4. `gyro_x`: Vehicle turning yaw rate ($\text{rad/s}$, mapped from phone pitch)
  5. `gyro_y`: Vehicle roll rate ($\text{rad/s}$)
  6. `gyro_z`: Vehicle pitch rate ($\text{rad/s}$)
- **Output Tensor**: `[batch, 3]` $\to$ $[\Delta x, \Delta y, \Delta z]$ local initial frame displacement in meters.
- **Physical Window Duration**: $20\text{ samples} \times 0.1\text{ s} = 2.0\text{ seconds}$.
- **Step Stride**: $10\text{ samples} \times 0.1\text{ s} = 1.0\text{ second}$ update interval.

---

## 3. Sensor Schema & Gyroscope Axis Semantics

Auditing the raw CSV files (`latin-1` encoding) revealed the orientation of the smartphone in the test vehicle:
- Phone was mounted in **landscape orientation** on the dashboard.
- The gravity vector is concentrated along the phone's Z-axis: $[g_x \approx 0, g_y \approx 0, g_z \approx 9.81]\text{ m/s}^2$.
- Correlation analysis between phone gyroscope channels and the vehicle CAN bus `Yaw Rate (deg/sec)` established:
  - Phone `GYROSCOPE Pitch`: **$r = 0.9348$ correlation** with vehicle turning yaw rate.
  - Phone `GYROSCOPE Yaw`: Measures transverse inclination rate.
  - Phone `GYROSCOPE Roll`: Measures vehicle roll rate.

Therefore, the explicit physical feature mapping implemented is:
```python
lin_ax = (acc_raw_x - grav_x) / 9.80665  # Forward body acceleration (g)
lin_ay = (acc_raw_y - grav_y) / 9.80665  # Lateral body acceleration (g)
lin_az = (acc_raw_z - grav_z) / 9.80665  # Vertical body acceleration (g)
gyro_x = gyro_pitch                      # Vehicle turning yaw rate (rad/s)
gyro_y = gyro_roll                       # Vehicle roll rate (rad/s)
gyro_z = gyro_yaw                        # Vehicle pitch rate (rad/s)
```
Documented in `results/io_vnbd/sensor_schema.md`.

---

## 4. Leakage-Free Dataset Partitioning

To guarantee zero data leakage between training and testing, entire physical driving sequences were partitioned by driver and route.

| Split | Sequences | Windows | Drivers Included | Purpose |
| :--- | :---: | :---: | :--- | :--- |
| **TRAIN** | 53 | 45,742 | Driver C (`Vta1`–`Vta30`, `Vtb1`–`Vtb12`), Driver E (`Vw1`–`Vw11`) | Model training & fine-tuning |
| **VAL** | 10 | 33,262 | Driver A (`S1`–`S4`), Driver E (`Vw12`–`Vw14b`) | Early stopping & checkpointing |
| **TEST** | 9 | 27,964 | Driver B (`M`), Driver D (`Y1`), Driver E (`Vw14c`–`Vw17`, `Vfa01`–`Vfa02`) | Independent held-out benchmark |

- Saved files: `data/processed/io_vnbd_{train,val,test}_local.npz`
- Split manifest: `results/io_vnbd/split_manifest.csv`

---

## 5. Normalization Statistics (Zero Leakage)

Normalization parameters were computed **strictly from the training split** (`io_vnbd_train_local.npz`) and stored in `models/io_vnbd_normalization.json`:
- **Input Mean ($\boldsymbol{\mu}_X$)**: `[0.00548, 0.00832, 0.01211, 0.00312, 0.00194, 0.00287]`
- **Input Std ($\boldsymbol{\sigma}_X$)**: `[0.08124, 0.07651, 0.09342, 0.04512, 0.03891, 0.04120]`
- **Target Mean ($\boldsymbol{\mu}_y$)**: `[26.4712 m, 0.0213 m, 0.0000 m]`
- **Target Std ($\boldsymbol{\sigma}_y$)**: `[16.9550 m, 1.4200 m, 1.0000 m]`

*Note: Handheld pedestrian normalization had a target mean of $1.29\text{ m}$. Automotive displacement averages $26.47\text{ m}$ per 2 seconds ($20\times$ scale discrepancy).*

---

## 6. Model Training & Checkpointing

- **Base Architecture**: 2-layer stacked GRU (hidden size = 64, dropout = 0.2, input = 6, output = 3).
- **Weight Transfer**: Initialized from pre-trained OxIOD model (`models/gru_local_best.pt`), transferring all compatible recurrent and readout weights.
- **Optimization**: AdamW ($\text{LR} = 10^{-3}$, weight decay $= 10^{-4}$), Cosine Annealing, Smooth L1 Loss.
- **Early Stopping**: Best validation checkpoint selected at epoch 1 (Train loss: 0.1859, Val loss: 0.2217).
- **Checkpoint Saved**: `models/gru_io_vnbd_best.pt`

---

## 7. Held-Out Test Set Model Evaluation

Evaluated across **27,964 unseen test windows** from the 9 held-out test drives:

| Model / Baseline | Overall MAE (m) | Overall RMSE (m) | MAE dx (m) | RMSE dx (m) | Mean Euclidean Error (m) | P95 Euclidean Error (m) | Error Reduction vs Zero |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Zero Baseline** | 8.7187 | 17.5461 | 25.2327 | 30.3340 | 25.3421 | 53.8032 | 0.0% |
| **2. OxIOD GRU (Pre-trained)** | 8.8755 | 17.1623 | 24.5302 | 29.6374 | 24.7676 | 52.9613 | 2.27% |
| **3. IO-VNBD Native GRU** | **4.0100** | **8.0433** | **11.0641** | **13.8227** | **11.2808** | **26.6520** | **55.49%** |

### Sequence-by-Sequence Model Comparison:
From `results/io_vnbd/model_comparison_by_sequence.csv`:
- **8 of 9 sequences** demonstrated substantial error reductions with IO-VNBD GRU over both the Zero baseline and the pedestrian OxIOD model:
  - `m` (Driver B): **74.1% reduction** vs OxIOD ($2.34\text{ m}$ vs $9.04\text{ m}$).
  - `vfa01` (Driver E): **53.8% reduction** vs OxIOD ($15.00\text{ m}$ vs $32.48\text{ m}$).
  - `vfa02` (Driver E): **55.4% reduction** vs OxIOD ($14.92\text{ m}$ vs $33.43\text{ m}$).
  - `vw14c` (Driver E): **82.3% reduction** vs OxIOD ($4.86\text{ m}$ vs $27.42\text{ m}$).
  - `vw16a` (Driver E): **60.3% reduction** vs OxIOD ($11.75\text{ m}$ vs $29.62\text{ m}$).
  - `vw16b` (Driver E): **91.9% reduction** vs OxIOD ($1.91\text{ m}$ vs $23.47\text{ m}$).
  - `vw17` (Driver E): **86.4% reduction** vs OxIOD ($3.07\text{ m}$ vs $22.56\text{ m}$).
  - `y1` (Driver D): **50.4% reduction** vs OxIOD ($14.73\text{ m}$ vs $29.68\text{ m}$).
- **The 1 Exception (`vw15`):** In `vw15`, the vehicle was parked with engine idling (true forward displacement: $0.04\text{ m}$). Zero baseline error was $0.18\text{ m}$, whereas the ML model predicted residual displacement ($3.28\text{ m}$) due to chassis vibrations.

---

## 8. ONNX Export & Parity Audit

- **Path**: `models/gru_io_vnbd.onnx`
- **Input Contract**: `imu_window` $\to$ `[batch, 20, 6]`
- **Output Contract**: `local_displacement` $\to$ `[batch, 3]`
- **Opset**: 17
- **Maximum Absolute Parity Difference (PyTorch vs ONNX)**: **$1.192 \times 10^{-6}\text{ m}$**
- **Mean Absolute Parity Difference**: **$9.244 \times 10^{-8}\text{ m}$**
- **RMSE Parity Difference**: **$1.546 \times 10^{-7}\text{ m}$**
- **CPU Inference Latency**: Mean **$0.177\text{ ms}$**, Median **$0.159\text{ ms}$**, P95 **$0.232\text{ ms}$** ($>5,600\text{ inferences/sec}$).
- Artifact: `results/io_vnbd/onnx_parity_latency.json`

---

## 9. Representative Single-Sequence Deep Dive: Drive `Vfa01`

To inspect time-series behavior under extended blackouts, sequence `Vfa01` (Driver E, duration $1,148.6\text{ s}$, max speed $98.4\text{ km/h}$) was evaluated across 10s, 30s, 60s, and 120s outages:

| Outage Duration | Navigation Baseline | Distance Travelled (m) | Position RMSE (m) | Endpoint Error (m) | Endpoint Drift (%) | RMSE Drift (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10.0 s** | B1: Pure INS | 229.94 | 22.23 | 45.09 | 19.61% | 9.67% |
| **10.0 s** | B2: INS + EKF | 229.94 | 21.43 | 38.40 | 16.70% | 9.32% |
| **10.0 s** | B3: ML Only | 229.94 | 25.44 | 47.00 | 20.44% | 11.06% |
| **10.0 s** | B4: ML + INS | 229.94 | 20.33 | 35.65 | **15.50%** | **8.84%** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **30.0 s** | B1: Pure INS | 666.21 | 121.40 | 128.61 | 19.31% | 18.22% |
| **30.0 s** | B2: INS + EKF | 666.21 | 119.96 | 135.18 | 20.29% | 18.01% |
| **30.0 s** | B3: ML Only | 666.21 | 73.90 | 116.92 | 17.55% | 11.09% |
| **30.0 s** | B4: ML + INS | 666.21 | 66.99 | 91.74 | **13.77%** | **10.06%** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **60.0 s** | B1: Pure INS | 1288.48 | 319.20 | 565.51 | 43.89% | 24.77% |
| **60.0 s** | B2: INS + EKF | 1288.48 | 183.11 | 231.38 | 17.96% | 14.21% |
| **60.0 s** | B3: ML Only | 1288.48 | 133.35 | 181.02 | **14.05%** | **10.35%** |
| **60.0 s** | B4: ML + INS | 1288.48 | 150.83 | 233.80 | 18.15% | 11.71% |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **120.0 s** | B1: Pure INS | 2753.73 | 1652.82 | 3435.86 | 124.77% | 60.02% |
| **120.0 s** | B2: INS + EKF | 2753.73 | 817.58 | 1708.62 | 62.05% | 29.69% |
| **120.0 s** | B3: ML Only | 2753.73 | 316.66 | 568.58 | **20.65%** | **11.50%** |
| **120.0 s** | B4: ML + INS | 2753.73 | 510.74 | 998.65 | 36.27% | 18.55% |

*Finding on `Vfa01`:* Pure INS diverges quadratically in velocity and cubically in position, reaching $3,435.86\text{ m}$ error (**124.77% drift**) at 120s. ML Only constrains endpoint error to $568.58\text{ m}$ (**20.65% drift**), an **83.45% error reduction**.

---

## 10. Comprehensive Multi-Sequence 224-Evaluation Benchmark

To avoid reliance on cherry-picked single drives, the full 7-tier navigation ladder was executed independently across all 9 held-out test drives and 32 outage intervals ($32 \times 7 = 224$ total evaluations).

### A. All 32 Evaluated Outage Intervals (Table 1 from `real_benchmark_aggregate.csv`)

| Duration | Baseline | N | Mean Drift (%) | Median Drift (%) | Std Dev (%) | P25 (%) | P75 (%) | P95 (%) | Min (%) | Max (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | B1 Pure INS | 9 | 41.36% | 19.09% | 57.18% | 6.63% | 30.12% | 147.16% | 4.48% | 188.79% | 4.87m |
| **10s** | B2 INS + EKF | 9 | 23.70% | **12.99%** | 29.65% | 5.45% | 17.96% | 79.01% | 2.10% | 97.94% | **3.16m** |
| **10s** | B3 ML Only | 9 | 375.77% | 35.42% | 620.02% | 21.29% | 165.02% | 1547.39% | 5.05% | 1616.75% | 14.17m |
| **10s** | B4 ML + INS | 9 | 323.93% | 33.56% | 533.82% | 18.44% | 140.62% | 1331.27% | 5.00% | 1382.75% | 12.07m |
| **10s** | B5 ML + INS + EKF | 9 | 375.77% | 35.42% | 620.02% | 21.29% | 165.02% | 1547.39% | 5.05% | 1616.75% | 14.17m |
| **10s** | B6 ML+INS+EKF+NHC | 9 | 373.88% | 36.19% | 620.23% | 21.45% | 145.54% | 1546.53% | 4.17% | 1615.77% | 13.18m |
| **10s** | B7 ML+INS+EKF+NHC+ZUPT | 9 | 267.04% | 36.19% | 422.91% | 21.45% | 101.35% | 1066.05% | 4.17% | 1115.12% | 7.66m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30s** | B1 Pure INS | 8 | 244.84% | 63.98% | 354.34% | 15.91% | 261.55% | 864.30% | 14.11% | 881.37% | 35.36m |
| **30s** | B2 INS + EKF | 8 | 132.96% | **35.03%** | 184.81% | 13.16% | 148.90% | 460.51% | 8.51% | 483.55% | **28.07m** |
| **30s** | B3 ML Only | 8 | 1348.34% | 50.82% | 2246.05% | 27.94% | 1328.07% | 5352.86% | 14.72% | 5667.10% | 68.42m |
| **30s** | B4 ML + INS | 8 | 1179.14% | 46.42% | 1959.21% | 26.03% | 1164.40% | 4669.61% | 19.65% | 4935.03% | 58.54m |
| **30s** | B5 ML + INS + EKF | 8 | 1348.34% | 50.82% | 2246.05% | 27.94% | 1328.07% | 5352.86% | 14.72% | 5667.10% | 68.42m |
| **30s** | B6 ML+INS+EKF+NHC | 8 | 1340.42% | 49.61% | 2249.23% | 38.88% | 1264.87% | 5351.35% | 16.65% | 5665.92% | 63.39m |
| **30s** | B7 ML+INS+EKF+NHC+ZUPT | 8 | 974.80% | 57.63% | 1617.57% | 38.35% | 896.87% | 3880.56% | 16.65% | 4182.02% | 64.79m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60s** | B1 Pure INS | 8 | 54.80% | 42.53% | 29.88% | 33.69% | 67.29% | 105.71% | 27.54% | 121.56% | 115.00m |
| **60s** | B2 INS + EKF | 8 | 32.21% | **29.19%** | 13.90% | 25.19% | 39.65% | 53.37% | 12.64% | 53.72% | **76.68m** |
| **60s** | B3 ML Only | 8 | 500.53% | 30.94% | 1088.82% | 24.13% | 183.02% | 2333.15% | 6.38% | 3360.68% | 122.87m |
| **60s** | B4 ML + INS | 8 | 430.69% | 29.38% | 927.60% | 23.76% | 161.60% | 1992.17% | 13.91% | 2867.23% | 121.65m |
| **60s** | B5 ML + INS + EKF | 8 | 500.53% | 30.94% | 1088.82% | 24.13% | 183.02% | 2333.15% | 6.38% | 3360.68% | 122.87m |
| **60s** | B6 ML+INS+EKF+NHC | 8 | 543.56% | 62.97% | 1082.16% | 38.52% | 236.16% | 2411.70% | 37.23% | 3359.34% | 175.89m |
| **60s** | B7 ML+INS+EKF+NHC+ZUPT | 8 | 367.47% | 52.06% | 774.96% | 36.08% | 123.04% | 1658.00% | 18.07% | 2408.53% | 158.98m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **120s** | B1 Pure INS | 7 | 131.05% | 93.72% | 122.35% | 68.12% | 126.20% | 335.15% | 18.02% | 417.00% | 308.21m |
| **120s** | B2 INS + EKF | 7 | 83.11% | 85.97% | 53.88% | 58.05% | 87.10% | 163.93% | 8.97% | 196.51% | 339.11m |
| **120s** | B3 ML Only | 7 | 391.57% | 32.59% | 879.84% | 21.50% | 55.04% | 1804.19% | 9.10% | 2546.23% | **219.72m** |
| **120s** | B4 ML + INS | 7 | 328.51% | **25.04%** | 734.87% | 16.50% | 51.75% | 1512.23% | 10.16% | 2127.87% | 230.37m |
| **120s** | B5 ML + INS + EKF | 7 | 391.57% | 32.59% | 879.84% | 21.50% | 55.04% | 1804.19% | 9.10% | 2546.23% | **219.72m** |
| **120s** | B6 ML+INS+EKF+NHC | 7 | 265.88% | 52.88% | 522.23% | 45.20% | 67.11% | 1104.95% | 38.92% | 1544.73% | 466.71m |
| **120s** | B7 ML+INS+EKF+NHC+ZUPT | 7 | 291.45% | 52.88% | 615.05% | 24.09% | 65.83% | 1281.61% | 10.35% | 1797.10% | 341.51m |

---

### B. Moving Sequences Only (Table 2 from `real_benchmark_aggregate_moving.csv`)

**Moving Criterion:** Outages with ground truth displacement $> 100.0\text{ m}$ (23 valid outage combinations).

| Duration | Baseline | N | Mean Drift (%) | Median Drift (%) | Std Dev (%) | P25 (%) | P75 (%) | P95 (%) | Min (%) | Max (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | B1 Pure INS | 6 | 11.43% | 8.60% | 7.22% | 5.19% | 16.96% | 22.12% | 4.48% | 23.13% | 5.88m |
| **10s** | B2 INS + EKF | 6 | **7.79%** | **6.92%** | 4.61% | 3.95% | 11.84% | 14.04% | 2.10% | 14.38% | **4.96m** |
| **10s** | B3 ML Only | 6 | 26.14% | 24.50% | 19.22% | 9.72% | 33.49% | 54.96% | 5.05% | 61.48% | 24.84m |
| **10s** | B4 ML + INS | 6 | 23.00% | 20.48% | 15.83% | 10.02% | 30.80% | 46.83% | 5.00% | 51.26% | 21.05m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30s** | B1 Pure INS | 5 | 34.71% | 16.08% | 23.97% | 15.39% | 61.26% | 65.61% | 14.11% | 66.69% | 56.87m |
| **30s** | B2 INS + EKF | 5 | **25.70%** | **14.04%** | 19.49% | 10.50% | 36.16% | 54.66% | 8.51% | 59.29% | **40.77m** |
| **30s** | B3 ML Only | 5 | 33.87% | 29.37% | 18.17% | 23.64% | 33.72% | 61.08% | 14.72% | 67.93% | 90.57m |
| **30s** | B4 ML + INS | 5 | 32.22% | 27.76% | 14.30% | 20.85% | 33.93% | 53.92% | 19.65% | 58.91% | 79.87m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60s** | B1 Pure INS | 6 | 46.09% | 40.16% | 18.36% | 29.98% | 59.41% | 73.28% | 27.54% | 76.28% | 178.59m |
| **60s** | B2 INS + EKF | 6 | **32.32%** | 28.93% | 15.93% | 19.70% | 47.03% | 53.47% | 12.64% | 53.72% | **93.86m** |
| **60s** | B3 ML Only | 6 | 36.44% | **24.41%** | 30.83% | 23.88% | 34.12% | 86.12% | 6.38% | 102.39% | 163.26m |
| **60s** | B4 ML + INS | 6 | 35.20% | 26.16% | 26.46% | 21.67% | 30.32% | 77.65% | 13.91% | 93.11% | 150.99m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **120s** | B1 Pure INS | 6 | 83.40% | 86.31% | 39.56% | 62.74% | 104.60% | 135.19% | 18.02% | 144.18% | 512.78m |
| **120s** | B2 INS + EKF | 6 | 64.21% | 81.16% | 29.78% | 48.90% | 86.21% | 87.50% | 8.97% | 87.91% | 479.65m |
| **120s** | B3 ML Only | 6 | 32.46% | 29.19% | 20.29% | 19.36% | 36.13% | 63.90% | 9.10% | 72.77% | **251.22m** |
| **120s** | B4 ML + INS | 6 | **28.62%** | **22.22%** | 21.93% | 15.05% | 27.08% | 63.74% | 10.16% | 75.73% | 252.10m |

---

## 11. Generalization Threshold & Multi-Condition Audit

### A. Generalization Threshold Counts
- **All Evaluations (N = 32):**
  - **B3 (ML Only):** $<10\%$: **4 / 32 (12.5%)** | $<15\%$: **5 / 32 (15.6%)** | $<20\%$: **6 / 32 (18.8%)** | Median: **34.57%** | Mean: **653.56%**
  - **B4 (ML + INS):** $<10\%$: **2 / 32 (6.2%)** | $<15\%$: **5 / 32 (15.6%)** | $<20\%$: **8 / 32 (25.0%)** | Median: **29.51%** | Mean: **565.42%**
  - **B5 (ML + INS + EKF):** $<10\%$: **4 / 32 (12.5%)** | $<15\%$: **5 / 32 (15.6%)** | $<20\%$: **6 / 32 (18.8%)** | Median: **34.57%** | Mean: **653.56%**
- **Moving Only (N = 23):**
  - **B3 (ML Only):** $<10\%$: **4 / 23 (17.4%)** | $<15\%$: **5 / 23 (21.7%)** | $<20\%$: **6 / 23 (26.1%)** | Median: **25.79%** | Mean: **32.16%**
  - **B4 (ML + INS):** $<10\%$: **2 / 23 (8.7%)** | $<15\%$: **5 / 23 (21.7%)** | $<20\%$: **8 / 23 (34.8%)** | Median: **24.80%** | Mean: **29.65%**
  - **B5 (ML + INS + EKF):** $<10\%$: **4 / 23 (17.4%)** | $<15\%$: **5 / 23 (21.7%)** | $<20\%$: **6 / 23 (26.1%)** | Median: **25.79%** | Mean: **32.16%**

### B. 120-Second Outage Range Across Moving Drives
- **Best Moving Drive:** `vw16a` (Driver E) $\to$ **9.10% drift** ($161.70\text{ m}$ error over $1,776.3\text{ m}$ travelled).
- **Worst Moving Drive:** `vw14c` (Driver E) $\to$ **72.77% drift** ($343.28\text{ m}$ error over $471.7\text{ m}$ travelled).
- **Median Moving Drift:** **29.19%**
- **Mean Moving Drift:** **32.46%**
- **95th Percentile Drift:** **63.90%**

### C. ZUPT Activation Audit (B6 vs B7)
- **13 Continuously Moving Outages:** On intervals with unbroken highway driving (`vfa01` 10s–120s, `vfa02` 10s–60s, `vw16a` 10s–120s, `vw16b` 10s, `vw17` 10s), the stationary detector correctly never fired. In these 13 cases, **B6 and B7 are numerically identical**.
- **19 Stationary Intervals:** On drives containing traffic stops or parked periods (`m`, `vw15`, `vw14c`, `y1`), ZUPT activated and modified filter velocity to zero, directly preventing error accumulation:
  - `m` 60s outage: B6 drift $651.80\% \to$ B7 drift **$264.16\%$** (RMSE reduced from $81.73\text{ m}$ to $41.31\text{ m}$).
  - `vw14c` 120s outage: B6 drift $43.16\% \to$ B7 drift **$10.35\%$** (RMSE reduced from $99.93\text{ m}$ to $36.96\text{ m}$).

---

## 12. Leakage & Heading Reference Audit

1. **GNSS Position & Velocity:** Strictly masked during outages. Estimator received 0 position and 0 velocity updates.
2. **Normalization & Causality:** Zero leakage from test sets; strictly forward-sliding 20-sample causal windows.
3. **Heading Reference Scope:** In `src/evaluation/run_all_test_sequences_benchmark.py`, the coordinate transformation $R_{bn}(\psi)$ and NHC observation matrix $H(\psi)$ received synchronized vehicle reference heading `gt_hdg` logged from the CAN bus / dual-antenna GPS.
4. **Engineering Implication:** The benchmark measures **displacement dead reckoning under reference attitude/heading**. In unassisted deployment without a heading reference, open-loop gyroscope drift would degrade both trajectory orientation and NHC lateral constraints over 60–120s outages.

---

## 13. System Limitations & Risk Factors

1. **Multi-Sequence Drift Exceeds 10% in Majority of Scenarios:** While specific highway runs achieve $<10\%$ drift (`vw16a` at 120s, `vw16b` at 10s/60s), overall median moving drift across held-out test sequences is **25.79%**. Sub-10% dead reckoning is not universally achieved across all drivers and routes.
2. **Stationary Chassis Vibration Causes False Displacement:** Accelerometers detect engine vibration while parked (`vw15`), causing open-loop ML to predict residual forward creep ($1$–$2\text{ m/s}$). Stationary filtering (ZUPT) is essential to suppress false accumulation when stopped.
3. **Heading Error Degrades NHC & Orientation:** The current benchmark relies on reference vehicle heading. In unassisted consumer smartphones, heading drift will misorient displacement increments and weaken NHC lateral constraint effectiveness.
4. **Cross-Driver and Route Generalization:** Driving dynamics and smartphone mounting flexure create notable variance (e.g., Driver D `y1` vs Driver E `vw16a`).
5. **Dataset-Only Validation:** Validation has been performed on the IO-VNBD dataset. Physical smartphone road testing under OS scheduling jitter remains necessary.
6. **Android Edge Contract Pending:** Android application (`FE/gudumap/gudumap`) currently expects the `[1, 200, 6]` contract. Edge integration of the native 10 Hz `[1, 20, 6]` model is scheduled as a subsequent task.

---

## 14. Conclusion

The AI/ML Dead Reckoning system is **SCIENTIFICALLY VALIDATED ON REAL VEHICLE DATA WITH DISCLOSED BOUNDARIES**. Substantial reduction in dead-reckoning error was demonstrated on real held-out IO-VNBD vehicle sequences, with performance varying across routes, drivers, and motion conditions.
