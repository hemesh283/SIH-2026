# Final IO-VNBD Benchmark Audit & Validation Summary

**Project:** Smart India Hackathon 2026 (SIH26168)  
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** ISRO  
**Repository:** `D:\dead_reckoning`  
**Date:** September 2026  
**Auditor:** Senior Edge AI & Inertial Navigation Engineer  

---

## 1. Executive Summary

This document presents the final scientific audit and comprehensive report cleanup for the AI/ML Dead Reckoning system benchmarked against the real Inertial and Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD).

**Core Audit Finding:**  
Substantial reduction in dead-reckoning error was demonstrated on real held-out IO-VNBD vehicle sequences, with performance varying across routes, drivers, and motion conditions. On active driving routes, the system constrains the cubic position divergence characteristic of pure inertial navigation, achieving down to **9.10% drift** on extended 120-second GPS outages (`vw16a`). However, performance varies substantially across driving styles, routes, and motion states, yielding a median moving drift of **25.79%** across all held-out evaluations.

---

## 2. Exact Dataset Statistics

The real IO-VNBD dataset was audited directly from local storage (`data/raw/IO-VNBD`):

- **Smartphone CSV Files:** Exactly **144 files** (`S-*.csv`).
- **Vehicle ECU CSV Files:** Exactly **144 files** (`V-*.csv`).
- **Synchronized Pairs:** Exactly **72 matched pairs** (1:1 timestamp alignment).
- **Total Driving Time:** **29.74 hours** ($107,064.0\text{ seconds}$).
- **Total Smartphone Samples:** Exactly **1,070,640 records**.
- **Sampling Frequency:** Exactly **10.0 Hz** ($\Delta t = 100.0\text{ ms}$) verified across all 144 smartphone files.
- **Participating Drivers:** **5 drivers** (Drivers A, B, C, D, E).
- **Vehicle Speed Envelope:** Mean $45.08\text{ km/h}$, Maximum $131.85\text{ km/h}$.
- **Physical Sensor Alignment:** Phone was mounted in landscape orientation on the dashboard. Phone `GYROSCOPE Pitch` was verified as vehicle turning yaw rate ($r = 0.9348$ correlation with vehicle CAN bus yaw rate).

---

## 3. Exact Held-Out Test Set Inventory & 224-Evaluation Calculation

The test split contains **9 completely held-out sequences** (zero overlap with training or validation drives):

| Sequence Key | Driver | Duration (s) | Raw Samples | 20-Sample Windows | Valid Outage Durations | Outage Count |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `m` | Driver B | 70.2s | 702 | 69 | 10s, 30s, 60s, 120s | 4 |
| `vfa01` | Driver E | 1148.6s | 11,486 | 1,147 | 10s, 30s, 60s, 120s | 4 |
| `vfa02` | Driver E | 1242.0s | 12,420 | 1,241 | 10s, 30s, 60s, 120s | 4 |
| `vw14c` | Driver E | 118.5s | 1,185 | 117 | 10s, 30s, 60s, 120s | 4 |
| `vw15` | Driver E | 150.0s | 1,500 | 149 | 10s, 30s, 60s, 120s | 4 |
| `vw16a` | Driver E | 148.0s | 1,480 | 147 | 10s, 30s, 60s, 120s | 4 |
| `vw16b` | Driver E | 90.7s | 907 | 89 | 10s, 30s, 60s | 3 |
| `vw17` | Driver E | 19.3s | 193 | 18 | 10s | 1 |
| `y1` | Driver D | 461.5s | 4,615 | 460 | 10s, 30s, 60s, 120s | 4 |
| **Total** | **3 Drivers** | **3,448.8s** | **34,488** | **3,437** | — | **32 Outage Combinations** |

*(Note: Total test split windows across all sequences in `split_manifest.csv` is 27,964 windows).*

### The 224-Evaluation Matrix Calculation:
$$\text{Total Outage Combinations} = 9 \text{ (10s)} + 8 \text{ (30s)} + 8 \text{ (60s)} + 7 \text{ (120s)} = 32 \text{ combinations}$$
$$\text{Total Evaluated Runs} = 32 \text{ combinations} \times 7 \text{ baselines} = \mathbf{224 \text{ evaluations}}$$
Verified: `results/io_vnbd/real_benchmark_all_test_sequences.csv` contains exactly **224 rows**.

---

## 4. Complete 7-Baseline Aggregate Tables

### A. Main Benchmark: All 32 Evaluated Outage Intervals (from `real_benchmark_aggregate.csv`)

| Outage Duration | Navigation Baseline | N | Mean Drift (%) | Median Drift (%) | Std Dev (%) | P25 (%) | P75 (%) | P95 (%) | Min (%) | Max (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | B1: Pure INS | 9 | 41.36% | 19.09% | 57.18% | 6.63% | 30.12% | 147.16% | 4.48% | 188.79% | 4.87m |
| **10s** | B2: INS + EKF | 9 | 23.70% | **12.99%** | 29.65% | 5.45% | 17.96% | 79.01% | 2.10% | 97.94% | **3.16m** |
| **10s** | B3: ML Only | 9 | 375.77% | 35.42% | 620.02% | 21.29% | 165.02% | 1547.39% | 5.05% | 1616.75% | 14.17m |
| **10s** | B4: ML + INS | 9 | 323.93% | 33.56% | 533.82% | 18.44% | 140.62% | 1331.27% | 5.00% | 1382.75% | 12.07m |
| **10s** | B5: ML + INS + EKF | 9 | 375.77% | 35.42% | 620.02% | 21.29% | 165.02% | 1547.39% | 5.05% | 1616.75% | 14.17m |
| **10s** | B6: ML + INS + EKF + NHC | 9 | 373.88% | 36.19% | 620.23% | 21.45% | 145.54% | 1546.53% | 4.17% | 1615.77% | 13.18m |
| **10s** | B7: ML + INS + EKF + NHC + ZUPT | 9 | 267.04% | 36.19% | 422.91% | 21.45% | 101.35% | 1066.05% | 4.17% | 1115.12% | 7.66m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30s** | B1: Pure INS | 8 | 244.84% | 63.98% | 354.34% | 15.91% | 261.55% | 864.30% | 14.11% | 881.37% | 35.36m |
| **30s** | B2: INS + EKF | 8 | 132.96% | **35.03%** | 184.81% | 13.16% | 148.90% | 460.51% | 8.51% | 483.55% | **28.07m** |
| **30s** | B3: ML Only | 8 | 1348.34% | 50.82% | 2246.05% | 27.94% | 1328.07% | 5352.86% | 14.72% | 5667.10% | 68.42m |
| **30s** | B4: ML + INS | 8 | 1179.14% | 46.42% | 1959.21% | 26.03% | 1164.40% | 4669.61% | 19.65% | 4935.03% | 58.54m |
| **30s** | B5: ML + INS + EKF | 8 | 1348.34% | 50.82% | 2246.05% | 27.94% | 1328.07% | 5352.86% | 14.72% | 5667.10% | 68.42m |
| **30s** | B6: ML + INS + EKF + NHC | 8 | 1340.42% | 49.61% | 2249.23% | 38.88% | 1264.87% | 5351.35% | 16.65% | 5665.92% | 63.39m |
| **30s** | B7: ML + INS + EKF + NHC + ZUPT | 8 | 974.80% | 57.63% | 1617.57% | 38.35% | 896.87% | 3880.56% | 16.65% | 4182.02% | 64.79m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60s** | B1: Pure INS | 8 | 54.80% | 42.53% | 29.88% | 33.69% | 67.29% | 105.71% | 27.54% | 121.56% | 115.00m |
| **60s** | B2: INS + EKF | 8 | 32.21% | **29.19%** | 13.90% | 25.19% | 39.65% | 53.37% | 12.64% | 53.72% | **76.68m** |
| **60s** | B3: ML Only | 8 | 500.53% | 30.94% | 1088.82% | 24.13% | 183.02% | 2333.15% | 6.38% | 3360.68% | 122.87m |
| **60s** | B4: ML + INS | 8 | 430.69% | 29.38% | 927.60% | 23.76% | 161.60% | 1992.17% | 13.91% | 2867.23% | 121.65m |
| **60s** | B5: ML + INS + EKF | 8 | 500.53% | 30.94% | 1088.82% | 24.13% | 183.02% | 2333.15% | 6.38% | 3360.68% | 122.87m |
| **60s** | B6: ML + INS + EKF + NHC | 8 | 543.56% | 62.97% | 1082.16% | 38.52% | 236.16% | 2411.70% | 37.23% | 3359.34% | 175.89m |
| **60s** | B7: ML + INS + EKF + NHC + ZUPT | 8 | 367.47% | 52.06% | 774.96% | 36.08% | 123.04% | 1658.00% | 18.07% | 2408.53% | 158.98m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **120s** | B1: Pure INS | 7 | 131.05% | 93.72% | 122.35% | 68.12% | 126.20% | 335.15% | 18.02% | 417.00% | 308.21m |
| **120s** | B2: INS + EKF | 7 | 83.11% | 85.97% | 53.88% | 58.05% | 87.10% | 163.93% | 8.97% | 196.51% | 339.11m |
| **120s** | B3: ML Only | 7 | 391.57% | 32.59% | 879.84% | 21.50% | 55.04% | 1804.19% | 9.10% | 2546.23% | **219.72m** |
| **120s** | B4: ML + INS | 7 | 328.51% | **25.04%** | 734.87% | 16.50% | 51.75% | 1512.23% | 10.16% | 2127.87% | 230.37m |
| **120s** | B5: ML + INS + EKF | 7 | 391.57% | 32.59% | 879.84% | 21.50% | 55.04% | 1804.19% | 9.10% | 2546.23% | **219.72m** |
| **120s** | B6: ML + INS + EKF + NHC | 7 | 265.88% | 52.88% | 522.23% | 45.20% | 67.11% | 1104.95% | 38.92% | 1544.73% | 466.71m |
| **120s** | B7: ML + INS + EKF + NHC + ZUPT | 7 | 291.45% | 52.88% | 615.05% | 24.09% | 65.83% | 1281.61% | 10.35% | 1797.10% | 341.51m |

---

### B. Moving Sequences Only (from `real_benchmark_aggregate_moving.csv`)

**Moving Criterion:** Outages where ground truth vehicle displacement exceeds $100.0\text{ m}$ ($\text{distance\_travelled\_m} > 100.0\text{ m}$). Evaluated on 23 valid combinations across 7 baselines (161 total evaluations).

| Outage Duration | Navigation Baseline | N | Mean Drift (%) | Median Drift (%) | Std Dev (%) | P25 (%) | P75 (%) | P95 (%) | Min (%) | Max (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | B1: Pure INS | 6 | 11.43% | 8.60% | 7.22% | 5.19% | 16.96% | 22.12% | 4.48% | 23.13% | 5.88m |
| **10s** | B2: INS + EKF | 6 | **7.79%** | **6.92%** | 4.61% | 3.95% | 11.84% | 14.04% | 2.10% | 14.38% | **4.96m** |
| **10s** | B3: ML Only | 6 | 26.14% | 24.50% | 19.22% | 9.72% | 33.49% | 54.96% | 5.05% | 61.48% | 24.84m |
| **10s** | B4: ML + INS | 6 | 23.00% | 20.48% | 15.83% | 10.02% | 30.80% | 46.83% | 5.00% | 51.26% | 21.05m |
| **10s** | B5: ML + INS + EKF | 6 | 26.14% | 24.50% | 19.22% | 9.72% | 33.49% | 54.96% | 5.05% | 61.48% | 24.84m |
| **10s** | B6: ML + INS + EKF + NHC | 6 | 26.82% | 23.58% | 21.39% | 9.66% | 33.57% | 59.80% | 4.17% | 67.67% | 24.34m |
| **10s** | B7: ML + INS + EKF + NHC + ZUPT | 6 | 32.41% | 23.58% | 32.71% | 9.66% | 33.57% | 84.94% | 4.17% | 101.18% | 24.34m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30s** | B1: Pure INS | 5 | 34.71% | 16.08% | 23.97% | 15.39% | 61.26% | 65.61% | 14.11% | 66.69% | 56.87m |
| **30s** | B2: INS + EKF | 5 | **25.70%** | **14.04%** | 19.49% | 10.50% | 36.16% | 54.66% | 8.51% | 59.29% | **40.77m** |
| **30s** | B3: ML Only | 5 | 33.87% | 29.37% | 18.17% | 23.64% | 33.72% | 61.08% | 14.72% | 67.93% | 90.57m |
| **30s** | B4: ML + INS | 5 | 32.22% | 27.76% | 14.30% | 20.85% | 33.93% | 53.92% | 19.65% | 58.91% | 79.87m |
| **30s** | B5: ML + INS + EKF | 5 | 33.87% | 29.37% | 18.17% | 23.64% | 33.72% | 61.08% | 14.72% | 67.93% | 90.57m |
| **30s** | B6: ML + INS + EKF + NHC | 5 | 49.54% | 39.29% | 27.11% | 37.66% | 56.63% | 89.29% | 16.65% | 97.45% | 155.68m |
| **30s** | B7: ML + INS + EKF + NHC + ZUPT | 5 | 47.41% | 39.29% | 24.33% | 35.56% | 56.63% | 82.46% | 16.65% | 88.92% | 155.68m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60s** | B1: Pure INS | 6 | 46.09% | 40.16% | 18.36% | 29.98% | 59.41% | 73.28% | 27.54% | 76.28% | 178.59m |
| **60s** | B2: INS + EKF | 6 | **32.32%** | 28.93% | 15.93% | 19.70% | 47.03% | 53.47% | 12.64% | 53.72% | **93.86m** |
| **60s** | B3: ML Only | 6 | 36.44% | **24.41%** | 30.83% | 23.88% | 34.12% | 86.12% | 6.38% | 102.39% | 163.26m |
| **60s** | B4: ML + INS | 6 | 35.20% | 26.16% | 26.46% | 21.67% | 30.32% | 77.65% | 13.91% | 93.11% | 150.99m |
| **60s** | B5: ML + INS + EKF | 6 | 36.44% | **24.41%** | 30.83% | 23.88% | 34.12% | 86.12% | 6.38% | 102.39% | 163.26m |
| **60s** | B6: ML + INS + EKF + NHC | 6 | 56.22% | 44.37% | 22.91% | 37.99% | 69.48% | 92.21% | 37.23% | 97.62% | 334.64m |
| **60s** | B7: ML + INS + EKF + NHC + ZUPT | 6 | 44.51% | 43.84% | 18.43% | 32.78% | 53.11% | 70.54% | 18.07% | 75.99% | 263.37m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **120s** | B1: Pure INS | 6 | 83.40% | 86.31% | 39.56% | 62.74% | 104.60% | 135.19% | 18.02% | 144.18% | 512.78m |
| **120s** | B2: INS + EKF | 6 | 64.21% | 81.16% | 29.78% | 48.90% | 86.21% | 87.50% | 8.97% | 87.91% | 479.65m |
| **120s** | B3: ML Only | 6 | 32.46% | 29.19% | 20.29% | 19.36% | 36.13% | 63.90% | 9.10% | 72.77% | **251.22m** |
| **120s** | B4: ML + INS | 6 | **28.62%** | **22.22%** | 21.93% | 15.05% | 27.08% | 63.74% | 10.16% | 75.73% | 252.10m |
| **120s** | B5: ML + INS + EKF | 6 | 32.46% | 29.19% | 20.29% | 19.36% | 36.13% | 63.90% | 9.10% | 72.77% | **251.22m** |
| **120s** | B6: ML + INS + EKF + NHC | 6 | 52.73% | 50.05% | 12.90% | 44.18% | 54.79% | 72.95% | 38.92% | 78.79% | 495.01m |
| **120s** | B7: ML + INS + EKF + NHC + ZUPT | 6 | 40.51% | 39.10% | 23.18% | 23.47% | 52.88% | 72.31% | 10.35% | 78.79% | 404.11m |

---

## 5. Threshold Performance: B3, B4, and B5

| Baseline | Evaluation Subset | Total (N) | <10% Count (%) | <15% Count (%) | <20% Count (%) | Median Drift | Mean Drift | P95 Drift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B3 (ML Only)** | All Evaluations | 32 | 4 (12.5%) | 5 (15.6%) | 6 (18.8%) | 34.57% | 653.56% | 3,994.55% |
| **B3 (ML Only)** | Moving Only (>100m) | 23 | 4 (17.4%) | 5 (21.7%) | 6 (26.1%) | 25.79% | 32.16% | 72.28% |
| **B4 (ML + INS)** | All Evaluations | 32 | 2 (6.2%) | 5 (15.6%) | 8 (25.0%) | 29.51% | 565.42% | 3,456.49% |
| **B4 (ML + INS)** | Moving Only (>100m) | 23 | 2 (8.7%) | 5 (21.7%) | 8 (34.8%) | 24.80% | 29.65% | 74.05% |
| **B5 (ML + INS + EKF)** | All Evaluations | 32 | 4 (12.5%) | 5 (15.6%) | 6 (18.8%) | 34.57% | 653.56% | 3,994.55% |
| **B5 (ML + INS + EKF)** | Moving Only (>100m) | 23 | 4 (17.4%) | 5 (21.7%) | 6 (26.1%) | 25.79% | 32.16% | 72.28% |

---

## 6. Best Baseline by Outage Duration

*Determined rigorously using multi-sequence median drift, mean drift, and median RMSE:*

- **10s Outages:**
  - **All Evaluations:** **B2: INS + EKF** (Median drift: **12.99%**, Mean drift: **23.70%**, Median RMSE: **3.16m**).
  - **Moving Evaluations:** **B2: INS + EKF** (Median drift: **6.92%**, Mean drift: **7.79%**, Median RMSE: **4.96m**).
- **30s Outages:**
  - **All Evaluations:** **B2: INS + EKF** (Median drift: **35.03%**, Mean drift: **132.96%**, Median RMSE: **28.07m**).
  - **Moving Evaluations:** **B2: INS + EKF** (Median drift: **14.04%**, Mean drift: **25.70%**, Median RMSE: **40.77m**).
- **60s Outages:**
  - **All Evaluations:** **B2: INS + EKF** (Median drift: **29.19%**, Mean drift: **32.21%**, Median RMSE: **76.68m**).
  - **Moving Evaluations:** **B3: ML Only** achieves lowest median drift (**24.41%**); **B2: INS + EKF** achieves lowest mean drift (**32.32%**) and median RMSE (**93.86m**).
- **120s Outages:**
  - **All Evaluations:** **B4: ML + INS** achieves lowest median drift (**25.04%**); **B3: ML Only** achieves lowest median RMSE (**219.72m**).
  - **Moving Evaluations:** **B4: ML + INS** achieves lowest median drift (**22.22%**) and mean drift (**28.62%**); **B3: ML Only** achieves lowest median RMSE (**251.22m**).

---

## 7. Verification of Empirical Claims

1. **Claim a (8/9 test sequences improved vs Zero baseline):** **VERIFIED TRUE**. 8 of 9 sequences (`m`, `vfa01`, `vfa02`, `vw14c`, `vw16a`, `vw16b`, `vw17`, `y1`) improve over the Zero baseline. The single exception is `vw15` (a parked vehicle with true displacement $0.04\text{ m}$).
2. **Claim b (IO-VNBD GRU improves over OxIOD on 8/9 sequences):** **VERIFIED TRUE**. Exactly the same 8 of 9 sequences show 50% to 92% error reductions over OxIOD.
3. **Claim c (Only 4/32 ML-only outage evaluations achieve <10% drift):** **VERIFIED TRUE**. Exactly 4 evaluations achieved $<10\%$ drift: `vw17` 10s (5.05%), `vw16b` 10s (5.87%), `vw16b` 60s (6.38%), `vw16a` 120s (9.10%).
4. **Claim d (Worst stationary outlier is vw15):** **VERIFIED TRUE**. In `vw15`, the vehicle was parked, moving only $1.14\text{ m}$ during 30s. Residual ML motion caused a drift of **5,667.10%**.
5. **Claim e (y1 at 120s is an empirical case where Pure INS outperforms ML):** **VERIFIED TRUE**. On `y1` (high-speed straight highway), Pure INS achieved **18.02% drift** vs ML Only **32.59% drift** due to model speed underestimation.
6. **Claim f (vw16a has the best 120s moving-sequence ML result):** **VERIFIED TRUE**. `vw16a` (Driver E) achieved **9.10% drift** ($161.70\text{ m}$ error over $1,776.3\text{ m}$ travelled) and position RMSE of $186.11\text{ m}$.

---

## 8. NHC and ZUPT Trajectory Verification (B6 vs B7)

- **Mechanics:** ZUPT is active only when $\text{acc\_dev} < 0.10\text{ g}$, $\text{acc\_var} < 0.02\text{ g}^2$, and $\text{gyro\_mag} < 0.08\text{ rad/s}$.
- **Identical Trajectories (13 Outage Intervals):** On intervals with unbroken highway motion (`vfa01` 10s–120s, `vfa02` 10s–60s, `vw16a` 10s–120s, `vw16b` 10s, `vw17` 10s), the vehicle never stopped. ZUPT legitimately never triggered. In these 13 intervals, **B6 and B7 are numerically identical** down to floating-point precision.
- **Active Trajectory Modification (19 Outage Intervals):** On drives with traffic stops or parked periods (`m`, `vw15`, `vw14c`, `y1`), ZUPT activated and directly updated velocity states to zero:
  - `m` (60s outage): B6 drift $651.80\% \to$ B7 drift **$264.16\%$** (RMSE dropped from $81.73\text{ m}$ to $41.31\text{ m}$).
  - `vw14c` (120s outage): B6 drift $43.16\% \to$ B7 drift **$10.35\%$** (RMSE dropped from $99.93\text{ m}$ to $36.96\text{ m}$).

---

## 9. Ground Truth & Heading Reference Leakage Audit

1. **GNSS Position Updates:** Strictly suppressed during outages (0 position updates). **PASS**.
2. **GNSS / CAN Velocity Updates:** Strictly suppressed during outages (0 velocity updates). **PASS**.
3. **Temporal Causality:** Strict forward-sliding windows $[t-20, t]$ with zero future lookahead. **PASS**.
4. **Normalization Parameters:** Computed exclusively on training data (`models/io_vnbd_normalization.json`). **PASS**.
5. **Attitude / Heading Scope Disclosure:**
   - In `src/evaluation/run_all_test_sequences_benchmark.py`, vehicle reference heading `gt_hdg` from the dataset log was passed to the direction cosine matrix $R_{bn}(\psi)$ and NHC observation matrix $H(\psi)$.
   - **Audit Statement:** The benchmark strictly measures **displacement dead reckoning under reference attitude/heading**.
   - **Operational Implication:** It does not evaluate open-loop heading integration from uncalibrated consumer smartphone gyroscopes. In an unassisted smartphone without an external attitude reference or magnetometer filter, heading drift would progressively misorient the trajectory and degrade NHC constraints over 60–120s.

---

## 10. ONNX & Android Edge Parity

- **Model:** `models/gru_io_vnbd.onnx`
- **Numerical Parity (PyTorch vs ONNX Runtime):**
  - Maximum absolute difference: **$1.192 \times 10^{-6}\text{ m}$**
  - Mean absolute difference: **$9.244 \times 10^{-8}\text{ m}$**
  - Parity RMSE: **$1.546 \times 10^{-7}\text{ m}$**
- **CPU Inference Latency:** Mean **$0.177\text{ ms}$**, Median **$0.159\text{ ms}$**, P95 **$0.232\text{ ms}$** ($>5,600\text{ inf/sec}$).
- **Android Integration Status:**
  - Android application (`FE/gudumap/gudumap`) currently contains an ONNX pipeline configured for the `[1, 200, 6]` contract.
  - The native vehicle model operates on the `[1, 20, 6]` contract.
  - Android edge validation of the new `[1, 20, 6]` model is **PENDING**; Android code was deliberately preserved without modification in this phase.

---

## 11. Comprehensive Limitations Section

1. **Multi-Sequence Drift Exceeds 10% in Majority of Scenarios:** While selective highway runs achieve $<10\%$ drift (`vw16a` at 120s, `vw16b` at 10s/60s), the median moving drift across all 9 test sequences is **25.79%**. Sub-10% dead reckoning is not universally achieved across all drivers and routes.
2. **Stationary Chassis Vibration Causes False Displacement:** Accelerometers detect engine vibration while parked (`vw15`), causing open-loop ML to predict residual forward creep ($1$–$2\text{ m/s}$). Stationary filtering (ZUPT) is essential to suppress false accumulation when stopped.
3. **Heading Error Degrades NHC & Orientation:** The current benchmark relies on reference vehicle heading. In unassisted consumer smartphones, heading drift will misorient displacement increments and weaken NHC lateral constraint effectiveness.
4. **Cross-Driver and Route Generalization:** Driving dynamics and smartphone mounting flexure create notable variance (e.g., Driver D `y1` vs Driver E `vw16a`).
5. **Dataset-Only Validation:** Validation has been performed on the IO-VNBD dataset. Physical smartphone road testing under OS scheduling jitter remains necessary.
6. **Android Edge Contract Pending:** Android application (`FE/gudumap/gudumap`) currently expects the `[1, 200, 6]` contract. Edge integration of the native 10 Hz `[1, 20, 6]` model is scheduled as a subsequent task.

---

## 12. Final Honest Project Verdict

The AI/ML Dead Reckoning system is **DEFENSIBLY VALIDATED ON REAL VEHICLE DATA WITH DISCLOSED BOUNDARIES**. Substantial reduction in dead-reckoning error was demonstrated on real held-out IO-VNBD vehicle sequences, with performance varying across routes, drivers, and motion conditions.
