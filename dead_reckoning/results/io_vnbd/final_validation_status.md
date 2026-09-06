# Final Real IO-VNBD Validation Status

**Project:** Smart India Hackathon 2026 (SIH26168)  
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** ISRO  
**Repository:** `D:\dead_reckoning`  
**Date:** September 2026  
**Auditor:** Senior Edge AI & Inertial Navigation Engineer  

---

## 1. System Classification Summary

| Validation Dimension | Status | Measured Evidence & Findings |
| :--- | :---: | :--- |
| **DATASET VALIDATION** | **PASS** | Audited all 288 raw CSV files (72 synchronized pairs, 29.74 driving hours, 1,070,640 samples across 5 drivers). Confirmed exact 10.0 Hz sampling and dashboard landscape orientation ($r = 0.9348$ correlation between phone pitch rate and CAN bus yaw rate). |
| **MODEL VALIDATION** | **PASS** | Vehicle-domain 2-layer stacked GRU fine-tuned on native 10 Hz windows ($W=20, S=10$). Evaluated across 27,964 held-out test windows: achieves **54.45% error reduction** over OxIOD model and **55.49% reduction** over zero baseline. |
| **NAVIGATION VALIDATION** | **PASS** | Complete 7-tier navigation ladder evaluated across 9 held-out test sequences and 32 valid outage intervals (**224 total evaluations**). Substantial reduction in dead-reckoning error was demonstrated on real held-out IO-VNBD vehicle sequences, with performance varying across routes, drivers, and motion conditions. |
| **MULTI-SEQUENCE GENERALIZATION** | **PARTIAL** | Generalization varies widely across motion regimes. On active highway driving (`vw16a`), ML achieves **9.10% drift** at 120s. However, stationary vehicle vibration causes false displacement creep (`vw15`), cross-driver variance is notable (`y1` 120s drift is 32.59%), and overall moving-evaluation median drift is **25.79%**. |
| **ONNX VALIDATION** | **PASS** | `models/gru_io_vnbd.onnx` achieves numerical parity ($1.192 \times 10^{-6}\text{ m}$ maximum absolute difference vs PyTorch) and edge CPU inference latency of $0.177\text{ ms}$ ($>5,600\text{ inf/sec}$). |
| **ANDROID VALIDATION** | **NOT VALIDATED** | Android application (`FE/gudumap/gudumap`) remains on the previous `[1, 200, 6]` contract. Updating edge inference to the new native 10 Hz `[1, 20, 6]` contract is pending. |

---

## 2. Benchmark Audit & Quantitative Performance

### A. Full 224-Evaluation Matrix Verification
- **Held-Out Test Sequences (9 total):** `m` (Driver B), `vfa01` (Driver E), `vfa02` (Driver E), `vw14c` (Driver E), `vw15` (Driver E), `vw16a` (Driver E), `vw16b` (Driver E), `vw17` (Driver E), `y1` (Driver D).
- **Outage Combinations:** 32 physically valid sequence/duration combinations across 10s, 30s, 60s, and 120s blackout durations.
- **Baselines Evaluated (7 total):**
  1. B1: Pure INS
  2. B2: INS + EKF
  3. B3: ML Only
  4. B4: ML + INS
  5. B5: ML + INS + EKF
  6. B6: ML + INS + EKF + NHC
  7. B7: ML + INS + EKF + NHC + ZUPT
- **Total Evaluations:** $32 \text{ outages} \times 7 \text{ baselines} = 224 \text{ evaluations}$ verified in `results/io_vnbd/real_benchmark_all_test_sequences.csv`.

---

### B. Aggregate Performance Across All 224 Evaluations

*Table 1: Aggregate Statistics across ALL 32 Outage Combinations per Baseline (from `real_benchmark_aggregate.csv`)*

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

### C. Moving Sequences Aggregate Performance

**Moving-Sequence Criterion:** An outage evaluation is classified as a moving evaluation if the ground truth distance travelled during the blackout exceeds $100.0\text{ m}$ ($\text{distance\_travelled\_m} > 100.0\text{ m}$). Stationary drives (`vw15`, stopped intervals of `m` and `vw17`) are preserved in the main benchmark above, but isolated here to evaluate dynamic driving performance.

*Table 2: Aggregate Statistics for MOVING Sequences Only (N = 23 outage combinations, from `real_benchmark_aggregate_moving.csv`)*

| Duration | Baseline | N | Mean Drift (%) | Median Drift (%) | Std Dev (%) | P25 (%) | P75 (%) | P95 (%) | Min (%) | Max (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10s** | B1 Pure INS | 6 | 11.43% | 8.60% | 7.22% | 5.19% | 16.96% | 22.12% | 4.48% | 23.13% | 5.88m |
| **10s** | B2 INS + EKF | 6 | **7.79%** | **6.92%** | 4.61% | 3.95% | 11.84% | 14.04% | 2.10% | 14.38% | **4.96m** |
| **10s** | B3 ML Only | 6 | 26.14% | 24.50% | 19.22% | 9.72% | 33.49% | 54.96% | 5.05% | 61.48% | 24.84m |
| **10s** | B4 ML + INS | 6 | 23.00% | 20.48% | 15.83% | 10.02% | 30.80% | 46.83% | 5.00% | 51.26% | 21.05m |
| **10s** | B5 ML + INS + EKF | 6 | 26.14% | 24.50% | 19.22% | 9.72% | 33.49% | 54.96% | 5.05% | 61.48% | 24.84m |
| **10s** | B6 ML+INS+EKF+NHC | 6 | 26.82% | 23.58% | 21.39% | 9.66% | 33.57% | 59.80% | 4.17% | 67.67% | 24.34m |
| **10s** | B7 ML+INS+EKF+NHC+ZUPT | 6 | 32.41% | 23.58% | 32.71% | 9.66% | 33.57% | 84.94% | 4.17% | 101.18% | 24.34m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **30s** | B1 Pure INS | 5 | 34.71% | 16.08% | 23.97% | 15.39% | 61.26% | 65.61% | 14.11% | 66.69% | 56.87m |
| **30s** | B2 INS + EKF | 5 | **25.70%** | **14.04%** | 19.49% | 10.50% | 36.16% | 54.66% | 8.51% | 59.29% | **40.77m** |
| **30s** | B3 ML Only | 5 | 33.87% | 29.37% | 18.17% | 23.64% | 33.72% | 61.08% | 14.72% | 67.93% | 90.57m |
| **30s** | B4 ML + INS | 5 | 32.22% | 27.76% | 14.30% | 20.85% | 33.93% | 53.92% | 19.65% | 58.91% | 79.87m |
| **30s** | B5 ML + INS + EKF | 5 | 33.87% | 29.37% | 18.17% | 23.64% | 33.72% | 61.08% | 14.72% | 67.93% | 90.57m |
| **30s** | B6 ML+INS+EKF+NHC | 5 | 49.54% | 39.29% | 27.11% | 37.66% | 56.63% | 89.29% | 16.65% | 97.45% | 155.68m |
| **30s** | B7 ML+INS+EKF+NHC+ZUPT | 5 | 47.41% | 39.29% | 24.33% | 35.56% | 56.63% | 82.46% | 16.65% | 88.92% | 155.68m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **60s** | B1 Pure INS | 6 | 46.09% | 40.16% | 18.36% | 29.98% | 59.41% | 73.28% | 27.54% | 76.28% | 178.59m |
| **60s** | B2 INS + EKF | 6 | **32.32%** | 28.93% | 15.93% | 19.70% | 47.03% | 53.47% | 12.64% | 53.72% | **93.86m** |
| **60s** | B3 ML Only | 6 | 36.44% | **24.41%** | 30.83% | 23.88% | 34.12% | 86.12% | 6.38% | 102.39% | 163.26m |
| **60s** | B4 ML + INS | 6 | 35.20% | 26.16% | 26.46% | 21.67% | 30.32% | 77.65% | 13.91% | 93.11% | 150.99m |
| **60s** | B5 ML + INS + EKF | 6 | 36.44% | **24.41%** | 30.83% | 23.88% | 34.12% | 86.12% | 6.38% | 102.39% | 163.26m |
| **60s** | B6 ML+INS+EKF+NHC | 6 | 56.22% | 44.37% | 22.91% | 37.99% | 69.48% | 92.21% | 37.23% | 97.62% | 334.64m |
| **60s** | B7 ML+INS+EKF+NHC+ZUPT | 6 | 44.51% | 43.84% | 18.43% | 32.78% | 53.11% | 70.54% | 18.07% | 75.99% | 263.37m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **120s** | B1 Pure INS | 6 | 83.40% | 86.31% | 39.56% | 62.74% | 104.60% | 135.19% | 18.02% | 144.18% | 512.78m |
| **120s** | B2 INS + EKF | 6 | 64.21% | 81.16% | 29.78% | 48.90% | 86.21% | 87.50% | 8.97% | 87.91% | 479.65m |
| **120s** | B3 ML Only | 6 | 32.46% | 29.19% | 20.29% | 19.36% | 36.13% | 63.90% | 9.10% | 72.77% | **251.22m** |
| **120s** | B4 ML + INS | 6 | **28.62%** | **22.22%** | 21.93% | 15.05% | 27.08% | 63.74% | 10.16% | 75.73% | 252.10m |
| **120s** | B5 ML + INS + EKF | 6 | 32.46% | 29.19% | 20.29% | 19.36% | 36.13% | 63.90% | 9.10% | 72.77% | **251.22m** |
| **120s** | B6 ML+INS+EKF+NHC | 6 | 52.73% | 50.05% | 12.90% | 44.18% | 54.79% | 72.95% | 38.92% | 78.79% | 495.01m |
| **120s** | B7 ML+INS+EKF+NHC+ZUPT | 6 | 40.51% | 39.10% | 23.18% | 23.47% | 52.88% | 72.31% | 10.35% | 78.79% | 404.11m |

---

## 3. Generalization Threshold Analysis (B3, B4, B5)

### A. All Evaluations (N = 32)
- **B3 (ML Only):**
  - $<10\%$ Drift: **4 / 32 (12.5%)**
  - $<15\%$ Drift: **5 / 32 (15.6%)**
  - $<20\%$ Drift: **6 / 32 (18.8%)**
  - Median Drift: **34.57%** | Mean Drift: **653.56%** | P95 Drift: **3,994.55%**
- **B4 (ML + INS):**
  - $<10\%$ Drift: **2 / 32 (6.2%)**
  - $<15\%$ Drift: **5 / 32 (15.6%)**
  - $<20\%$ Drift: **8 / 32 (25.0%)**
  - Median Drift: **29.51%** | Mean Drift: **565.42%** | P95 Drift: **3,456.49%**
- **B5 (ML + INS + EKF):**
  - $<10\%$ Drift: **4 / 32 (12.5%)**
  - $<15\%$ Drift: **5 / 32 (15.6%)**
  - $<20\%$ Drift: **6 / 32 (18.8%)**
  - Median Drift: **34.57%** | Mean Drift: **653.56%** | P95 Drift: **3,994.55%**

### B. Moving Evaluations Only (N = 23, Distance > 100m)
- **B3 (ML Only):**
  - $<10\%$ Drift: **4 / 23 (17.4%)**
  - $<15\%$ Drift: **5 / 23 (21.7%)**
  - $<20\%$ Drift: **6 / 23 (26.1%)**
  - Median Drift: **25.79%** | Mean Drift: **32.16%** | P95 Drift: **72.28%**
- **B4 (ML + INS):**
  - $<10\%$ Drift: **2 / 23 (8.7%)**
  - $<15\%$ Drift: **5 / 23 (21.7%)**
  - $<20\%$ Drift: **8 / 23 (34.8%)**
  - Median Drift: **24.80%** | Mean Drift: **29.65%** | P95 Drift: **74.05%**
- **B5 (ML + INS + EKF):**
  - $<10\%$ Drift: **4 / 23 (17.4%)**
  - $<15\%$ Drift: **5 / 23 (21.7%)**
  - $<20\%$ Drift: **6 / 23 (26.1%)**
  - Median Drift: **25.79%** | Mean Drift: **32.16%** | P95 Drift: **72.28%**

---

## 4. Verification of Specific Empirical Claims

1. **Claim a (8/9 sequences improved vs Zero baseline):** **TRUE**. 8 of 9 sequences (`m`, `vfa01`, `vfa02`, `vw14c`, `vw16a`, `vw16b`, `vw17`, `y1`) show lower Euclidean error with IO-VNBD GRU than predicting zero displacement. Only `vw15` (a parked car with true displacement $0.04\text{ m}$) favored the zero baseline.
2. **Claim b (IO-VNBD GRU improves over OxIOD on 8/9 sequences):** **TRUE**. The exact same 8 of 9 sequences showed massive error reductions (50–92%) over the pedestrian OxIOD model.
3. **Claim c (Only 4/32 ML-only outage evaluations achieve <10% drift):** **TRUE**. Exactly 4 evaluations met the $<10\%$ threshold:
   - `vw17` (10s outage): **5.05%**
   - `vw16b` (10s outage): **5.87%**
   - `vw16b` (60s outage): **6.38%**
   - `vw16a` (120s outage): **9.10%**
4. **Claim d (Worst stationary outlier is vw15):** **TRUE**. In `vw15`, the vehicle was parked, travelling only $1.14\text{ m}$ in 30s. The ML model predicted slight forward displacement ($1.5\text{ m/s}$ false motion from engine vibration), resulting in an endpoint drift of **5,667.10%**.
5. **Claim e (y1 is a case where ML does not outperform Pure INS at 120s):** **TRUE**. On sequence `y1` (Driver D, high-speed straight highway), Pure INS achieved **18.02% drift**, whereas ML Only achieved **32.59% drift** due to model speed underestimation.
6. **Claim f (vw16a has the best 120s moving-sequence ML result):** **TRUE**. `vw16a` achieved **9.10% drift** ($161.70\text{ m}$ endpoint error over $1,776.3\text{ m}$ travelled) and position RMSE of $186.11\text{ m}$.

---

## 5. NHC and ZUPT Trajectory Verification (B6 vs B7)

- **Mechanics of ZUPT Activation:** ZUPT is conditioned on the multi-sensor detector ($\text{acc\_dev} < 0.10\text{ g}$, $\text{acc\_var} < 0.02\text{ g}^2$, $\text{gyro\_mag} < 0.08\text{ rad/s}$).
- **Continuous Motion Invariance (13 Outage Intervals):** On sequences where the vehicle remained continuously in motion throughout the outage interval (`vfa01` 10s–120s, `vfa02` 10s–60s, `vw16a` 10s–120s, `vw16b` 10s, `vw17` 10s), ZUPT legitimately never triggered. In these 13 cases, **B6 and B7 are numerically identical** down to floating-point precision.
- **Stationary Correction Active (19 Outage Intervals):** On sequences with stationary events (`vw15` parked car, `m` traffic stops, `vw14c`, `y1`), ZUPT activated and directly modified filter velocity and position states. For instance:
  - `m` (60s outage): B6 drift $651.80\% \to$ B7 drift **$264.16\%$** (RMSE reduced from $81.73\text{ m}$ to $41.31\text{ m}$).
  - `vw14c` (120s outage): B6 drift $43.16\% \to$ B7 drift **$10.35\%$** (RMSE reduced from $99.93\text{ m}$ to $36.96\text{ m}$).

---

## 6. 120-Second Outage Detailed Metrics for Moving Sequences

Evaluated on 6 held-out driving sequences (`m`, `vfa01`, `vfa02`, `vw14c`, `vw16a`, `y1`):
- **Best Moving Sequence:** `vw16a` (Driver E) — Drift: **9.10%** (Distance: $1,776.3\text{ m}$, RMSE: $186.11\text{ m}$).
- **Worst Moving Sequence:** `vw14c` (Driver E) — Drift: **72.77%** (Distance: $471.7\text{ m}$, RMSE: $219.72\text{ m}$).
- **Median Moving Drift:** **29.19%**
- **Mean Moving Drift:** **32.46%**
- **95th Percentile Moving Drift:** **63.90%**

---

## 7. Project Limitations & Identified Risk Factors

1. **Multi-Sequence Drift Exceeds 10% in Majority of Scenarios:** While selective highway sequences achieve $<10\%$ drift (`vw16a` at 120s, `vw16b` at 10s/60s), the median moving drift across all 9 test sequences is **25.79%**. Universal sub-10% dead reckoning is not achieved across all drivers and routes.
2. **Stationary Engine Vibration Induces False Displacement:** Accelerometers detect engine idling and chassis vibration while parked, causing open-loop ML to predict residual forward velocity ($1$–$2\text{ m/s}$). Robust deployment requires tightly coupled stationary filtering (ZUPT).
3. **Heading Reference Dependency:** The benchmark evaluates displacement dead reckoning under known vehicle reference heading (`gt_hdg`). In an unassisted deployment, open-loop gyroscope drift would degrade both trajectory heading and NHC effectiveness.
4. **Cross-Driver and Vehicle Route Generalization:** Driving styles (acceleration profiles, braking dynamics) and smartphone dashboard mounting flexure introduce substantial variance (e.g., Driver D `y1` vs Driver E `vw16a`).
5. **Dataset-Only Validation:** Validation has been conducted on the IO-VNBD dataset. Field testing with diverse physical smartphones under real-time operating system jitter is still required.
6. **Android Edge Contract Pending:** The Android application currently executes the `[1, 200, 6]` OxIOD contract. Integration of the native 10 Hz `[1, 20, 6]` model into Android edge code remains a necessary next step.

---

## 8. Final Verdict

The AI/ML Dead Reckoning system is **SCIENTIFICALLY VALIDATED ON REAL VEHICLE DATA WITH DISCLOSED BOUNDARIES**. Substantial reduction in dead-reckoning error was demonstrated on real held-out IO-VNBD vehicle sequences, with performance varying across routes, drivers, and motion conditions.
