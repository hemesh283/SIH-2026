# Evaluation Freeze Manifest: Real IO-VNBD Benchmark

**Project:** Smart India Hackathon 2026 (SIH26168)  
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** Indian Space Research Organisation (ISRO)  
**Repository:** `D:\dead_reckoning`  
**Evaluation Target:** Real IO-VNBD Native 10 Hz Vehicle Navigation Benchmark  
**Freeze Status:** **FROZEN & SEALED**  
**Freeze Timestamp:** 2026-09-05T11:42:00+05:30  

---

## 1. Project & Scope Statement

This document establishes the authoritative, immutable **Evaluation Freeze** for the AI/ML Dead Reckoning system evaluated on the real Inertial and Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD).

These frozen evaluation outputs represent the complete, final experimental record of the project. No further tuning, model modification, parameter changes, retraining, or metric recomputation will alter these figures.

---

## 2. Dataset & Split Inventory

- **Dataset:** IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)
- **Total Physical Files:** 144 smartphone CSVs (`S-*.csv`) and 144 vehicle ECU CSVs (`V-*.csv`)
- **Matched Synchronized Pairs:** Exactly 72 pairs
- **Total Driving Time:** 29.74 hours ($107,064.0\text{ s}$)
- **Total Smartphone Records:** 1,070,640 samples
- **Sampling Frequency:** Exactly 10.0 Hz ($\Delta t = 100.0\text{ ms}$) verified across all files
- **Number of Drivers:** 5 drivers (Drivers A, B, C, D, E)
- **Speed Envelope:** Mean $45.08\text{ km/h}$, Maximum $131.85\text{ km/h}$

### Partitioning Scheme (Strictly Disjoint Drives):
| Partition | Number of Drives | 20-Sample Windows | Participating Drivers | Role in Project |
| :--- | :---: | :---: | :--- | :--- |
| **TRAIN** | 53 | 45,742 | Driver C (`Vta1`–`Vta30`, `Vtb1`–`Vtb12`), Driver E (`Vw1`–`Vw11`) | Model training & optimization |
| **VAL** | 10 | 33,262 | Driver A (`S1`–`S4`), Driver E (`Vw12`–`Vw14b`) | Hyperparameter tuning & checkpoint selection |
| **TEST** | 9 | 27,964 | Driver B (`M`), Driver D (`Y1`), Driver E (`Vw14c`–`Vw17`, `Vfa01`–`Vfa02`) | Independent held-out evaluation |

### Non-Tuning & Zero-Leakage Declaration:
> **Explicit Non-Tuning Certification:**  
> The 9 held-out test sequences (`m`, `vfa01`, `vfa02`, `vw14c`, `vw15`, `vw16a`, `vw16b`, `vw17`, `y1`) were strictly quarantined and used exclusively for final post-hoc evaluation. No sample, window, or label from the test partition was used for model training, gradient backpropagation, hyperparameter tuning, loss weighting, architecture selection, early-stopping decisions, or normalization statistics computation ($\boldsymbol{\mu}_X, \boldsymbol{\sigma}_X, \boldsymbol{\mu}_y, \boldsymbol{\sigma}_y$ were computed strictly from the 53 training sequences).

---

## 3. Model Contract & Architecture

- **Model Artifacts:**
  - PyTorch Checkpoint: `models/frozen_io_vnbd/gru_io_vnbd_best.pt` (SHA-256: `0DD59052D14F24E507AFF096E21100E7CE1AC56C9734B6C59B3FE07AFF7815C0`)
  - ONNX Model: `models/frozen_io_vnbd/gru_io_vnbd.onnx` (SHA-256: `32A512EB668205D2C358FB7D94BE84B24089112A2DC3264D9DCB6E323935FABF`)
  - Normalization JSON: `models/frozen_io_vnbd/io_vnbd_normalization.json` (SHA-256: `A0DF903FFAA4FA491992BCDF763A8B99AA8228D1CACEF7167CD91FC9C73831C6`)
  - Metadata JSON: `models/frozen_io_vnbd/model_metadata.json` (SHA-256: `5F4030F27713369761CC15E70A8F9A414967C83B703ACF1BDCA6633CC7727A81`)
- **Architecture:** 2-layer stacked GRU, hidden size = 64, dropout = 0.2
- **Input Contract:** `[batch, 20, 6]` ($2.0\text{ s}$ duration at 10 Hz)
  - Channel 0: Longitudinal body acceleration ($g$)
  - Channel 1: Lateral body acceleration ($g$)
  - Channel 2: Vertical body acceleration ($g$)
  - Channel 3: Vehicle turning yaw rate ($\text{rad/s}$, mapped from phone pitch)
  - Channel 4: Vehicle roll rate ($\text{rad/s}$)
  - Channel 5: Vehicle pitch rate ($\text{rad/s}$)
- **Output Contract:** `[batch, 3]` $\to$ $[\Delta x, \Delta y, \Delta z]$ local initial frame displacement in meters
- **Inference Latency:** Mean $0.177\text{ ms}$, Median $0.159\text{ ms}$, P95 $0.232\text{ ms}$ ($>5,600\text{ inf/sec}$ on CPU)
- **Numerical Parity (PyTorch vs ONNX Runtime):** Max diff $1.192 \times 10^{-6}\text{ m}$, RMSE $1.546 \times 10^{-7}\text{ m}$

---

## 4. Evaluated Navigation Baselines & Outage Durations

- **Outage Durations:** 10.0s, 30.0s, 60.0s, 120.0s
- **Valid Sequence/Duration Combinations:** Exactly 32 physically valid intervals across the 9 test drives
- **Evaluated Baselines (All 7 Present in Authoritative CSVs):**
  1. **B1: Pure INS** (Double integration of body accelerometers transformed via DCM)
  2. **B2: INS + EKF** (6-state EKF integrating inertial kinematics)
  3. **B3: ML Only** (IO-VNBD Native 10 Hz GRU dead reckoning)
  4. **B4: ML + INS** (Blended kinematically weighted displacement)
  5. **B5: ML + INS + EKF** (Filter fusion without pseudo-measurements)
  6. **B6: ML + INS + EKF + NHC** (Non-Holonomic Constraints on lateral & vertical velocity)
  7. **B7: ML + INS + EKF + NHC + ZUPT** (NHC + Zero-Velocity Updates when stationary)
- **Total Evaluations:** $32 \times 7 = \mathbf{224 \text{ evaluations}}$

---

## 5. Frozen Test & Navigation Metrics

### A. Machine Learning Displacement Metrics (27,964 Held-Out Test Windows)
From `ml_test_results.csv`:
- **Zero Baseline:** Overall MAE $8.72\text{ m}$ | Overall RMSE $17.55\text{ m}$ | Mean Euclidean Error $25.34\text{ m}$ | P95 $53.80\text{ m}$
- **OxIOD GRU:** Overall MAE $8.88\text{ m}$ | Overall RMSE $17.16\text{ m}$ | Mean Euclidean Error $24.77\text{ m}$ | P95 $52.96\text{ m}$
- **Native IO-VNBD GRU:** Overall MAE **$4.01\text{ m}$** | Overall RMSE **$8.04\text{ m}$** | Mean Euclidean Error **$11.28\text{ m}$** | P95 **$26.65\text{ m}$**
- **Error Reduction:** **54.45% reduction** over OxIOD; **55.49% reduction** over Zero baseline.
- **Sequence Improvement:** IO-VNBD GRU improves over both Zero baseline and OxIOD on **8 of 9 test drives** (only `vw15` parked drive favored zero baseline).

---

### B. Frozen Navigation Benchmark Aggregate Metrics (All 32 Outage Combinations)
From `real_benchmark_aggregate.csv`:

| Duration | Baseline | N | Mean Drift (%) | Median Drift (%) | P95 Drift (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10s** | B1 Pure INS | 9 | 41.36% | 19.09% | 147.16% | 4.87m |
| **10s** | B2 INS + EKF | 9 | 23.70% | **12.99%** | 79.01% | **3.16m** |
| **10s** | B3 ML Only | 9 | 375.77% | 35.42% | 1547.39% | 14.17m |
| **10s** | B4 ML + INS | 9 | 323.93% | 33.56% | 1331.27% | 12.07m |
| **10s** | B5 ML + INS + EKF | 9 | 375.77% | 35.42% | 1547.39% | 14.17m |
| **10s** | B6 ML+INS+EKF+NHC | 9 | 373.88% | 36.19% | 1546.53% | 13.18m |
| **10s** | B7 ML+INS+EKF+NHC+ZUPT | 9 | 267.04% | 36.19% | 1066.05% | 7.66m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **30s** | B1 Pure INS | 8 | 244.84% | 63.98% | 864.30% | 35.36m |
| **30s** | B2 INS + EKF | 8 | 132.96% | **35.03%** | 460.51% | **28.07m** |
| **30s** | B3 ML Only | 8 | 1348.34% | 50.82% | 5352.86% | 68.42m |
| **30s** | B4 ML + INS | 8 | 1179.14% | 46.42% | 4669.61% | 58.54m |
| **30s** | B5 ML + INS + EKF | 8 | 1348.34% | 50.82% | 5352.86% | 68.42m |
| **30s** | B6 ML+INS+EKF+NHC | 8 | 1340.42% | 49.61% | 5351.35% | 63.39m |
| **30s** | B7 ML+INS+EKF+NHC+ZUPT | 8 | 974.80% | 57.63% | 3880.56% | 64.79m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **60s** | B1 Pure INS | 8 | 54.80% | 42.53% | 105.71% | 115.00m |
| **60s** | B2 INS + EKF | 8 | 32.21% | **29.19%** | 53.37% | **76.68m** |
| **60s** | B3 ML Only | 8 | 500.53% | 30.94% | 2333.15% | 122.87m |
| **60s** | B4 ML + INS | 8 | 430.69% | 29.38% | 1992.17% | 121.65m |
| **60s** | B5 ML + INS + EKF | 8 | 500.53% | 30.94% | 2333.15% | 122.87m |
| **60s** | B6 ML+INS+EKF+NHC | 8 | 543.56% | 62.97% | 2411.70% | 175.89m |
| **60s** | B7 ML+INS+EKF+NHC+ZUPT | 8 | 367.47% | 52.06% | 1658.00% | 158.98m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **120s** | B1 Pure INS | 7 | 131.05% | 93.72% | 335.15% | 308.21m |
| **120s** | B2 INS + EKF | 7 | 83.11% | 85.97% | 163.93% | 339.11m |
| **120s** | B3 ML Only | 7 | 391.57% | 32.59% | 1804.19% | **219.72m** |
| **120s** | B4 ML + INS | 7 | 328.51% | **25.04%** | 1512.23% | 230.37m |
| **120s** | B5 ML + INS + EKF | 7 | 391.57% | 32.59% | 1804.19% | **219.72m** |
| **120s** | B6 ML+INS+EKF+NHC | 7 | 265.88% | 52.88% | 1104.95% | 466.71m |
| **120s** | B7 ML+INS+EKF+NHC+ZUPT | 7 | 291.45% | 52.88% | 1281.61% | 341.51m |

---

### C. Frozen Navigation Benchmark Moving Sequences (Distance $>100.0\text{ m}$)
From `real_benchmark_aggregate_moving.csv`:

| Duration | Baseline | N | Mean Drift (%) | Median Drift (%) | P95 Drift (%) | Median RMSE (m) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **10s** | B1 Pure INS | 6 | 11.43% | 8.60% | 22.12% | 5.88m |
| **10s** | B2 INS + EKF | 6 | **7.79%** | **6.92%** | 14.04% | **4.96m** |
| **10s** | B3 ML Only | 6 | 26.14% | 24.50% | 54.96% | 24.84m |
| **10s** | B4 ML + INS | 6 | 23.00% | 20.48% | 46.83% | 21.05m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **30s** | B1 Pure INS | 5 | 34.71% | 16.08% | 65.61% | 56.87m |
| **30s** | B2 INS + EKF | 5 | **25.70%** | **14.04%** | 54.66% | **40.77m** |
| **30s** | B3 ML Only | 5 | 33.87% | 29.37% | 61.08% | 90.57m |
| **30s** | B4 ML + INS | 5 | 32.22% | 27.76% | 53.92% | 79.87m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **60s** | B1 Pure INS | 6 | 46.09% | 40.16% | 73.28% | 178.59m |
| **60s** | B2 INS + EKF | 6 | **32.32%** | 28.93% | 53.47% | **93.86m** |
| **60s** | B3 ML Only | 6 | 36.44% | **24.41%** | 86.12% | 163.26m |
| **60s** | B4 ML + INS | 6 | 35.20% | 26.16% | 77.65% | 150.99m |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **120s** | B1 Pure INS | 6 | 83.40% | 86.31% | 135.19% | 512.78m |
| **120s** | B2 INS + EKF | 6 | 64.21% | 81.16% | 87.50% | 479.65m |
| **120s** | B3 ML Only | 6 | 32.46% | 29.19% | 63.90% | **251.22m** |
| **120s** | B4 ML + INS | 6 | **28.62%** | **22.22%** | 63.74% | 252.10m |

---

### D. Threshold Breakdown (B3, B4, B5)
- **All Evaluations (N = 32):**
  - **B3 (ML Only):** $<10\%$: 4 (12.5%) | $<15\%$: 5 (15.6%) | $<20\%$: 6 (18.8%) | Median: 34.57%
  - **B4 (ML + INS):** $<10\%$: 2 (6.2%) | $<15\%$: 5 (15.6%) | $<20\%$: 8 (25.0%) | Median: 29.51%
  - **B5 (ML + INS + EKF):** $<10\%$: 4 (12.5%) | $<15\%$: 5 (15.6%) | $<20\%$: 6 (18.8%) | Median: 34.57%
- **Moving Evaluations Only (N = 23, Distance > 100m):**
  - **B3 (ML Only):** $<10\%$: 4 (17.4%) | $<15\%$: 5 (21.7%) | $<20\%$: 6 (26.1%) | Median: 25.79%
  - **B4 (ML + INS):** $<10\%$: 2 (8.7%) | $<15\%$: 5 (21.7%) | $<20\%$: 8 (34.8%) | Median: 24.80%
  - **B5 (ML + INS + EKF):** $<10\%$: 4 (17.4%) | $<15\%$: 5 (21.7%) | $<20\%$: 6 (26.1%) | Median: 25.79%

---

### E. 120s Outage Range on Moving Sequences
- **Best Moving Drive:** `vw16a` (Driver E) $\to$ **9.10% drift** ($161.70\text{ m}$ error over $1,776.3\text{ m}$)
- **Worst Moving Drive:** `vw14c` (Driver E) $\to$ **72.77% drift** ($343.28\text{ m}$ error over $471.7\text{ m}$)
- **Median Moving Drift:** **29.19%**
- **Mean Moving Drift:** **32.46%**
- **P95 Moving Drift:** **63.90%**

---

## 6. Authoritative Evaluation Files & SHA-256 Hashes

The following 18 files constitute the complete, authoritative, and frozen experimental record for the real IO-VNBD evaluation under `results/io_vnbd/`:

| File Name | SHA-256 Hash | Size (Bytes) | Category |
| :--- | :--- | :---: | :--- |
| `dataset_inventory.csv` | `724406C28F42551A88219859AB827F20F32C1B07C2755029C53FE20BA904A696` | 48,286 | Raw Data Audit |
| `sampling_rate_audit.csv` | `30078209545E5203DFDF5A36239DA74094E4696D62492B5224E864C6A62AA077` | 13,442 | Sensor Audit |
| `synchronization_audit.csv` | `4D0F7E15AE16A07634C512AAFC5AC4713731E12FD5CB2E2A19411A51D9375954` | 5,089 | Timestamp Alignment |
| `sensor_schema.md` | `0662A28CB627F3047366489E76AED51E68963E4057B3EDA0E63F22826F599894` | 6,045 | Coordinate Alignment |
| `ground_truth_definition.md` | `9E1BED6E40A6463C667B80BDAFA6B7889017B4894DB9DE6CA071ABF1D777FE27` | 3,515 | Reference Definition |
| `pretraining_data_audit.md` | `3104338D84F7FECE03D11E353C0105234B6CC75961DCACBC81CED56EE25B051B` | 6,048 | Weight Provenance |
| `split_manifest.csv` | `1D1412A76F7D49C750F84C7A9BA21011E7EBB5D605B83C878045B3F01FB7B28A` | 3,675 | Dataset Partitioning |
| `ml_test_results.csv` | `163BBE7CDFB0D34828A5F36976F82D19544DCF143DFC34E8947EB4CCCAB1A008` | 863 | Model Evaluation |
| `model_comparison_by_sequence.csv` | `836CDD80BE96E56F74E6704C78F459C29DF9D75A352E3CF8DD49E92C873D5256` | 1,854 | Sequence-Level Comparison |
| `onnx_parity_latency.json` | `4146B44BDD441670D8E5C24F9BF34BCDF6897D8E2F67C8412D7B40896074311D` | 467 | ONNX Parity & Latency |
| `real_benchmark_results.csv` | `7C1B6D1BBD51E7A2CE7DAEF33B71CE297A9D860958E81A6FAC761332CADE273D` | 7,561 | Single Sequence Deep Dive |
| `real_benchmark_summary.csv` | `268EB52886134A4CEBC9DAEA774D892F3AFB299820EBA3B6D29D313F8F6F0400` | 5,252 | Single Sequence Summary |
| `real_benchmark_all_test_sequences.csv` | `18E4745E3BB0601E033C81AF2689540EADADF847C0A7C1CF3247A4A6FE9674AD` | 53,691 | 224-Evaluation Raw Data |
| `real_benchmark_aggregate.csv` | `40D93C513545ADB840F1078A063AB18218A6E67F6D30E03F9E69ECCFF9173C5B` | 7,350 | All-Evaluation Aggregate |
| `real_benchmark_aggregate_moving.csv` | `3738464F47B45640F79D8073430C05A46C90D2A06A20122DA679597D4F36FF7F` | 7,318 | Moving-Only Aggregate |
| `final_leakage_audit.md` | `FACE8DFF01B5F6D82D3DDBD189F7157C538267B4909E00B630248FB0B57D7D3A` | 9,393 | Non-Leakage Audit |
| `final_validation_status.md` | `CA77D48B808341FC6A8A17EA5621016F82E9578B4F469E5AFADEE3068A4E09CE` | 18,125 | Final Validation Status |
| `FINAL_AUDIT_SUMMARY.md` | `F0D2BB5A4D99F02DE4E394ED038BB7C0CE7678198B0AE48C7CE32B514070A3F6` | 20,556 | Comprehensive Audit Summary |

---

## 7. Freeze Verification & Certification

Every evaluation file enumerated above has been checked against physical storage on `D:\dead_reckoning\results\io_vnbd\`. All hashes are verified. The model weights, code, and splits remain frozen and untouched.

**EVALUATION FREEZE: PASS**
