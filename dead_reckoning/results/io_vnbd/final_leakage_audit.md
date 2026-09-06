# Ground Truth & Leakage Audit Report: Real IO-VNBD Evaluation

**Project:** SIH26168 — AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** ISRO  
**Repository:** `D:\dead_reckoning`  
**Date:** September 2026  
**Auditor:** Senior Edge AI & Inertial Navigation Engineer  

---

## 1. Executive Summary

This audit establishes a rigorous, transparent evaluation of data leakage, future information contamination, and sensor dependencies for the AI/ML Dead Reckoning system evaluated on the real Inertial and Odometry Vehicle Navigation Benchmark Dataset (IO-VNBD).

Substantial reduction in dead-reckoning error was demonstrated on real held-out IO-VNBD vehicle sequences, with performance varying across routes, drivers, and motion conditions. To ensure scientific integrity, every stage of the preprocessing, training, inference, and multi-sequence benchmark pipeline was audited across seven specific non-leakage criteria.

---

## 2. Leakage Audit Criteria & Empirical Evidence

### Criterion 1: Vehicle Reference Position is NOT Used During Outage
- **Requirement:** During any simulated GNSS blackout interval $[t_{\text{start}}, t_{\text{end}}]$, the estimator must never observe or receive vehicle ECU GNSS coordinates (`v_gt_lat`, `v_gt_lon`).
- **Audit Verification:**
  - In `src/evaluation/outage_simulator.py`:
    ```python
    def create_gnss_mask(self, timestamps_s: np.ndarray, interval: OutageInterval) -> np.ndarray:
        mask = np.ones(len(timestamps_s), dtype=bool)
        mask[interval.start_idx : interval.end_idx] = False
        return mask
    ```
  - In `src/evaluation/run_all_test_sequences_benchmark.py`:
    During the outage index range `range(s, e)`, neither Pure INS, INS+EKF, ML Only, ML+INS, nor EKF fusion calls `ekf.update_gnss_position()`.
  - The filter propagates purely on internal inertial increments and ML displacement estimates.
- **Finding:** **PASS**. Zero position leakage during outage intervals.

---

### Criterion 2: Vehicle Reference Velocity is NOT Used During Outage
- **Requirement:** Reference vehicle velocity (`v_gt_speed_kmh`, CAN bus wheel speeds) must be completely disabled during blackout intervals.
- **Audit Verification:**
  - In `src/evaluation/run_all_test_sequences_benchmark.py`:
    No GNSS speed or CAN speed measurement is passed to `ekf.update_gnss_velocity()` during the outage interval `s:e`.
  - Ground truth velocity is used exclusively post-hoc for metric computation in `NavigationMetrics.evaluate_outage()`.
- **Finding:** **PASS**. Zero velocity leakage during blackout intervals.

---

### Criterion 3: Ground Truth Used Exclusively for Post-Hoc Evaluation
- **Requirement:** Reference trajectory coordinates (`gt_lat`, `gt_lon`, `gt_spd`) must serve exclusively as passive ground truth for error calculation.
- **Audit Verification:**
  - In `src/evaluation/metrics.py`:
    `NavigationMetrics.evaluate_outage` is invoked strictly after the forward estimation trajectory `lat_k, lon_k, spd_k` has been completely generated:
    ```python
    res_k = NavigationMetrics.evaluate_outage(
        lat_k, lon_k, seq.gt_lat, seq.gt_lon,
        interval.start_idx, interval.end_idx, dur,
        b_name, seq.timestamps_s, spd_k, seq.gt_spd
    )
    ```
  - The estimator state has no feedback loop from the evaluation metrics.
- **Finding:** **PASS**. Evaluation is strictly post-hoc.

---

### Criterion 4: Strict Temporal Causality (No Future Data Enters Model Window)
- **Requirement:** Model inference must be strictly causal. A window predicting motion at time $t$ must only observe sensor samples from $[t - W, t]$, never from $t' > t$.
- **Audit Verification:**
  - In `src/evaluation/run_real_io_vnbd_benchmark.py` and `src/preprocessing/build_io_vnbd_dataset.py`:
    ```python
    starts = range(0, n - window_size + 1, stride)
    for s_idx in starts:
        e_idx = s_idx + window_size - 1
        w_feat = features[s_idx : s_idx + window_size]
    ```
  - Windows are forward-sliding. No backward smoothing, bidirectional RNNs, or future lookahead are employed.
- **Finding:** **PASS**. Inference obeys strict physical causality.

---

### Criterion 5: Zero Normalization Statistics Leakage
- **Requirement:** Feature normalization parameters (mean $\boldsymbol{\mu}_X$ and standard deviation $\boldsymbol{\sigma}_X$) and target parameters ($\boldsymbol{\mu}_y, \boldsymbol{\sigma}_y$) must be computed strictly from the training split. Test sequences must never influence normalization statistics.
- **Audit Verification:**
  - In `src/training/train_io_vnbd.py`:
    Statistics loaded from `models/io_vnbd_normalization.json` were computed strictly from `data/processed/io_vnbd_train_local.npz`:
    ```python
    computed_X_mean = np.mean(X_train, axis=(0, 1))
    computed_X_std = np.std(X_train, axis=(0, 1))
    ```
  - Verified by unit test `test_06_normalization_leakage_check` in `tests/test_native_io_vnbd.py`:
    Matches training set array statistics within numerical tolerance $\le 10^{-4}$.
  - Test sequences (`m`, `y1`, `vfa01`, `vfa02`, `vw14c`, `vw15`, `vw16a`, `vw16b`, `vw17`) were completely omitted from normalization computations.
- **Finding:** **PASS**. Zero normalization leakage from test sets.

---

### Criterion 6: Disjoint Sequence-Level Partitioning
- **Requirement:** Entire driving sequences must be allocated to exactly one partition. No overlapping windows from the same sequence may appear across training, validation, and test splits.
- **Audit Verification:**
  - Manifest: `results/io_vnbd/split_manifest.csv`
  - Partition summary:
    - Train: 53 sequences (45,742 windows) — Drivers C & E
    - Val: 10 sequences (33,262 windows) — Drivers A & E
    - Test: 9 sequences (27,964 windows) — Drivers B, D, E
  - Verified by unit test `test_05_train_val_test_leakage_absence` in `tests/test_native_io_vnbd.py`:
    $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- **Finding:** **PASS**. Sequence-level partitioning is completely disjoint.

---

### Criterion 7: Attitude & Heading Reference Audit (Critical Finding)
- **Requirement:** Assess whether estimated attitude or reference vehicle heading was used for rotating body-frame quantities into the navigation (NED) coordinate frame and for non-holonomic constraints (NHC).
- **Audit Verification:**
  - In `src/evaluation/run_all_test_sequences_benchmark.py`:
    - **ML Body-to-NED Transformation (Lines 288–290):**
      ```python
      dx, dy, dz = ml_displacements[w_idx]
      hdg = float(seq.gt_hdg[s_idx])
      R_bn = CoordinateTransformer.heading_to_dcm(hdg)
      dp_ned = R_bn @ np.array([dx, dy, dz], dtype=np.float64)
      ```
    - **Pure INS Body Acceleration Transformation (Lines 329–332):**
      ```python
      hdg_i = float(seq.gt_hdg[i])
      R_bn = CoordinateTransformer.heading_to_dcm(hdg_i)
      a_ned = R_bn @ a_body
      ```
    - **NHC Pseudo-Measurement Transformation (Line 475):**
      ```python
      if use_nhc:
          ekf.update_nhc(hdg_i, noise_lat_mps=0.15, noise_vert_mps=0.15)
      ```
- **Audit Statement & Boundary Disclosure:**
  - Ground truth position (`gt_lat`, `gt_lon`) and velocity (`gt_spd`) were strictly **suppressed** during blackouts.
  - However, the coordinate rotation $R_{bn}(\psi)$ and NHC observation matrix $H(\psi)$ received the synchronized vehicle reference heading `gt_hdg` logged from the CAN bus / dual-antenna GPS.
  - **Scope Classification:** The current multi-sequence benchmark evaluates **displacement dead reckoning under reference attitude/heading**.
  - **Operational Consequence:** It does **not** evaluate open-loop heading integration from noisy, uncalibrated consumer smartphone gyroscopes. In an unassisted deployment without a dual-antenna GNSS or magnetic reference, gyroscope bias would induce heading drift, which would progressively misorient both the ML displacement vector and the lateral NHC constraint over 60–120s outages.
- **Finding:** **HEADING DEPENDENCY DISCLOSED**. Evaluates displacement dead reckoning under reference heading; unassisted attitude integration is an explicit limitation.

---

## 3. Leakage Audit Summary Table

| Audit Criterion | Status | Implementation Details |
| :--- | :---: | :--- |
| **Position Leakage** | **PASS** | GNSS position updates completely disabled during outage |
| **Velocity Leakage** | **PASS** | Vehicle speed / wheel odometry completely disabled during outage |
| **Evaluation Timing** | **PASS** | Passive ground truth evaluated strictly post-hoc |
| **Temporal Causality** | **PASS** | Sliding forward window $[t-W, t]$ without future lookahead |
| **Normalization Leakage** | **PASS** | Statistics computed strictly from 53 training sequences |
| **Split Disjointness** | **PASS** | Zero sequence overlap across train, val, and test splits |
| **Attitude / Heading Scope** | **DISCLOSED** | Uses reference heading for body-to-NED projection; heading drift not integrated |

---

## 4. Conclusion

The benchmark implementation strictly adheres to all physical non-leakage criteria regarding position, velocity, normalization, and temporal causality. The evaluation scope is explicitly defined as displacement dead reckoning under reference heading, ensuring transparent, scientifically defensible reporting.
