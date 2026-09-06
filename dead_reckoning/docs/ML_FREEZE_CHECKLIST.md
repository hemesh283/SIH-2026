# ML Layer Freeze & Verification Checklist

**Project:** Smart India Hackathon 2026 (SIH26168)  
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** Indian Space Research Organisation (ISRO)  
**Repository:** `D:\dead_reckoning`  
**Freeze Status:** **VERIFIED & SEALED**  
**Audit Date:** September 2026  

---

## 1. Cryptographic Artifact Identity Checklist

| Component | Target Path | Frozen Archive Path | SHA-256 Hash | Status |
| :--- | :--- | :--- | :--- | :---: |
| **ONNX Model** | `models/gru_io_vnbd.onnx` | `models/frozen_io_vnbd/gru_io_vnbd.onnx` | `32A512EB668205D2C358FB7D94BE84B24089112A2DC3264D9DCB6E323935FABF` | **MATCH** |
| **PyTorch Checkpoint** | `models/gru_io_vnbd_best.pt` | `models/frozen_io_vnbd/gru_io_vnbd_best.pt` | `0DD59052D14F24E507AFF096E21100E7CE1AC56C9734B6C59B3FE07AFF7815C0` | **MATCH** |
| **Normalization JSON** | `models/io_vnbd_normalization.json` | `models/frozen_io_vnbd/io_vnbd_normalization.json` | `A0DF903FFAA4FA491992BCDF763A8B99AA8228D1CACEF7167CD91FC9C73831C6` | **MATCH** |
| **Model Metadata** | `models/model_metadata.json` | `models/frozen_io_vnbd/model_metadata.json` | `5F4030F27713369761CC15E70A8F9A414967C83B703ACF1BDCA6633CC7727A81` | **MATCH** |

- [x] All production models match their frozen archives bit-for-bit.
- [x] No weights, biases, or checkpoint metadata have been modified.

---

## 2. Preprocessing & Temporal Contract Checklist

- [x] **Native 10 Hz Sampling:** Sensor pipeline operates strictly at 10.0 Hz ($\Delta t = 0.100\text{ s}$).
- [x] **No Sample Fabrication:** Artificial 100 Hz spline interpolation is completely eliminated.
- [x] **Window Length:** Exactly 20 consecutive samples ($2.0\text{ seconds}$ physical duration).
- [x] **Update Stride:** Exactly 10 samples ($1.0\text{ second}$ update interval, 50% temporal overlap).
- [x] **Temporal Causality:** Strict forward-sliding buffer without future lookahead or backward smoothing.

---

## 3. Physical Units & Channel Ordering Checklist

- [x] **Linear Acceleration Units:** Android `Sensor.TYPE_LINEAR_ACCELERATION` ($\text{m/s}^2$) converted to units of gravity ($g$) via division by $9.80665$:
  $$a_g = \frac{a_{\text{m/s}^2}}{9.80665}$$
- [x] **Angular Velocity Units:** Android `Sensor.TYPE_GYROSCOPE` in $\text{rad/s}$ (no scale factor required).
- [x] **Physical Mounting Alignment:** Phone dashboard landscape orientation established; phone Pitch rate is aligned with vehicle turning yaw rate ($r = 0.9348$ correlation).
- [x] **Input Channel Sequence:** Strictly ordered as:
  1. `acc_x` (Longitudinal body acceleration in $g$)
  2. `acc_y` (Lateral body acceleration in $g$)
  3. `acc_z` (Vertical body acceleration in $g$)
  4. `gyro_x` (Vehicle turning yaw rate in $\text{rad/s}$)
  5. `gyro_y` (Vehicle roll rate in $\text{rad/s}$)
  6. `gyro_z` (Vehicle pitch rate in $\text{rad/s}$)

---

## 4. Tensor Dimensions & Normalization Checklist

- [x] **Input Tensor Dimension:** `[1, 20, 6]` (`[batch, sequence_length, channels]`).
- [x] **Output Tensor Dimension:** `[1, 3]` (`[batch, 3]`).
- [x] **Feature Normalization Equation:** $X_{\text{norm}}[i, j] = (X[i, j] - \mu_X[j]) / \sigma_X[j]$.
- [x] **Target Denormalization Equation:** $\Delta p[k] = y_{\text{norm}}[k] \cdot \sigma_y[k] + \mu_y[k]$.
- [x] **Zero Test Set Leakage:** Parameters ($\boldsymbol{\mu}_X, \boldsymbol{\sigma}_X, \boldsymbol{\mu}_y, \boldsymbol{\sigma}_y$) computed exclusively from the 53 training drives.

---

## 5. Output Semantics & Architectural Boundary Checklist

- [x] **Physical Meaning:** $[\Delta x, \Delta y, \Delta z]^T$ represents 3D translational displacement in meters within the vehicle's initial body frame over the window.
- [x] **No Geodetic Output:** Confirmed that the neural network does **NOT** output latitude, longitude, altitude, or global coordinates.
- [x] **Navigation Engine Separation:** Confirmed that downstream Navigation Engine is strictly responsible for:
  - Heading & attitude determination ($\psi$).
  - Coordinate rotation: $\Delta \mathbf{p}_{\text{NED}} = R_{bn}(\psi) [\Delta x, \Delta y, \Delta z]^T$.
  - Metric-to-geodetic integration into (Latitude, Longitude).
  - Application of pseudo-measurements (NHC, ZUPT) and EKF filtering.

---

## 6. Edge Operational & Error Handling Checklist

- [x] **100% Offline Execution:** Confirmed zero network calls, external API dependencies, or cloud services.
- [x] **Deterministic Latency:** Single-window inference executes in $0.177\text{ ms}$ on CPU ($>5,600\text{ inf/s}$).
- [x] **Underfill Handling:** Inference suppressed when buffer contains $<20$ samples.
- [x] **Non-Finite Value Handling:** Windows containing `NaN` or `Inf` are rejected before forward pass.
- [x] **Runtime Exception Fallback:** Inference failure catches exceptions and falls back to kinematic extrapolation.

---

## 7. Evaluation Freeze Consistency Checklist

- [x] **Split Isolation:** 53 train, 10 val, 9 test sequences with zero drive overlap.
- [x] **Outage Protocol:** 10s, 30s, 60s, 120s outages with zero GNSS position or velocity leakage during blackouts.
- [x] **7 Baselines Verified:** B1 Pure INS through B7 ML+INS+EKF+NHC+ZUPT present in all authoritative result files.
- [x] **224 Evaluation Runs:** Verified in `results/io_vnbd/real_benchmark_all_test_sequences.csv`.
- [x] **Evaluation Manifest Sealed:** Documented in `results/io_vnbd/EVALUATION_FREEZE_MANIFEST.md` (SHA-256: `CAC61661DD647CE0EABB3844F1F5E2071C62ACEF2E767B3E47667FA1A3FA1BD5`).

---

## 8. Final Freeze Status

**ML LAYER FREEZE: PASS (100% VERIFIED)**
