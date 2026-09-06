# Final ML Inference Contract: Edge Dead Reckoning Model

**Project:** Smart India Hackathon 2026 (SIH26168)  
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** Indian Space Research Organisation (ISRO)  
**Repository:** `D:\dead_reckoning`  
**Document Status:** **FINAL & FROZEN**  
**Effective Date:** September 2026  

---

## 1. Executive Contract Overview

This document specifies the exact, immutable technical interface between the AI/ML inference engine and the downstream Navigation Engine. Any edge implementation (Android ONNX Runtime, C++ embedded runtime, or Python reference pipeline) must strictly comply with the definitions, tensor layouts, unit transformations, and error handling protocols defined herein.

---

## 2. Authoritative Model Artifacts & File Paths

The production ML model has been audited, evaluated, and cryptographically frozen. It must never be retrained, fine-tuned, or re-exported from modified weights.

| Component | Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **ONNX Model (Primary Edge Artifact)** | `models/frozen_io_vnbd/gru_io_vnbd.onnx`<br>`models/gru_io_vnbd.onnx` | `32A512EB668205D2C358FB7D94BE84B24089112A2DC3264D9DCB6E323935FABF` |
| **PyTorch Checkpoint (Reference)** | `models/frozen_io_vnbd/gru_io_vnbd_best.pt`<br>`models/gru_io_vnbd_best.pt` | `0DD59052D14F24E507AFF096E21100E7CE1AC56C9734B6C59B3FE07AFF7815C0` |
| **Normalization Parameters (JSON)** | `models/frozen_io_vnbd/io_vnbd_normalization.json`<br>`models/io_vnbd_normalization.json` | `A0DF903FFAA4FA491992BCDF763A8B99AA8228D1CACEF7167CD91FC9C73831C6` |
| **Model Metadata (JSON)** | `models/frozen_io_vnbd/model_metadata.json`<br>`models/model_metadata.json` | `5F4030F27713369761CC15E70A8F9A414967C83B703ACF1BDCA6633CC7727A81` |

> [!CAUTION]
> **Immutability Clause:** The ONNX model must NOT be regenerated or replaced with altered weights. CI/CD and deployment packaging scripts must verify the SHA-256 hash before embedding the model into the Android APK.

---

## 3. Input Tensor Contract

### A. Dimensions & Data Type
- **Tensor Name:** `imu_window`
- **Tensor Shape:** `[1, 20, 6]` (Batch Size = 1, Sequence Length = 20 samples, Channels = 6)
- **Batch Processing Shape:** `[N, 20, 6]` for batch inference
- **Data Type:** `float32` (IEEE 754 32-bit floating point)

### B. Temporal Parameters
- **Sampling Frequency:** Exactly **10.0 Hz** ($\Delta t = 0.100\text{ s} \pm 0.015\text{ s}$)
- **Window Length:** Exactly **20 consecutive samples** ($2.0\text{ seconds}$ physical duration)
- **Update Stride:** **10 samples** ($1.0\text{ second}$ update interval, 50% temporal overlap)

### C. Channel Ordering & Units
Input features must be ordered strictly as follows:

| Index | Feature Name | Physical Semantic | Sensor Source | Raw Unit | Required Unit for Normalization |
| :---: | :---: | :--- | :--- | :---: | :---: |
| **0** | `acc_x` | Forward longitudinal linear acceleration | Android `TYPE_LINEAR_ACCELERATION` | $\text{m/s}^2$ | **$g$** ($a / 9.80665$) |
| **1** | `acc_y` | Lateral transverse linear acceleration | Android `TYPE_LINEAR_ACCELERATION` | $\text{m/s}^2$ | **$g$** ($a / 9.80665$) |
| **2** | `acc_z` | Vertical linear acceleration | Android `TYPE_LINEAR_ACCELERATION` | $\text{m/s}^2$ | **$g$** ($a / 9.80665$) |
| **3** | `gyro_x` | Vehicle turning yaw rate | Android `TYPE_GYROSCOPE` (Pitch axis in landscape) | $\text{rad/s}$ | **$\text{rad/s}$** |
| **4** | `gyro_y` | Vehicle roll rate | Android `TYPE_GYROSCOPE` (Roll axis in landscape) | $\text{rad/s}$ | **$\text{rad/s}$** |
| **5** | `gyro_z` | Vehicle pitch rate | Android `TYPE_GYROSCOPE` (Yaw axis in landscape) | $\text{rad/s}$ | **$\text{rad/s}$** |

> [!IMPORTANT]
> **Physical Sensor Unit Transformation:**
> - Android `TYPE_LINEAR_ACCELERATION` outputs acceleration in $\text{m/s}^2$ with gravity already subtracted by the Android sensor fusion HAL.
> - The model requires acceleration in units of gravity ($g$). The client code must divide raw $\text{m/s}^2$ values by $9.80665$:
>   $$a_g = \frac{a_{\text{m/s}^2}}{9.80665}$$
> - Android `TYPE_GYROSCOPE` outputs in $\text{rad/s}$, which directly matches the model requirement without scale conversion.

---

## 4. Exact Normalization & Denormalization Procedure

Feature normalization and target denormalization must strictly utilize the frozen parameters from `io_vnbd_normalization.json`.

### A. Input Feature Normalization
For each sample $i \in [0, 19]$ and channel $j \in [0, 5]$:
$$X_{\text{norm}}[0, i, j] = \frac{X[0, i, j] - \mu_X[j]}{\sigma_X[j]}$$

Where:
- $\boldsymbol{\mu}_X = [0.0027463287, -0.0129796378, 0.0025934221, 0.0010569283, -0.0013110938, -0.0013259545]$
- $\boldsymbol{\sigma}_X = [0.2192199081, 0.2064620256, 0.0930955634, 0.3022365868, 0.1888296902, 0.1109523773]$

### B. ONNX Inference Pass
$$y_{\text{norm}} = \text{ONNX\_Runtime\_Run}(X_{\text{norm}}) \quad \in \mathbb{R}^{1 \times 3}$$

### C. Output Target Denormalization
For each coordinate $k \in [0, 2]$:
$$\Delta p[k] = y_{\text{norm}}[k] \cdot \sigma_y[k] + \mu_y[k]$$

Where:
- $\boldsymbol{\mu}_y = [26.46901131, 0.01971586, 0.0]$
- $\boldsymbol{\sigma}_y = [16.95499420, 1.42004120, 1.0]$

---

## 5. Output Tensor Contract & Physical Semantics

- **Tensor Name:** `local_displacement`
- **Output Shape:** `[1, 3]`
- **Data Type:** `float32`
- **Physical Meaning:** Metric displacement vector $[\Delta x, \Delta y, \Delta z]^T$ traversed during the window interval:
  - $\Delta x$: Forward displacement in meters along the initial vehicle heading axis.
  - $\Delta y$: Lateral displacement in meters perpendicular to vehicle heading.
  - $\Delta z$: Vertical displacement in meters.

> [!CRITICAL]
> **NO Geodetic Output:**
> - The GRU neural network does **NOT** output latitude, longitude, altitude, or global WGS-84 coordinates.
> - The GRU does **NOT** maintain global position state across calls.
> - It outputs strictly a relative translational vector in the vehicle's body/local coordinate frame.

---

## 6. Architecture & System Boundary: ML vs Navigation Engine

```
[Smartphone IMU]
    │  Linear Accel (m/s²), Gyro (rad/s) at 10 Hz
    ▼
┌─────────────────────────────────────────────────────────────┐
│               ML INFERENCE LAYER (This Contract)            │
│  1. Unit conversion (m/s² -> g)                             │
│  2. Sliding buffer management (20 samples, stride 10)       │
│  3. Input Normalization using io_vnbd_normalization.json     │
│  4. ONNX Runtime forward pass (gru_io_vnbd.onnx)            │
│  5. Target Denormalization -> [dx, dy, dz] in meters        │
└──────────────────────────────┬──────────────────────────────┘
                               │  Local Frame Displacement [dx, dy, dz] (m)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 NAVIGATION ENGINE (Downstream)              │
│  1. Vehicle Attitude & Heading Estimation (AHRS / EKF)      │
│  2. Coordinate Rotation: dp_NED = R_bn(heading) * [dx,dy,dz]│
│  3. Non-Holonomic Constraints (NHC) enforcement             │
│  4. Zero-Velocity Updates (ZUPT) filtering when stopped     │
│  5. Metric-to-Geodetic Conversion -> (Latitude, Longitude)  │
│  6. Map-matching & UI rendering                             │
└─────────────────────────────────────────────────────────────┘
```

### Allocation of Responsibilities:
1. **ML Inference Layer Responsibility:**  
   Accurately estimate 3D translational displacement magnitude and direction in the vehicle's local body frame from raw high-rate inertial shock and vibration patterns.
2. **Navigation Engine Responsibility:**  
   - Determine vehicle orientation (heading $\psi$, pitch $\theta$, roll $\phi$) via gyro integration and attitude filtering.
   - Rotate body displacements into North-East-Down (NED) navigation coordinates:
     $$\begin{bmatrix} \Delta p_N \\ \Delta p_E \\ \Delta p_D \end{bmatrix} = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix}$$
   - Apply geodetic spherical/ellipsoidal projections to accumulate latitude and longitude:
     $$\Delta \text{lat} = \frac{\Delta p_N}{R_M + h}, \quad \Delta \text{lon} = \frac{\Delta p_E}{(R_N + h)\cos(\text{lat})}$$
   - Apply EKF fusion with GNSS when available, and pseudo-measurements (NHC, ZUPT) during GNSS outages.

---

## 7. Error Handling & Edge Failure Protocols

The ML inference wrapper must implement defensive input validation and deterministic failure recovery:

1. **Insufficient Sensor Samples:**
   - *Condition:* FIFO ring buffer contains fewer than 20 samples (e.g., initial startup).
   - *Action:* Suppress inference. Return a `Pending` status. Do NOT zero-pad or duplicate samples.
2. **Invalid / Out-of-Range Sensor Readings:**
   - *Condition:* $|a_g| > 8.0\text{ g}$ or $|\omega| > 20.0\text{ rad/s}$ (sensor clipping or hardware disconnect).
   - *Action:* Reject the sample, log a hardware warning, and hold the previous valid motion estimate.
3. **Non-Finite Values (`NaN` or `Inf`):**
   - *Condition:* Any element in the 20-sample window contains `NaN`, `+Inf`, or `-Inf`.
   - *Action:* Immediately reject the entire window. Return zero displacement $[0.0, 0.0, 0.0]^T$. Do NOT pass non-finite tensors to ONNX Runtime.
4. **ONNX Runtime Execution Exception:**
   - *Condition:* Native C++ inference crash, memory allocation failure, or session error.
   - *Action:* Catch exception, log error code, and seamlessly fall back to kinematic extrapolation using raw accelerometers or stationary zero-velocity update.

---

## 8. Offline & On-Device Operational Guarantee

- **Strict Offline Execution:** All ML inference must execute 100% locally on the device CPU using ONNX Runtime.
- **Zero Network Traffic:** The inference pipeline must have zero dependencies on cellular network connectivity, Wi-Fi, cloud inference APIs, remote servers, or external telemetry.
- **Deterministic Latency:** Single-window inference latency on modern smartphone ARM CPUs must remain $< 1.0\text{ ms}$ (benchmark measured $0.177\text{ ms}$ on standard x86/ARM hardware), providing $>1,000\times$ headroom above the 1.0 Hz step rate.
