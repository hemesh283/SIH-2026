# Dead Reckoning ML Module: Android Integration Specification

This document provides the definitive integration specification for the Dead Reckoning machine learning deployment module. It is intended for mobile and application engineers integrating `DeadReckoningInference` (Python) or `gru_local.onnx` (ONNX Runtime Mobile / C++ / Java / Kotlin).

---

## 1. Executive Contract Overview

The ML model predicts **3D local displacement** over a **2.0-second fixed window** from 6-axis IMU sensor streams.

- **Input Tensor Shape:** `[1, 200, 6]` (batch=1, timesteps=200, features=6)
- **Output Tensor Shape:** `[3]` (or `[1, 3]` in batch mode)
- **Output Values:** `[dx, dy, dz]` in **meters**
- **Output Coordinate Frame:** Local body frame at the start of the 2-second window ($R_{start}^T \Delta p$).

---

## 2. Detailed Technical Specifications

### A. What the Android App Must Provide
The Android client is responsible for capturing device motion, converting sensor readings into model-compatible units, buffering samples, and maintaining a 200-sample sliding window:
1. Linear acceleration along 3 device axes (X, Y, Z) with gravity removed.
2. Calibrated angular velocity (gyroscope) along 3 device axes (X, Y, Z).
3. Uniform sampling at **100 Hz** (10 ms per sample).
4. An optional orientation source (e.g. `Sensor.TYPE_ROTATION_VECTOR`) if the application needs to integrate local step displacements into global world coordinates.

### B. Exact Six-Feature Order
The input tensor's feature dimension (dim 2) must **strictly** adhere to this column sequence:

| Index | Feature Name | Description | Source / Sensor |
|:---:|:---:|:---|:---|
| **0** | `acc_x` | Linear acceleration along device X-axis (gravity removed) | `Sensor.TYPE_LINEAR_ACCELERATION` (X) |
| **1** | `acc_y` | Linear acceleration along device Y-axis (gravity removed) | `Sensor.TYPE_LINEAR_ACCELERATION` (Y) |
| **2** | `acc_z` | Linear acceleration along device Z-axis (gravity removed) | `Sensor.TYPE_LINEAR_ACCELERATION` (Z) |
| **3** | `gyro_x` | Angular velocity around device X-axis | `Sensor.TYPE_GYROSCOPE` (X) |
| **4** | `gyro_y` | Angular velocity around device Y-axis | `Sensor.TYPE_GYROSCOPE` (Y) |
| **5** | `gyro_z` | Angular velocity around device Z-axis | `Sensor.TYPE_GYROSCOPE` (Z) |

### C. Expected Units & Conversions
> [!CAUTION]
> **CRITICAL SENSOR UNIT CONFLICT:**
> - Android `Sensor.TYPE_LINEAR_ACCELERATION` reports values in **$\text{m/s}^2$**.
> - The model was trained on OxIOD iOS data where linear acceleration is reported in **$g$** ($1\text{ g} = 9.80665\text{ m/s}^2$).
> - **Failure to convert $\text{m/s}^2 \rightarrow g$ will scale accelerometer inputs by $\approx 9.8\times$, invalidating all displacement predictions.**

1. **Acceleration Conversion:**
   $$a_{\text{model}} [g] = \frac{a_{\text{Android}} [\text{m/s}^2]}{9.80665}$$
2. **Gyroscope Units:**
   - Android `Sensor.TYPE_GYROSCOPE` outputs in **$\text{rad/s}$**.
   - Model expects **$\text{rad/s}$**.
   - **No scaling required.**

### D. Expected Sampling Frequency
- **Nominal Frequency:** **$100\text{ Hz}$** ($\Delta t = 0.010\text{ seconds}$).
- **Sampling Interval:** Exactly $10\text{ ms}$ between consecutive samples.
- **Android Behavior:** Android sensor event callbacks deliver events with irregular timestamp jitter and device-specific rates (e.g., `SENSOR_DELAY_GAME` can fluctuate between 50 Hz and 200 Hz).
- **Client Requirement:** The Android client **must resample/interpolate** sensor readings (e.g., via linear interpolation or zero-order hold onto a uniform 100 Hz grid) before feeding the 200-sample buffer.

### E. Required Preprocessing
1. **Gravity Removal:** Use Android `Sensor.TYPE_LINEAR_ACCELERATION` (which uses hardware sensor fusion) or an onboard high-pass / Kalman gravity filter. Do **not** pass raw accelerometer readings (`Sensor.TYPE_ACCELEROMETER`), as they contain the $1\text{ g}$ gravity vector.
2. **Resampling:** Uniform 100 Hz grid interpolation.
3. **Unit Scaling:** Divide linear acceleration components by $9.80665$.
4. **Buffering:** Maintain a FIFO buffer of 200 samples.

### F. Required Window Size
- **Window Length:** Exactly **200 samples** ($200 \times 10\text{ ms} = 2.0\text{ seconds}$).
- **Stride:** 
  - Non-overlapping execution: Stride = 200 samples (runs once every 2 seconds).
  - Overlapping execution: Stride = 50 or 100 samples (runs every 0.5s or 1.0s for higher-frequency position updates).

### G. Normalization Procedure
The neural network requires input features to be standardized using the fixed training set statistics.

Given raw window sample $x \in \mathbb{R}^6$:
$$x_{\text{normalized}} = \frac{x - \mu_X}{\sigma_X}$$

**Exact Training Statistics:**
```json
{
  "mean": [
    0.006345765665173531,
    0.020579669624567032,
    -0.008580482564866543,
    0.0027845546137541533,
    0.04215956851840019,
    0.08918707817792892
  ],
  "std": [
    0.07353448122739792,
    0.09233938157558441,
    0.09523431211709976,
    0.31493133306503296,
    0.36222153902053833,
    0.5868740081787109
  ]
}
```

The raw network output $\hat{y}_{\text{raw}} \in \mathbb{R}^3$ is denormalized to meters:
$$\text{displacement}_{\text{meters}} = \hat{y}_{\text{raw}} \cdot \sigma_y + \mu_y$$

**Exact Target Denormalization Statistics:**
```json
{
  "target_mean": [0.6047252416610718, -1.032145619392395, -0.47675254940986633],
  "target_std": [0.7513378262519836, 0.4383731782436371, 0.236590176820755]
}
```
*(Note: `DeadReckoningInference` automatically handles both normalization and denormalization. If calling ONNX directly in Android, apply these two transformations).*

### H. Model Input Tensor Shape
- **ONNX Model Input Name:** `imu_window`
- **Shape:** `[batch_size, 200, 6]` (batch size is dynamic; default is 1).
- **Data Type:** Float32.

### I. Model Output Shape
- **ONNX Model Output Name:** `local_displacement`
- **Shape:** `[batch_size, 3]`
- **Interpretation:** `[dx, dy, dz]` in meters over the 2-second interval.

---

## 3. Orientation Dependency & Global Trajectory Integration

### A. Separation of Training vs. Deployment
- **During Training:** OxIOD used an external Vicon optical motion capture system to determine initial orientation $R_{\text{start}}$ and compute ground-truth local displacement targets:
  $$y = R_{\text{start}}^T \cdot (p_{\text{end}} - p_{\text{start}})$$
- **During Inference:** The GRU model **does NOT take orientation as an input**. It maps 6-DOF IMU dynamics directly to local body displacement $\mathbf{\Delta p}_{\text{local}} = [dx, dy, dz]^T$.
- **During Trajectory Tracking (Deployment):** To construct a global room or map trajectory ($P_k \in \mathbb{R}^3$), the Android app must rotate local displacement into the world frame using an on-device orientation estimator:
  $$P_k = P_{k-1} + R_{k-1} \cdot \begin{bmatrix} dx \\ dy \\ dz \end{bmatrix}_k$$
  where $R_{k-1}$ is the $3 \times 3$ rotation matrix obtained from Android's `SensorManager.getRotationMatrixFromVector(...)` using `Sensor.TYPE_ROTATION_VECTOR` or `Sensor.TYPE_GAME_ROTATION_VECTOR`.

---

## 4. Operational Limitations & Out-of-Scope Elements

### J. What the ML Module Does NOT Handle
1. **Sensor Event Polling:** The ML module does not connect to the Android hardware sensor bus.
2. **Frequency Resampling / Timestamp Synchronization:** The ML module assumes a steady 100 Hz stream. It does not interpolate missing timestamps.
3. **Gravity Separation:** If raw accelerometer data containing gravity is passed, predictions will be invalid.
4. **Global Trajectory Integration:** The model outputs step displacement vectors; the host application must perform dead-reckoning vector accumulation ($P_k = P_{k-1} + R \Delta p$).
5. **Zero-Velocity Update (ZUPT):** The model does not explicitly clamp stationary drift; an external stationary detector is recommended to suppress drift when the user is not moving.

### K. Known Deployment Limitations
- **Motion Domain:** The model was trained specifically on **handheld** phone recordings (users holding a smartphone while walking). Performance on pocket, handbag, running, or vehicle mounts has not yet been benchmarked.
- **Sampling Frequency Strictness:** Deviations from 100 Hz (e.g., passing 50 Hz data) will effectively stretch the apparent temporal dynamics, corrupting displacement magnitude.
- **Drift Accumulation:** Open-loop dead reckoning inherently accumulates positional drift over time without external anchors (such as GPS, Wi-Fi RTT, or visual odometry).
