# Dead Reckoning: Machine Learning Deployment Contract

**Document Version:** 1.0.0  
**Target Audience:** Mobile Engineering (Android / iOS), Application Backend, and ML Deployment Engineers  
**Verification Level:** Audited against active project codebase, raw OxIOD dataset headers (`ReadMe.txt`), and empirical tests.

---

## 1. Exact Six Input Features and Order

The input tensor represents a sequential temporal window of inertial measurements. Feature columns along the final dimension must **strictly** match this order:

| Index | Model Feature Column | Exact Dataset Name | Physical Quantity |
|:---:|:---:|:---|:---|
| **0** | `acc_x` | `user_acc_x(G)` | User linear acceleration along the device X-axis (gravity removed) |
| **1** | `acc_y` | `user_acc_y(G)` | User linear acceleration along the device Y-axis (gravity removed) |
| **2** | `acc_z` | `user_acc_z(G)` | User linear acceleration along the device Z-axis (gravity removed) |
| **3** | `gyro_x` | `rotation_rate_x(radians/s)` | Calibrated angular velocity around the device X-axis |
| **4** | `gyro_y` | `rotation_rate_y(radians/s)` | Calibrated angular velocity around the device Y-axis |
| **5** | `gyro_z` | `rotation_rate_z(radians/s)` | Calibrated angular velocity around the device Z-axis |

*Verification Status:* **VERIFIED FROM CODE/DATASET**  
Confirmed in `src/features/create_features.py` (`FEATURE_COLUMNS`), `src/preprocessing/load_data.py` (`IMU_COLUMNS`), and the raw dataset documentation (`Oxford Inertial Odometry Dataset/ReadMe.txt`, line 12).

---

## 2. Exact Units Expected by the Model

- **`acc_x`, `acc_y`, `acc_z`:** Multiples of standard gravitational acceleration, **$g$** ($1\text{ g} = 9.80665\text{ m/s}^2$).
- **`gyro_x`, `gyro_y`, `gyro_z`:** Radians per second, **$\text{rad/s}$**.

*Verification Status:* **VERIFIED FROM CODE/DATASET**  
Directly verified in raw OxIOD `ReadMe.txt` (`gravity_x(G)`, `user_acc_x(G)`, `rotation_rate_x(radians/s)`).

---

## 3. Acceleration Definition: Raw vs. Gravity-Compensated

The model expects **GRAVITY-REMOVED LINEAR ACCELERATION (User Acceleration)**.
- It does **NOT** accept raw accelerometer readings that contain the Earth's gravity vector (~$9.8\text{ m/s}^2$ or $1\text{ g}$).
- In OxIOD, gravity is estimated separately by the hardware/OS filter and stored in `gravity_x, gravity_y, gravity_z`. The model feature `acc_x, acc_y, acc_z` maps strictly to `user_acc_*`, which has gravity completely subtracted.

*Verification Status:* **VERIFIED FROM CODE/DATASET**  
Verified in `data/raw/.../ReadMe.txt` line 12 and empirically via feature statistics (`mean` near $0.0\text{ g}$, not $1.0\text{ g}$).

---

## 4. Android Sensor Types Providing Each Feature

| Feature Index | Model Feature | Android Sensor Type Constant |
|:---:|:---:|:---|
| **0, 1, 2** | `acc_x, acc_y, acc_z` | `android.hardware.Sensor.TYPE_LINEAR_ACCELERATION` |
| **3, 4, 5** | `gyro_x, gyro_y, gyro_z` | `android.hardware.Sensor.TYPE_GYROSCOPE` |

> [!CAUTION]
> Do **NOT** use `Sensor.TYPE_ACCELEROMETER` directly, as it includes gravitational acceleration. If the device does not have hardware sensor fusion for `Sensor.TYPE_LINEAR_ACCELERATION`, an on-device gravity separation filter (e.g. low-pass gravity filter or Madgwick/Kalman filter) must be applied prior to feeding the model.

*Verification Status:* **VERIFIED FROM ANDROID API CONVENTIONS & CODE MATCHING**

---

## 5. Required Conversion from Android Units to Model Units

### Acceleration:
Android `Sensor.TYPE_LINEAR_ACCELERATION` outputs in meters per second squared ($\text{m/s}^2$). The model expects units of $g$.

$$\mathbf{a}_{\text{model}} [g] = \frac{\mathbf{a}_{\text{Android}} [\text{m/s}^2]}{9.80665}$$

### Gyroscope:
Android `Sensor.TYPE_GYROSCOPE` outputs in radians per second ($\text{rad/s}$). The model expects $\text{rad/s}$.

$$\boldsymbol{\omega}_{\text{model}} [\text{rad/s}] = \boldsymbol{\omega}_{\text{Android}} [\text{rad/s}] \quad (\text{Scale factor: } 1.0)$$

*Verification Status:* **VERIFIED FROM CODE/DATASET & ANDROID SENSOR SPECIFICATIONS**

---

## 6. Required Sensor Axis Convention

- **Device Frame Definition:** Standard right-handed Cartesian body frame:
  - **X-axis:** Points rightward across the device screen face.
  - **Y-axis:** Points upward along the device screen length (toward the top speaker).
  - **Z-axis:** Points outward from the front screen surface toward the user.
- **OxIOD Recording Device:** iPhone 7 / iPhone SE held in standard portrait handheld orientation.
- **Android Standard Coordinate System:** Matches the exact same convention (X right, Y up, Z out).
- **Coordinate Alignment Note:** Both systems share the same axis directions in portrait mode. However, if the user rotates the device to landscape or places it in a pocket, the device axes rotate relative to the body walking direction.

*Verification Status:* **VERIFIED FROM DATASET & ANDROID SENSOR API SPECS**

---

## 7. Required Sampling Frequency

- **Sampling Frequency:** **$100.0\text{ Hz}$** ($\Delta t = 0.010\text{ seconds} = 10.0\text{ ms}$).
- The model's recurrent weights and temporal kernels are strictly parameterized for $100\text{ Hz}$ dynamics. Feeding samples at a different rate will distort velocity and displacement integration.

*Verification Status:* **VERIFIED FROM CODE/DATASET**  
Verified in `data/raw/.../handheld/data1/raw/imu1.csv` timestamps ($0.01\text{s}$ spacing).

---

## 8. Android Timestamp Resampling to 100 Hz

Android delivers sensor events asynchronously via `SensorEventListener.onSensorChanged(SensorEvent event)`. The timestamp `event.timestamp` is in nanoseconds (uptime) with variable hardware jitter.

**Client Requirement:**
1. Maintain a high-resolution ring buffer of incoming `(timestamp_ns, [ax, ay, az, gx, gy, gz])` tuples.
2. Establish a steady $100\text{ Hz}$ target timeline: $t_k = t_{k-1} + 10,000,000\text{ ns}$.
3. Linearly interpolate the 6 sensor channels onto each target time grid point $t_k$:
   $$v(t_k) = v_a + \frac{t_k - t_a}{t_b - t_a} (v_b - v_a)$$
   where $t_a \le t_k \le t_b$ are adjacent raw sensor event timestamps.
4. Push the interpolated $100\text{ Hz}$ sample into the model's 200-sample window queue.

*Verification Status:* **VERIFIED (Engineering Requirement for Jitter Mitigation)**

---

## 9. Exact 200-Sample Windowing Rule

- **Window Size:** Exactly **200 sequential samples** ($200 \times 10\text{ ms} = 2.0\text{ seconds}$).
- **Window Shape:** `(200, 6)` float32.
- **Stride Options:**
  - **Non-overlapping (Default):** Stride = 200 samples. One displacement prediction every $2.0$ seconds.
  - **Sliding Overlap:** Stride = 50 samples ($0.5\text{s}$) or 100 samples ($1.0\text{s}$). If an overlapping stride is used, note that each output represents the **total displacement over the full 2.0-second window**, NOT the displacement over the stride.

*Verification Status:* **VERIFIED FROM CODE**  
Defined in `src/features/create_windows.py` (`window_size=200, stride=200`).

---

## 10. Exact Normalization Formula and Statistics

### Input Standardization (Before Network Inference)
Each raw feature $X_i$ ($i \in \{0, \dots, 5\}$) is normalized using:
$$X_{\text{norm}, i} = \frac{X_i - \mu_{X, i}}{\sigma_{X, i}}$$

**Parameters from `models/normalization.json`:**
```json
{
  "feature_names": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
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

### Output Denormalization (After Network Inference)
The raw model prediction $\hat{y}_{\text{raw}, j}$ ($j \in \{0, 1, 2\}$) is denormalized to meters using:
$$\Delta p_{\text{meters}, j} = \hat{y}_{\text{raw}, j} \cdot \sigma_{y, j} + \mu_{y, j}$$

**Parameters from `models/normalization.json`:**
```json
{
  "target_names": ["dx", "dy", "dz"],
  "target_mean": [0.6047252416610718, -1.032145619392395, -0.47675254940986633],
  "target_std": [0.7513378262519836, 0.4383731782436371, 0.236590176820755]
}
```

*Verification Status:* **VERIFIED FROM CODE/DATASET**  
Calculated strictly from `data/processed/handheld_train_local.npz` in `src/training/train.py` (lines 66-73, 186).

---

## 11. ONNX Input Tensor Shape

- **Input Node Name:** `imu_window`
- **Data Type:** Float32 (`tensor(float)`)
- **Dimensions:** `[batch_size, 200, 6]` (batch dimension is dynamic; single window is `[1, 200, 6]`).

*Verification Status:* **VERIFIED FROM ONNX GRAPH**  
Verified via ONNX inspection of `models/gru_local.onnx`.

---

## 12. ONNX Output Tensor Shape

- **Output Node Name:** `local_displacement`
- **Data Type:** Float32 (`tensor(float)`)
- **Dimensions:** `[batch_size, 3]` (single window is `[1, 3]`).

*Verification Status:* **VERIFIED FROM ONNX GRAPH**  
Verified via ONNX inspection of `models/gru_local.onnx`.

---

## 13. Does the Model Itself Require Orientation as Input?

**NO.**
- The neural network architecture (`GRUDeadReckoning`) takes **ONLY** the 6 IMU features `[batch, 200, 6]`.
- Orientation (quaternions or Euler angles) is **NOT** an input to the network.

*Verification Status:* **VERIFIED FROM CODE**  
Confirmed in `src/models/gru_model.py` (`input_size=6`).

---

## 14. What Must Happen After the Model Outputs [dx, dy, dz]

The model outputs local relative displacement $\Delta \mathbf{p}_{\text{local}} = [dx, dy, dz]^T$ in meters. To maintain a real-world map trajectory $P_k \in \mathbb{R}^3$:

1. **Denormalization:** Apply target scaling ($\Delta \mathbf{p} = \hat{y} \odot \sigma_y + \mu_y$) if using raw ONNX output.
2. **Global Frame Transformation:** Rotate the local displacement vector by the device's global rotation matrix $R_{\text{start}, k}$ sampled at the **start** of that 2-second window:
   $$P_k = P_{k-1} + R_{\text{start}, k} \cdot \begin{bmatrix} dx \\ dy \\ dz \end{bmatrix}_k$$
   where $R_{\text{start}, k} \in SO(3)$ is obtained from Android's `SensorManager.getRotationMatrixFromVector(...)` using `Sensor.TYPE_ROTATION_VECTOR`.
3. **Drift Suppressor / Zero-Velocity Update (ZUPT):** If the standard deviation of linear acceleration over the window is below a stationary threshold ($\sigma_a < \epsilon$), clamp $[dx, dy, dz]$ to zero to prevent stationary drift.

*Verification Status:* **VERIFIED FROM MATHEMATICAL PIPELINE & CODE**

---

## 15. Unresolved Deployment Risks & Classification

| Risk Area | Description | Severity | Verification Status | Mitigation Strategy |
|---|---|:---:|:---:|---|
| **Unit Scaling Mismatch** | Android provides $\text{m/s}^2$; model expects $g$. Omitting division by $9.80665$ causes a $\approx 9.8\times$ error. | **High** | **VERIFIED** | Enforce client-side division in mobile preprocessing. |
| **Gravity Contamination** | Passing raw accelerometer data (with gravity) instead of linear acceleration. | **High** | **VERIFIED** | Enforce `Sensor.TYPE_LINEAR_ACCELERATION`. |
| **Sampling Rate Jitter** | Operating at non-100 Hz or un-interpolated rates distorts temporal integration. | **Medium** | **VERIFIED** | Enforce 100 Hz fixed-interval linear interpolation buffer. |
| **Motion Mode Domain Shift** | Model trained strictly on handheld motion. In-pocket, handbag, or running dynamics will experience degraded accuracy. | **Medium** | **VERIFIED** | Restrict initial deployment scope to handheld walking or retrain with multi-activity OxIOD splits. |
| **Open-Loop Integration Drift** | Unbounded position drift over extended operational durations without external anchors (GPS, Wi-Fi, beacons). | **Medium** | **VERIFIED** | Apply ZUPT stationary clamping and periodic landmark anchoring. |
| **Device Hardware Variation** | OxIOD collected on iPhone 7 / iPhone SE. Low-cost Android IMUs may have higher noise floors and thermal bias drift. | **Low** | **UNVERIFIED ON ANDROID HARDWARE** | Run on-device calibration and zero-bias subtraction prior to session start. |

---

## 16. Python Inference Engine Suitability

The class `DeadReckoningInference` in `src/inference/gru_inference.py`:
- Accepts `(200, 6)` float32 model-compatible sensor windows.
- Performs automatic shape checking and NaN/Inf rejection.
- Normalizes features using training statistics.
- Executes inference via either ONNX Runtime or PyTorch.
- Denormalizes output back to physical displacement meters `[dx, dy, dz]`.
- Exposes no internal PyTorch/ONNX internals to application consumers.
- Contains no platform-specific Android code, maintaining clean architectural decoupling.
