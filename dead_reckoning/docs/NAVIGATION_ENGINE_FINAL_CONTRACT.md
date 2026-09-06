# Final Navigation Engine Contract: Dead Reckoning Architecture

**Project:** Smart India Hackathon 2026 (SIH26168)  
**Problem Statement:** AI/ML based Intelligent Dead Reckoning System for Seamless Navigation  
**Organization:** Indian Space Research Organisation (ISRO)  
**Repository:** `D:\dead_reckoning`  
**Layer:** Navigation Engine (State Estimation, EKF, Coordinate Transforms, Constraints, Map Interface)  
**Contract Status:** **FROZEN & AUDITED**  
**Effective Date:** September 2026  

---

## 1. Executive Summary & Architectural Classification

This document specifies the authoritative contract for the Dead Reckoning Navigation Engine. The engine bridges local 3D translational displacements predicted by the on-device ML model with high-rate inertial sensors, orientation filters, motion constraints, and the user-facing map.

### Architectural Classification & Implementation Reality:
- **Navigation Filter:** Explicitly classified as a **SIMPLIFIED 6-STATE MVP NAVIGATION FILTER**. It models translational position and velocity in local North-East-Down (NED) coordinates. It is **NOT** a full 15-state error-state inertial navigation EKF (ES-EKF); it does not explicitly estimate accelerometer biases, gyroscope biases, or attitude errors.
- **Orientation Pipeline:** Relies on Android's fused `Sensor.TYPE_ROTATION_VECTOR` (or vehicle CAN reference heading during offline benchmarking). Unassisted open-loop gyro heading integration is not performed.
- **Constraints (NHC & ZUPT):** Implemented as discrete Kalman filter measurement updates. Velocity is corrected through Kalman innovation and covariance reduction; it is never hard-clamped or overwritten.
- **Map Matching:** Default active component is `PassThroughMapMatcher`. No artificial road snapping, trajectory fabrication, or hardcoded coordinates are used.

---

## A. System State Definition

The navigation system maintains two distinct state representations:

### 1. Internal Estimator State (NED Cartesian Frame)
Represented in the 6-state navigation filter ($\mathbf{x} \in \mathbb{R}^6$):
$$\mathbf{x} = \begin{bmatrix} p_N \\ p_E \\ p_D \\ v_N \\ v_E \\ v_D \end{bmatrix} \quad \begin{matrix} \text{North position relative to anchor (m)} \\ \text{East position relative to anchor (m)} \\ \text{Down position relative to anchor (m)} \\ \text{North velocity (m/s)} \\ \text{East velocity (m/s)} \\ \text{Down velocity (m/s)} \end{matrix}$$

### 2. High-Level Output State (`NavigationEngineState`)
Dispatched to UI components and trajectory logs:
- `latitude`: WGS-84 geodetic latitude in degrees ($\text{deg}$)
- `longitude`: WGS-84 geodetic longitude in degrees ($\text{deg}$)
- `altitude`: Height above ground / ellipsoid in meters ($h = -p_D$)
- `speed`: Horizontal speed in meters per second ($v_{\text{horiz}} = \sqrt{v_N^2 + v_E^2}$)
- `heading`: Azimuth in degrees clockwise from True North ($[0.0^\circ, 360.0^\circ)$)
- `distanceTravelled`: Cumulative horizontal path distance in meters
- `isInitialized`: Boolean indicating valid GNSS anchor / starting fix
- `isStationary`: Boolean from multi-signal ZUPT detector
- `isBlackout`: Boolean indicating active GNSS denial state
- `timestampNs`: Monotonic nanosecond hardware timestamp

---

## B. Coordinate-Frame Definitions

Five distinct coordinate frames are defined and maintained across the pipeline:

```
[Phone Sensor Frame (p)]
          │
          │ R_pv (Fixed mounting matrix)
          ▼
[Vehicle Body Frame (b / v)]
          │
          │ R_start (Initial window attitude)
          ▼
[Local Window Frame (b_0)] ──(ML predicts [dx, dy, dz])
          │
          │ R_bn(heading, pitch, roll)
          ▼
[Navigation Tangent Frame (NED - n)]
          │
          │ Ellipsoidal projection (WGS-84 / Earth Radius)
          ▼
[Geodetic Map Frame (WGS-84 - g)] -> (Latitude, Longitude)
```

1. **Phone Sensor Frame ($p$):** Standard Android hardware axes: $+X$ points right along display width, $+Y$ points up along display height, $+Z$ points out of screen towards user.
2. **Vehicle Body Frame ($b$ or $v$):** Standard automotive body axes: $+X$ points forward along vehicle longitudinal axis, $+Y$ points right (lateral), $+Z$ points down towards road surface.
3. **Local Window Frame ($b_0$):** The vehicle body orientation at the initial sample ($t=0$) of each 20-sample ($2.0\text{ s}$) inference window. The GRU outputs local displacement $[\Delta x, \Delta y, \Delta z]^T$ relative to this reference frame.
4. **Navigation Frame (NED - $n$):** Local tangent plane at geodetic anchor $(\text{lat}_0, \text{lon}_0)$. Cartesian axes: $+X$ North ($p_N$), $+Y$ East ($p_E$), $+Z$ Down ($p_D$).
5. **Geodetic Frame (WGS-84 - $g$):** Global Earth coordinates defined by latitude $\phi$, longitude $\lambda$, and ellipsoidal height $h$.

---

## C. Sensor Inputs and Units

The Navigation Engine consumes data from standard Android hardware sensors:

| Sensor Type | Android Constant | Raw Unit | Required Engine Unit | Preprocessing / Scale Factor |
| :--- | :--- | :---: | :---: | :--- |
| **Linear Acceleration** | `Sensor.TYPE_LINEAR_ACCELERATION` | $\text{m/s}^2$ | $g$ | Gravity subtracted by Android HAL; divided by $9.80665$ for ML buffer |
| **Angular Velocity** | `Sensor.TYPE_GYROSCOPE` | $\text{rad/s}$ | $\text{rad/s}$ | Unscaled calibrated angular rates ($g_x, g_y, g_z$) |
| **Rotation Vector** | `Sensor.TYPE_ROTATION_VECTOR` | Unitless | $\text{deg}, \text{rad}$ | Converted to $3 \times 3$ DCM $R$ and Euler angles $(\psi, \theta, \phi)$ |
| **GNSS Position** | Android `Location` API | $\text{deg}, \text{m}$ | $\text{deg}, \text{m}$ | Latitude, Longitude, Altitude (WGS-84) |
| **GNSS Velocity** | Android `Location` API | $\text{m/s}, \text{deg}$ | $\text{m/s}$ | Converted to $(v_N, v_E)$ via speed and bearing |

---

## D. Phone-to-Vehicle Coordinate Transformation

### Implementation Status: IMPLEMENTED — FIXED MOUNTING ASSUMPTION
- **Transformation Model:** Fixed 3x3 orthogonal rotation matrix $R_{pv}$:
  $$\mathbf{v}_{\text{vehicle}} = R_{pv} \mathbf{v}_{\text{phone}}$$
- **Mount Angle Formulation:** When mount angles (roll $\phi_m$, pitch $\theta_m$, yaw $\psi_m$) are configured:
  $$R_{pv} = R_z(\psi_m) R_y(\theta_m) R_x(\phi_m)$$
- **Default Baseline:** Identity matrix $I_3$ (assuming smartphone is securely affixed in landscape dashboard orientation with $+X$ forward, $+Y$ lateral, $+Z$ vertical).
- **Scope Limitation:** Dynamic online self-calibration (estimating $R_{pv}$ adaptively during arbitrary turns or in-pocket movements) is **NOT IMPLEMENTED / INTERFACE ONLY**. The vehicle must be driven with the phone in a fixed, known mount.

---

## E. Orientation & Heading Pipeline

### Implementation Status: IMPLEMENTED (FUSED SENSOR API / REFERENCE)
- **Primary Source:** Android OS `Sensor.TYPE_ROTATION_VECTOR`, which internally fuses gyroscope, accelerometer, and magnetometer in Android's Sensor HAL.
- **Matrix & Euler Extraction:**
  ```kotlin
  SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
  SensorManager.getOrientation(rotationMatrix, orientationAngles)
  val azimuthRad = orientationAngles[0] // Yaw / Heading
  val pitchRad = orientationAngles[1]   // Pitch
  val rollRad = orientationAngles[2]    // Roll
  ```
- **Heading Convention:** Converted to clockwise degrees from True North:
  $$\psi_{\text{deg}} = (\text{azimuthRad} \cdot \frac{180}{\pi} + 360) \pmod{360}$$
- **Velocity-Derived Heading Aid:** When vehicle horizontal speed exceeds $0.5\text{ m/s}$, heading is verified and cross-checked against the velocity vector:
  $$\psi_{\text{vel}} = \text{atan2}(v_E, v_N) \cdot \frac{180}{\pi}$$
- **Offline Benchmark Reference:** In the offline IO-VNBD Python benchmark, reference heading `gt_hdg` was utilized to isolate displacement dead reckoning from consumer gyro drift.

---

## F. INS Propagation

### Implementation Status: IMPLEMENTED — SIMPLIFIED MVP
- **Kinematic Model:** First-order discrete translational propagation.
- **Position Propagation:**
  $$\mathbf{p}_k = \mathbf{p}_{k-1} + \Delta \mathbf{p}_{\text{NED}}$$
- **Velocity Propagation:**
  $$\mathbf{v}_k = \frac{\Delta \mathbf{p}_{\text{NED}}}{\Delta t_{\text{safe}}}, \quad \Delta t_{\text{safe}} = \max(\Delta t, 0.001\text{ s})$$
- **Orientation Integration:** Attitude is tracked externally via rotation vectors/reference heading rather than open-loop angular rate integration.

---

## G. EKF Equations and State

### Implementation Status: IMPLEMENTED — SIMPLIFIED 6-STATE MVP NAVIGATION FILTER

#### 1. State Vector & Covariance
$$\mathbf{x} = [p_N, p_E, p_D, v_N, v_E, v_D]^T \in \mathbb{R}^6, \quad P \in \mathbb{R}^{6 \times 6}$$
Initial covariance:
$$P_0 = \text{diag}([\sigma_{\text{pos}}^2, \sigma_{\text{pos}}^2, \sigma_{\text{pos}}^2, \sigma_{\text{vel}}^2, \sigma_{\text{vel}}^2, \sigma_{\text{vel}}^2]), \quad \sigma_{\text{pos}} = 3.0\text{ m}, \, \sigma_{\text{vel}} = 0.3\text{ m/s}$$

#### 2. Prediction Step (Time Update)
Driven by metric displacement increment $\Delta \mathbf{p}_{\text{NED}}$ over stride interval $\Delta t$:
$$\mathbf{x}_{k|k-1} = \begin{bmatrix} \mathbf{p}_{k-1} + \Delta \mathbf{p}_{\text{NED}} \\ \Delta \mathbf{p}_{\text{NED}} / \Delta t \end{bmatrix}$$
State transition Jacobian $F$:
$$F = \begin{bmatrix} I_3 & \Delta t I_3 \\ 0_3 & I_3 \end{bmatrix}$$
Process noise covariance $Q$:
$$Q = \begin{bmatrix} q_{\text{pos}}^2 \Delta t \cdot I_3 & 0_3 \\ 0_3 & q_{\text{vel}}^2 \Delta t \cdot I_3 \end{bmatrix}, \quad q_{\text{pos}} = 0.05\text{ m}, \, q_{\text{vel}} = 0.20\text{ m/s}$$
Covariance propagation:
$$P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$

#### 3. Measurement Updates (Observation Model)
For any measurement $\mathbf{z}$ with observation matrix $H$ and noise covariance $R$:
$$\mathbf{y} = \mathbf{z} - H \mathbf{x}_{k|k-1} \quad \text{(Innovation)}$$
$$S = H P_{k|k-1} H^T + R \quad \text{(Innovation Covariance)}$$
$$K = P_{k|k-1} H^T S^{-1} \quad \text{(Kalman Gain)}$$
$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K \mathbf{y} \quad \text{(State Correction)}$$

#### 4. Joseph Form Covariance Update (Numerical Stability)
To guarantee positive semi-definiteness and symmetry under finite floating-point precision:
$$P_{k|k} = (I - KH) P_{k|k-1} (I - KH)^T + K R K^T$$

#### 5. Supported Measurement Models:
- **GNSS Position:** $\mathbf{z} = [p_N, p_E, p_D]^T$, $H = [I_3, 0_3]$, $R = \sigma_{\text{gnss\_pos}}^2 I_3$ ($\sigma = 2.5$–$3.0\text{ m}$)
- **GNSS Velocity:** $\mathbf{z} = [v_N, v_E, v_D]^T$, $H = [0_3, I_3]$, $R = \sigma_{\text{gnss\_vel}}^2 I_3$ ($\sigma = 0.2$–$0.3\text{ m/s}$)

---

## H. Non-Holonomic Constraints (NHC)

### Implementation Status: IMPLEMENTED
- **Physical Principle:** Land wheeled vehicles move along their longitudinal axis without lateral tire skidding or vertical takeoff under normal road driving.
- **Pseudo-Measurement Vector:**
  $$\mathbf{z}_{\text{nhc}} = \begin{bmatrix} v_{\text{lateral}} \\ v_{\text{vertical}} \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$
- **Observation Matrix $H_{\text{nhc}} \in \mathbb{R}^{2 \times 6}$:**
  Transforming NED velocity $[v_N, v_E, v_D]^T$ into vehicle body lateral and vertical axes:
  $$H_{\text{nhc}} = \begin{bmatrix} 0 & 0 & 0 & -\sin\psi & \cos\psi & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$
- **Noise Covariance $R_{\text{nhc}}$:**
  $$R_{\text{nhc}} = \begin{bmatrix} \sigma_{\text{lat}}^2 & 0 \\ 0 & \sigma_{\text{vert}}^2 \end{bmatrix}, \quad \sigma_{\text{lat}} = 0.15\text{ m/s}, \, \sigma_{\text{vert}} = 0.10\text{ m/s}$$
- **Activation Scope:** Active for all forward moving vehicle evaluations; suppresses unconstrained lateral trajectory drift.

---

## I. Zero-Velocity Updates (ZUPT)

### Implementation Status: IMPLEMENTED
- **Multi-Signal Stationary Detector:** Evaluates sliding window of recent IMU samples:
  1. *Acceleration Magnitude & Variance:* $\|a_{\text{lin}}\| < 0.25\text{ m/s}^2$ ($0.08\text{ g}$) and $\text{Var}(\|a\|) < 0.04\text{ (m/s}^2)^2$ ($0.015\text{ g}^2$).
  2. *Gyroscope Magnitude & Variance:* $\|\boldsymbol{\omega}\| < 0.10\text{ rad/s}$ ($0.05\text{ rad/s}$) and $\text{Var}(\|\boldsymbol{\omega}\|) < 0.01\text{ (rad/s)}^2$.
  3. *GNSS Speed Check (when fix available):* $v_{\text{gnss}} < 0.30\text{ m/s}$.
  4. *Debounce Filter:* Minimum 15 consecutive stationary samples required to transition state.
- **Pseudo-Measurement Model:**
  $$\mathbf{z}_{\text{zupt}} = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix}, \quad H_{\text{zupt}} = \begin{bmatrix} 0_3 & I_3 \end{bmatrix}, \quad R_{\text{zupt}} = \sigma_{\text{zupt}}^2 I_3, \quad \sigma_{\text{zupt}} = 0.02\text{–}0.05\text{ m/s}$$
- **Filter Update Policy:**
  Velocity is **NEVER** hard-clamped or directly overwritten. It is corrected strictly via Kalman innovation, ensuring consistent reduction of velocity covariance in $P$.

---

## J. GNSS Outage & Recovery Handling

### Implementation Status: IMPLEMENTED
- **Outage Transition:**
  - When GNSS signal is lost or blackout is triggered (`isBlackoutMode = true`), `ekf.updateGnssPosition()` and `ekf.updateGnssVelocity()` are strictly bypassed.
  - State propagation continues smoothly using ML displacements $\Delta \mathbf{p}_{\text{NED}}$, constrained by NHC and ZUPT.
- **Recovery Transition (GNSS Reacquired):**
  - When valid GNSS fixes resume, measurement updates re-engage.
  - The Kalman innovation $\mathbf{y} = \mathbf{z}_{\text{gnss}} - H\mathbf{x}$ updates the state smoothly according to the covariance ratio $P / (P + R)$.
  - Prevents trajectory tearing, jumping, or discontinuous coordinate snapping.

---

## K. ML-to-Navigation Interface

### Implementation Status: IMPLEMENTED
- **Input to ML:** Preprocessed 20-sample window at 10 Hz ($2.0\text{ s}$) formatted as `[1, 20, 6]` in units of $[g, g, g, \text{rad/s}, \text{rad/s}, \text{rad/s}]$.
- **Output from ML:** 3D local body-frame displacement $[\Delta x, \Delta y, \Delta z]^T$ in meters.
- **Coordinate Transformation Step:**
  $$\Delta \mathbf{p}_{\text{NED}} = R_{bn}(\psi) \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix} = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta z \end{bmatrix}$$
- **Stride Integration:** Evaluated at 10-sample stride ($1.0\text{ s}$ update). Displacement $\Delta \mathbf{p}_{\text{NED}}$ is injected directly into `ekf.predict(deltaPNed, dt=1.0)`.

---

## L. Map / Geodetic Interface

### Implementation Status: IMPLEMENTED
- **Local NED to Geodetic Formulation:**
  Using mean Earth radius $R_E = 6,371,000.0\text{ m}$ (or WGS-84 curvature radii):
  $$\Delta \phi = \frac{p_N}{R_E}, \quad \Delta \lambda = \frac{p_E}{R_E \cos(\phi_0)}$$
  $$\text{lat} = \text{lat}_0 + \Delta \phi \cdot \frac{180}{\pi}, \quad \text{lon} = \text{lon}_0 + \Delta \lambda \cdot \frac{180}{\pi}$$
- **Map-Matching Policy:**
  - Active implementation: `PassThroughMapMatcher` (`isEnabled = false`).
  - Trajectory points pass through unmodified without snapping to road networks.
  - Guarantees true dead-reckoning drift is visually and quantitatively honest.

---

## M. Numerical Safeguards

- **Time Step Bounds:** $\Delta t_{\text{safe}} = \max(\Delta t, 0.001\text{ s})$ prevents division by zero during rapid sensor callbacks.
- **Joseph Form Covariance:** Guarantees covariance matrix $P$ remains symmetric and positive semi-definite across thousands of prediction-update cycles.
- **Matrix Inversion Guards:** Explicit determinant check ($\det(S) \ne 0$) before inverting $S$ in Kalman updates; update skipped if singular.
- **Speed Clamping:** Speed calculation uses $\sqrt{\max(v_N^2 + v_E^2, 0.0)}$ preventing `NaN` from floating-point roundoff.

---

## N. Failure & Invalid-Data Handling

- **Sensor Disconnection:** Missing or delayed sensor samples are held by the resampler ring buffer; inference is suppressed until 20 valid samples are present.
- **Out-of-Bounds Rejection:** Accelerations $> 8.0\text{ g}$ or angular rates $> 20.0\text{ rad/s}$ are treated as shock artifacts and rejected.
- **NaN / Inf Quenching:** Input buffers containing non-finite values return zero displacement $[0, 0, 0]^T$ rather than propagating numerical corruption into filter states.
- **Uninitialized State:** If called before a valid GNSS starting fix, dead reckoning buffers incoming IMU samples but suppresses position integration until anchored.

---

## O. Known Assumptions and Limitations

1. **Simplified 6-State MVP Filter:** Sensor biases ($b_a, b_g$) and attitude errors are not estimated inside the state vector. Uncorrected gyro drift over extended blackouts directly impacts heading.
2. **Fixed Phone Mount Assumption:** Requires the device to remain firmly secured in a vehicle dashboard cradle. Dynamic orientation auto-calibration for loose or moving phones is not implemented.
3. **Planar Motion (NHC Assumption):** Assumes normal wheeled driving without significant tire slip, drifting, or vertical vehicle bounce.
4. **Offline Reference Heading Boundary:** Offline multi-sequence benchmark results used synchronized vehicle reference heading. In production without dual-antenna GNSS, heading drift from consumer gyroscopes will degrade performance.
5. **Android 10 Hz Contract Update Pending:** Android application currently contains the 100 Hz `[1, 200, 6]` OxIOD pipeline; updating edge Kotlin modules to the native 10 Hz `[1, 20, 6]` contract remains a subsequent task.
