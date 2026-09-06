# IO-VNBD Ground Truth Definition & Reference Standard

---

## 1. Ground Truth Source Selection
In the IO-VNBD dataset, positional and kinematic data are recorded in two distinct subsystems:
1. **Smartphone GPS (`S-*.csv`):** Internal consumer receiver of the Android smartphone, logged at 1 Hz (repeated at 10 Hz).
2. **Vehicle CAN / ECU GPS & Dynamics (`V-*.csv`):** High-precision vehicle GPS receiver integrated with the Ford Fiesta's Electronic Control Unit (ECU) and chassis CAN bus.

### Selected Standard: **Vehicle ECU (`V-*.csv`)**
The vehicle ECU data is designated as the **authoritative ground truth** for all supervised ML target construction and navigation benchmark drift evaluations.

---

## 2. Technical Specification

### A. Position Fields
- **Latitude:** `Latitude (degrees)` in `V-*.csv` (WGS-84 ellipsoid).
- **Longitude:** `Longitude (degrees)` in `V-*.csv` (WGS-84 ellipsoid).
- **Elevation:** `Height (km)` converted to meters ($h = \text{Height} \times 1000$).
- **Local Navigation Projection:** Converted to Cartesian North-East-Down (NED) metric coordinates $[\Delta p_N, \Delta p_E, \Delta p_D]^T$ relative to sequence anchor $(\text{lat}_0, \text{lon}_0)$ via:
  $$\Delta p_N = R_{\text{earth}} \cdot (\text{lat} - \text{lat}_0) \cdot \frac{\pi}{180}$$
  $$\Delta p_E = R_{\text{earth}} \cdot (\text{lon} - \text{lon}_0) \cdot \cos(\text{lat}_0) \cdot \frac{\pi}{180}$$

### B. Velocity Fields
- **Longitudinal / Ground Speed:** `Velocity (km/hr)` converted to $\text{m/s}$:
  $$v = \frac{\text{Velocity (km/hr)}}{3.6}$$
- **Auxiliary Odometry Verification:** 4-wheel independent rotational speeds (`Wheel Speed Front Left`, `Front Right`, `Rear Left`, `Rear Right` in $\text{rad/s}$) multiplied by tire rolling radius ($r \approx 0.30\text{ m}$) provides independent physical odometry cross-validation:
  $$v_{\text{wheel}} = \frac{\omega_{\text{RL}} + \omega_{\text{RR}}}{2} \cdot r_{\text{tire}}$$

### C. Orientation / Heading Fields
- **Course Over Ground:** `Heading (degrees)` clockwise from True North $[0, 360)$.
- **Yaw Rate:** `Yaw Rate (deg/sec)` from vehicle chassis gyro.

### D. Time Synchronization Standard
- **Time Base:** `Time Since Start of Day (seconds)` in `V-*.csv` has a verified 1-to-1 sample mapping with `TIME SINCE START (ms)` in `S-*.csv`.
- Both sequences are recorded at exactly **10.0 Hz** ($\Delta t = 100.0\text{ ms}$).

---

## 3. Justification Over Smartphone GPS
| Attribute | Smartphone GPS (`S-*.csv`) | Vehicle ECU (`V-*.csv`) — **SELECTED** |
| :--- | :--- | :--- |
| **Antenna Quality** | Tiny internal patch antenna on phone PCB | External roof-mounted vehicle antenna with ground plane |
| **Multipath Rejection** | Prone to severe urban multipath reflections | High SNR, satellite count tracked (`No of GPS Satellites`) |
| **Velocity Accuracy** | Coarse Doppler speed from smartphone chip | High-rate CAN velocity coupled with wheel sensors |
| **Update Rate** | 1 Hz updates duplicated to 10 Hz | Genuine 10 Hz kinematic trajectory |
| **Odometric Cross-check**| None | 4-channel wheel speed sensors, brake pressure, yaw rate |

---

## 4. Known Limitations
1. **10 Hz Quantization:** Ground truth is available at 10 Hz; intra-sample dynamics $(< 100\text{ ms})$ are linearly interpolated if evaluated at higher rates.
2. **Tunnel / Heavy Foliage Outages:** During severe GPS blackouts, vehicle GPS in `V-*.csv` may experience brief dilution of precision; sequence pre-filtering validates satellite count $\ge 4$ for anchor points.
