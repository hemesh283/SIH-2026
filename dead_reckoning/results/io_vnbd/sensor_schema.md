# IO-VNBD Sensor Schema & Field Definitions

**Dataset Source:** `D:\dead_reckoning\data\raw\IO-VNBD\Synchronised V abd S datasets`  
**Files Audited:** 144 Smartphone CSVs (`S-*.csv`) and 144 Vehicle CSVs (`V-*.csv`)  
**Total Pairs:** 72 Synchronized Recording Pairs across 5 Drivers (A, B, C, D, E)

---

## 1. Smartphone Dataset (`S-*.csv`)
Recorded via Android **AndroSensor** application mounted in the test vehicle dashboard holder.  
**Native Sampling Rate:** Exactly **10.0 Hz** ($\Delta t = 100.0\text{ ms}$).  
**Encoding:** `ISO-8859-1` / `latin-1` (due to superscript `²` and degree `°` symbols).

| Index | Exact Column Header | Cleaned Name | Unit | Type | Description & Navigational Meaning |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 1 | `GPS LATITUDE (degrees)` | `s_gps_lat` | degrees | Float64 | WGS-84 Geodetic Latitude from phone GNSS |
| 2 | ` GPS LONGITUDE (degrees)` | `s_gps_lon` | degrees | Float64 | WGS-84 Geodetic Longitude from phone GNSS |
| 3 | ` GPS ALTITUDE (m)` | `s_gps_alt` | meters | Float64 | Altitude above sea level |
| 4 | ` GPS SPEED (Kmh)` | `s_gps_speed` | km/h | Float32 | Smartphone GNSS ground speed (convert to m/s via $/ 3.6$) |
| 5 | ` GPS ACCURACY (m)` | `s_gps_acc` | meters | Float32 | Estimated horizontal accuracy radius of GNSS fix |
| 6 | ` GPS ORIENTATION (°)` | `s_gps_hdg` | degrees | Float32 | Course over ground clockwise from True North $[0, 360)$ |
| 7 | `GPS SATELLITES IN RANGE` | `s_gps_sats` | count | Int32 | Number of visible GNSS space vehicles |
| 8 | ` TIME SINCE START (ms)` | `timestamp_ms` | ms | Float64 | Elapsed time since recording start ($\Delta t \approx 100\text{ ms}$) |
| 9 | ` DATE (YYYY-MO-DD HH-MI-SS_SSS)` | `date_str` | string | String | Wall-clock UTC/local timestamp string |
| 10 | ` ACCELEROMETER X (m/s²) ` | `acc_raw_x` | $\text{m/s}^2$ | Float32 | Total phone acceleration X (including gravity component) |
| 11 | ` ACCELEROMETER Y (m/s²)` | `acc_raw_y` | $\text{m/s}^2$ | Float32 | Total phone acceleration Y (including gravity component) |
| 12 | ` ACCELEROMETER Z (m/s²)` | `acc_raw_z` | $\text{m/s}^2$ | Float32 | Total phone acceleration Z (including gravity component) |
| 13 | ` GRAVITY X (m/s²)` | `grav_x` | $\text{m/s}^2$ | Float32 | Gravity vector projection along X |
| 14 | ` GRAVITY Y (m/s²)` | `grav_y` | $\text{m/s}^2$ | Float32 | Gravity vector projection along Y |
| 15 | ` GRAVITY Z (m/s²)` | `grav_z` | $\text{m/s}^2$ | Float32 | Gravity vector projection along Z |
| 16 | ` GYROSCOPE Yaw (rad/s)` | `gyro_yaw` | rad/s | Float32 | Angular velocity around vehicle/phone Z-axis (turning rate) |
| 17 | ` GYROSCOPE Pitch (rad/s)` | `gyro_pitch` | rad/s | Float32 | Angular velocity around lateral pitch axis |
| 18 | ` GYROSCOPE Roll (rad/s)` | `gyro_roll` | rad/s | Float32 | Angular velocity around longitudinal roll axis |
| 19 | ` MAGNETIC FIELD X (μT)` | `mag_x` | $\mu\text{T}$ | Float32 | Tri-axial geomagnetic field strength X |
| 20 | ` MAGNETIC FIELD Y (μT)` | `mag_y` | $\mu\text{T}$ | Float32 | Tri-axial geomagnetic field strength Y |
| 21 | ` MAGNETIC FIELD Z (μT)` | `mag_z` | $\mu\text{T}$ | Float32 | Tri-axial geomagnetic field strength Z |
| 22 | ` ORIENTATION (Yaw) (°)` | `ori_yaw` | degrees | Float32 | Compass/Fused device orientation azimuth |
| 23 | ` ORIENTATION (Pitch) (°)` | `ori_pitch` | degrees | Float32 | Device inclination pitch angle |
| 24 | ` ORIENTATION (Roll ) (°)` | `ori_roll` | degrees | Float32 | Device inclination roll angle |

### Linear Acceleration Derivation:
$$\mathbf{a}_{\text{linear}} = \mathbf{a}_{\text{accel}} - \mathbf{g}_{\text{gravity}} \quad (\text{m/s}^2)$$
$$\mathbf{a}_g = \frac{\mathbf{a}_{\text{linear}}}{9.80665} \quad (g)$$

---

## 2. Vehicle CAN Bus / ECU Dataset (`V-*.csv`)
Recorded directly from vehicle Electronic Control Unit (Ford Fiesta research vehicle) via CAN bus.  
**Native Sampling Rate:** Exactly **10.0 Hz** ($\Delta t = 100.0\text{ ms}$).  
**Total Columns:** 29

| Index | Exact Column Header | Cleaned Name | Unit | Meaning |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `No of GPS Satellites Available` | `v_gps_sats` | count | Number of satellites tracked by vehicle GPS |
| 2 | ` Time Since Start of Day (seconds)` | `v_time_day_s` | seconds | High-precision time since midnight |
| 3 | ` Latitude (degrees)` | `v_gt_lat` | degrees | **Vehicle Reference Latitude (Ground Truth)** |
| 4 | ` Longitude (degrees)` | `v_gt_lon` | degrees | **Vehicle Reference Longitude (Ground Truth)** |
| 5 | ` Velocity (km/hr)` | `v_gt_speed_kmh`| km/h | **Vehicle Ground Truth Speed** (convert to m/s via $/ 3.6$) |
| 6 | ` Heading (degrees)` | `v_gt_heading` | degrees | **Vehicle Ground Truth Heading** $[0, 360)$ |
| 7 | ` Height (km)` | `v_gt_height` | km | Elevation above ellipsoid |
| 8 | ` Vertical velocity (km/hr)` | `v_gt_vert_vel`| km/h | Vertical ascent/descent rate |
| 9 | ` Sample period (seconds)` | `v_sample_dt` | seconds | CAN bus sampling interval (0.100 s) |
| 10 | ` Steering Angle (degrees)` | `v_steer_angle`| degrees | Driver handwheel steering angle |
| 11 | ` Wheel Speed Front Left (rad/sec)` | `wheel_spd_fl`| rad/s | Rotational speed of front-left wheel |
| 12 | ` Wheel Speed Front Right (rad/sec)`| `wheel_spd_fr`| rad/s | Rotational speed of front-right wheel |
| 13 | ` Wheel Speed Rear Left (rad/sec)` | `wheel_spd_rl`| rad/s | Rotational speed of rear-left wheel |
| 14 | ` Wheel Speed Rear Right (rad/sec)` | `wheel_spd_rr`| rad/s | Rotational speed of rear-right wheel |
| 15 | ` Yaw Rate (deg/sec)` | `v_yaw_rate` | deg/s | Vehicle body rotational rate |
| 16 | ` Indicated Vehicle Speed (km/hr)` | `v_ind_speed` | km/h | Speedometer reading from wheel odometry |
| 17 | ` Indicated Longitudinal Acceleration (g)` | `v_acc_long` | $g$ | Vehicle longitudinal accelerometer |
| 18 | ` Indicated Lateral Acceleration (g)` | `v_acc_lat` | $g$ | Vehicle centripetal lateral accelerometer |
| 19-29 | Handbrake, Gears, Engine RPM, Brake Pressure, Pedals | ... | mixed | Vehicle subsystem dynamics |
