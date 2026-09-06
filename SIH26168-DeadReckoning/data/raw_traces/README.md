# raw_traces/

Self-collected sensor-logger exports (real walk/drive tests). Never synthetic —
raw accelerometer, gyroscope, magnetometer, and GPS readings recorded during
real tests via a phone sensor-logging app (e.g. Sensor Logger-class apps).

Verification discipline before any trace is used for training:
- sanity-plot raw IMU streams against the known route
- check for sensor unit/axis errors
- verify timestamps align

Suggested naming: `YYYYMMDD_<walk|drive>_<location>_<device>.csv` (or the
sensor-logger app's native export format).

This folder is likely large / binary-heavy — consider `.gitignore`-ing raw
exports and committing only a manifest or a small sample if the repo is on GitHub.
