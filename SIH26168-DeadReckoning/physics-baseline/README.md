# physics-baseline/

Strapdown inertial mechanization + complementary filter + ZUPT.
Produces the raw dead-reckoned trajectory that the ML residual model corrects.

- `strapdown.py` — orientation (complementary filter), body→nav rotation +
  gravity removal, ZUPT stationary detection, trapezoidal double integration.
  Single file; see its module docstring for frame conventions and known
  simplifications (no EKF yet, gyro-only yaw drift without a magnetometer).
- `calibrate_zupt.py` — **run this before trusting ZUPT on any real trace.**
  `strapdown.detect_stationary()`'s variance thresholds were only ever fit
  against one synthetic noise profile (see
  `evaluation/AUDIT_Phase1-4_Report.md` Section 2 for the failure modes that
  produces: either ZUPT never fires, or it fires almost everywhere).
  This script measures real acc/gyro variance inside labeled stationary vs.
  moving windows from an actual recorded trace, checks whether a separating
  threshold even exists, and if so proposes one:
  ```
  python calibrate_zupt.py my_trace.csv --stationary 12 27 --moving 40 70 --plot calibration.png
  ```
  If it reports "NOT SEPARABLE" on a real trace, that means magnitude-variance
  thresholding can't work here at all — the fix is a different detector
  (per-axis variance, or a SHOE-style likelihood-ratio test), not a different
  number.

Libraries: NumPy, pandas; `calibrate_zupt.py --plot` additionally needs matplotlib.
