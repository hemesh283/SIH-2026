"""ZUPT threshold calibration -- operationalizes the caveat in strapdown.py's
detect_stationary() docstring: "plot acc/gyro variance for a real recorded
stationary segment vs. a real moving segment from your own data and set the
thresholds between them before trusting this on a real trace."

Why this exists: the shipped defaults (acc_var_thresh=0.0005,
gyro_var_thresh=0.0003) were fit against one synthetic noise profile only.
On that synthetic trace they produce two failure modes depending on setting
(see evaluation/AUDIT_Phase1-4_Report.md Section 2): too tight and ZUPT
never fires during a real stop, or loosened until it fires almost
everywhere, including during real motion. Real phone accelerometer/gyro
noise floors differ from the synthetic profile, so there's no shortcut
around measuring it -- this script measures it.

Usage: record a trace containing at least one clearly stationary segment
(phone flat on a table/held still) and at least one clearly moving segment
(normal walking), note their [start_s, end_s) windows by eye from the
timestamps, then:

    python calibrate_zupt.py my_trace.csv \
        --stationary 12 27 --moving 40 70 \
        --plot calibration.png

It reports the acc/gyro variance distribution inside each labeled window,
checks whether a separating threshold even exists (stationary-max <
moving-min), and if so proposes one. If the ranges overlap, magnitude-variance
thresholding cannot cleanly separate stance from motion on this trace at all
-- that's a real, actionable result (it means detect_stationary()'s approach
needs to change, e.g. per-axis variance or a SHOE-style likelihood-ratio
detector, not just a different number), not a failure of this script.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strapdown import load_imu_csv  # noqa: E402


def rolling_variance(t, values, window_s):
    """Same windowing detect_stationary() uses: centered rolling variance of
    the vector magnitude, keyed on median sample spacing."""
    dt_med = np.median(np.diff(t)) if len(t) > 1 else 0.01
    window = max(3, int(round(window_s / dt_med)))
    mag = np.linalg.norm(values, axis=1)
    return pd.Series(mag).rolling(window, center=True, min_periods=1).var().fillna(0).to_numpy()


def segment_mask(t, windows):
    mask = np.zeros(len(t), dtype=bool)
    for lo, hi in windows:
        mask |= (t >= lo) & (t < hi)
    return mask


def summarize(name, values):
    if len(values) == 0:
        return {"label": name, "n": 0}
    return {
        "label": name,
        "n": int(len(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def propose_threshold(stationary_stats, moving_stats, channel):
    """Geometric mean of the stationary ceiling and moving floor, i.e. the
    midpoint (in log space, since variances span orders of magnitude) of the
    gap between the two populations. Only meaningful if that gap is
    positive -- see the overlap warning in main()."""
    lo = stationary_stats["max"]
    hi = moving_stats["min"]
    if lo <= 0:
        lo = 1e-12
    if hi <= lo:
        return None
    return float(np.sqrt(lo * hi))


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate ZUPT acc/gyro variance thresholds from labeled stationary/moving windows in a real trace."
    )
    parser.add_argument("input_csv", help="Sensor Logger export: timestamp, acc_x/y/z, gyro_x/y/z[, mag_x/y/z]")
    parser.add_argument("--stationary", nargs=2, type=float, action="append", required=True,
                         metavar=("START_S", "END_S"),
                         help="A [start,end) window (seconds from trace start) known to be stationary. Repeatable.")
    parser.add_argument("--moving", nargs=2, type=float, action="append", required=True,
                         metavar=("START_S", "END_S"),
                         help="A [start,end) window known to be in motion. Repeatable.")
    parser.add_argument("--window-s", type=float, default=0.25,
                         help="Variance window, seconds -- must match what you'll pass to strapdown.py (default 0.25)")
    parser.add_argument("--time-unit", choices=["auto", "s", "ms", "ns"], default="auto")
    parser.add_argument("--gyro-in-degrees", action="store_true")
    parser.add_argument("--plot", help="Optional path to save a diagnostic plot (acc/gyro variance vs. time, labeled windows highlighted)")
    parser.add_argument("--json", help="Optional path to save the full report as JSON")
    args = parser.parse_args()

    df, _has_mag = load_imu_csv(args.input_csv, time_unit=args.time_unit, gyro_in_degrees=args.gyro_in_degrees)
    t = df["t"].to_numpy()
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()

    acc_var = rolling_variance(t, acc, args.window_s)
    gyro_var = rolling_variance(t, gyro, args.window_s)

    stationary_mask = segment_mask(t, args.stationary)
    moving_mask = segment_mask(t, args.moving)
    if not stationary_mask.any():
        raise SystemExit("no samples fell inside any --stationary window -- check the times against the trace's duration")
    if not moving_mask.any():
        raise SystemExit("no samples fell inside any --moving window -- check the times against the trace's duration")

    report = {"input_csv": args.input_csv, "window_s": args.window_s, "channels": {}}
    proposed = {}
    for channel_name, var in (("acc_var", acc_var), ("gyro_var", gyro_var)):
        stationary_stats = summarize("stationary", var[stationary_mask])
        moving_stats = summarize("moving", var[moving_mask])
        threshold = propose_threshold(stationary_stats, moving_stats, channel_name)
        report["channels"][channel_name] = {
            "stationary": stationary_stats,
            "moving": moving_stats,
            "proposed_threshold": threshold,
            "separable": threshold is not None,
        }
        proposed[channel_name] = threshold

        print(f"\n[{channel_name}]")
        print(f"  stationary: n={stationary_stats['n']}, "
              f"min={stationary_stats['min']:.6g}, p50={stationary_stats['p50']:.6g}, "
              f"p95={stationary_stats['p95']:.6g}, max={stationary_stats['max']:.6g}")
        print(f"  moving:     n={moving_stats['n']}, "
              f"min={moving_stats['min']:.6g}, p50={moving_stats['p50']:.6g}, "
              f"p95={moving_stats['p95']:.6g}, max={moving_stats['max']:.6g}")
        if threshold is None:
            print(f"  ** NOT SEPARABLE ** stationary-max ({stationary_stats['max']:.6g}) >= "
                  f"moving-min ({moving_stats['min']:.6g}) -- no single threshold on this channel's "
                  f"magnitude-variance can cleanly separate stance from motion on this trace. "
                  f"detect_stationary() will misclassify one or the other no matter what number you pick; "
                  f"consider per-axis variance, a longer/shorter window, or a likelihood-ratio (SHOE-style) "
                  f"detector instead of a magnitude threshold.")
        else:
            print(f"  proposed threshold: {threshold:.6g}  "
                  f"(geometric mean of the stationary ceiling and moving floor)")

    if all(v is not None for v in proposed.values()):
        print(f"\nSuggested CLI flags for strapdown.py:\n"
              f"  --acc-var-thresh {proposed['acc_var']:.6g} --gyro-var-thresh {proposed['gyro_var']:.6g}")
    else:
        print("\nNo suggested flags -- at least one channel isn't separable on this trace (see above).")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {args.json}")

    if args.plot:
        _save_plot(args.plot, t, acc_var, gyro_var, args.stationary, args.moving, proposed)
        print(f"Diagnostic plot written to {args.plot}")


def _save_plot(path, t, acc_var, gyro_var, stationary_windows, moving_windows, proposed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, var, name in zip(axes, (acc_var, gyro_var), ("acc_var", "gyro_var")):
        ax.plot(t, var, linewidth=0.8, color="black")
        ax.set_yscale("log")
        ax.set_ylabel(name)
        for lo, hi in stationary_windows:
            ax.axvspan(lo, hi, color="tab:blue", alpha=0.2, label="labeled stationary")
        for lo, hi in moving_windows:
            ax.axvspan(lo, hi, color="tab:orange", alpha=0.2, label="labeled moving")
        threshold = proposed.get(name)
        if threshold is not None:
            ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1, label="proposed threshold")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle("ZUPT calibration -- real-trace acc/gyro variance vs. labeled windows")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
