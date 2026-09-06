"""SHAP explainability for the trained LightGBM residual models (dossier Section 4.4).

The pitch: because the ML layer predicts only a *correction*, not the whole
trajectory, SHAP can directly answer -- for one specific prediction -- was
this correction driven mainly by a turn, a vibration signature suggesting a
bumpy road, or a long GNSS-outage duration? This module answers exactly
that question, by rolling the per-feature SHAP values up into the same
semantic groups features.py's docstring defines (turning / vibration /
blackout duration / motion mode / raw motion level).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import shap

FEATURE_GROUPS = {
    "turning": ["turn_rate_mean_abs", "turn_rate_max_abs", "heading_change_abs",
                "gyro_mean_z", "gyro_std_z"],
    "vibration_road_noise": ["vibration_acc", "vibration_gyro", "acc_mag_std", "gyro_mag_std"],
    "blackout_duration": ["time_since_last_gnss_fix"],
    "motion_mode": ["motion_mode", "baseline_speed"],
    "raw_motion_level": ["acc_mean_x", "acc_mean_y", "acc_mean_z", "acc_std_x", "acc_std_y", "acc_std_z",
                          "acc_mag_mean", "gyro_mean_x", "gyro_mean_y", "gyro_std_x", "gyro_std_y",
                          "gyro_mag_mean", "window_n_samples"],
}


def _group_of(feature_name):
    for group, members in FEATURE_GROUPS.items():
        if feature_name in members:
            return group
    return "other"


def build_explainer(model):
    """model: a fitted LGBMRegressor (sklearn wrapper) or lgb.Booster."""
    return shap.TreeExplainer(model)


def explain_row(explainer, X_row):
    """Per-feature SHAP contributions for one window's feature vector.

    X_row must be a single-row DataFrame with the same columns/dtypes used
    in training (motion_mode as pandas 'category').
    """
    shap_values = explainer.shap_values(X_row)
    values = np.asarray(shap_values).reshape(-1)
    return pd.Series(values, index=X_row.columns).sort_values(key=np.abs, ascending=False)


def explain_row_by_group(explainer, X_row):
    """Same SHAP contributions, rolled up into the semantic groups the
    dossier promises to explain (turn / vibration / blackout duration /
    motion mode)."""
    contributions = explain_row(explainer, X_row)
    return contributions.groupby(_group_of).sum().sort_values(key=np.abs, ascending=False)


def explain_prediction_human(explainer, X_row, target_name, prediction, unit="m"):
    """One-paragraph, jury-facing explanation for a single prediction."""
    grouped = explain_row_by_group(explainer, X_row)
    lines = [f"Predicted {target_name} correction: {prediction:+.2f}{unit}."]
    for group, contribution in grouped.items():
        if abs(contribution) < 1e-6:
            continue
        direction = "pushed the correction up" if contribution > 0 else "pulled the correction down"
        lines.append(f"  - {group.replace('_', ' ')}: {direction} by {contribution:+.3f}{unit}")
    return "\n".join(lines)
