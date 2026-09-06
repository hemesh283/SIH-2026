"""Train LightGBM residual regressors, one per target (dossier Section 4.2).

Grouping cross-validation by trace_id keeps every window from one recording
on the same side of a split -- a plain KFold would let two windows one
second apart (near-duplicate feature vectors, since windows overlap via the
stride) land in different folds, quietly inflating apparent accuracy
through leakage. Out-of-fold predictions from this CV are the honest,
held-out error uncertainty.py calibrates its duration buckets against, per
Section 4.3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold

from features import FEATURE_COLUMNS, TARGET_COLUMNS

CATEGORICAL_FEATURES = ["motion_mode"]

DEFAULT_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)


def _prepare_X(feature_df, feature_columns):
    X = feature_df[feature_columns].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def _make_folds(n_rows, groups, n_splits):
    n_groups = len(np.unique(groups)) if groups is not None else 0
    if groups is not None and n_groups >= n_splits:
        return list(GroupKFold(n_splits=n_splits).split(np.arange(n_rows), groups=groups))
    print(f"[train] only {n_groups} trace group(s) available -- falling back to plain "
          f"KFold(n_splits={min(n_splits, max(2, n_rows // 10))}). Group-aware CV needs "
          f">= {n_splits} distinct trace_ids; add more traces before trusting these "
          f"out-of-fold numbers as a real accuracy estimate.")
    n_splits = min(n_splits, max(2, n_rows // 10))
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(np.arange(n_rows)))


def train_residual_models(feature_df, target_columns=TARGET_COLUMNS, feature_columns=FEATURE_COLUMNS,
                           group_col="trace_id", n_splits=5, params=None):
    """Train one LightGBM regressor per residual target.

    Returns (models, oof_predictions):
    - models: {target_name: fitted LGBMRegressor}, each trained on the full
      feature_df (the deployed model should use all available data).
    - oof_predictions: DataFrame aligned to feature_df's index with one
      column per target, holding out-of-fold predictions from the CV used
      to evaluate `models` honestly.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    X = _prepare_X(feature_df, feature_columns)

    groups = feature_df[group_col].to_numpy() if group_col in feature_df.columns else None
    folds = _make_folds(len(X), groups, n_splits)

    models = {}
    oof = pd.DataFrame(index=feature_df.index)

    for target in target_columns:
        y = feature_df[target].to_numpy()
        oof_pred = np.full(len(y), np.nan)

        for train_idx, val_idx in folds:
            fold_model = lgb.LGBMRegressor(**params)
            fold_model.fit(X.iloc[train_idx], y[train_idx], categorical_feature=CATEGORICAL_FEATURES)
            oof_pred[val_idx] = fold_model.predict(X.iloc[val_idx])

        oof[target] = oof_pred
        rmse = float(np.sqrt(np.nanmean((oof_pred - y) ** 2)))
        print(f"[train] {target}: out-of-fold RMSE = {rmse:.4f}")

        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X, y, categorical_feature=CATEGORICAL_FEATURES)
        models[target] = final_model

    return models, oof


def save_models(models, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for target, model in models.items():
        model.booster_.save_model(str(out_dir / f"{target}.txt"))


def load_model(path):
    return lgb.Booster(model_file=str(path))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM residual models from a features CSV "
                                                   "(see features.build_features_and_targets)")
    parser.add_argument("features_csv", help="CSV produced by concatenating build_features_and_targets() "
                                              "output across traces")
    parser.add_argument("--out-dir", default="models", help="directory to save trained model files")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    feature_df = pd.read_csv(args.features_csv)
    models, oof = train_residual_models(feature_df, n_splits=args.n_splits)
    save_models(models, args.out_dir)
    oof.to_csv(Path(args.out_dir) / "oof_predictions.csv", index=False)
    print(f"[train] saved {len(models)} model(s) to {args.out_dir}/")


if __name__ == "__main__":
    main()
