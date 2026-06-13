#!/usr/bin/env python3
"""Train an XGBoost regressor on a vision-oracle-labelled dataset.

Reads a CSV produced by extract_labels_from_bag.py, fits XGBoost, prints
metrics, and saves the model as JSON for the runtime ml_steering_node.

Usage:
  python3 train_xgboost.py <dataset.csv> [--output model.json] \\
                          [--test-size 0.2] [--seed 42]
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    from sklearn.ensemble import HistGradientBoostingRegressor


FEATURE_COLS = [
    "n_lines", "mean_angle", "std_angle", "mean_abs_angle",
    "mean_length", "max_length", "left_count", "right_count", "balance_lr",
    "center_bottom", "center_error", "vanishing_x", "vanishing_error",
    "confidence",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Dataset CSV from extract_labels_from_bag.py")
    ap.add_argument("--output", default="xgb_steering.json",
                    help="Output model path (json).")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "omega_true" not in df.columns:
        sys.exit("CSV missing omega_true column")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"CSV missing feature columns: {missing}")

    X = df[FEATURE_COLS].values
    y = df["omega_true"].values

    print(f"Total samples       : {len(df)}")
    print(f"omega_true range    : [{y.min():.3f}, {y.max():.3f}]  "
          f"mean={y.mean():+.3f}  std={y.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed)
    print(f"train/test          : {len(X_tr)} / {len(X_te)}")
    print()

    if HAVE_XGB:
        model = XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
        )
        label = "XGBoost"
    else:
        # Fallback if xgboost isn't installed — same family, ships with sklearn.
        model = HistGradientBoostingRegressor(
            max_iter=args.n_estimators,
            learning_rate=args.learning_rate,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=args.seed,
        )
        label = "HistGradientBoosting fallback"

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_te = model.predict(X_te)
    predict_ms_per_sample = (time.perf_counter() - t0) * 1000 / max(1, len(X_te))
    pred_tr = model.predict(X_tr)

    metrics = {
        "model": label,
        "MAE_test":  mean_absolute_error(y_te, pred_te),
        "RMSE_test": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "R2_test":   r2_score(y_te, pred_te),
        "MAE_train":  mean_absolute_error(y_tr, pred_tr),
        "RMSE_train": float(np.sqrt(mean_squared_error(y_tr, pred_tr))),
        "R2_train":   r2_score(y_tr, pred_tr),
        "train_s": train_s,
        "predict_ms_per_sample": predict_ms_per_sample,
    }
    print("== metrics ==")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:24s} {v:+.5f}")
        else:
            print(f"  {k:24s} {v}")

    if HAVE_XGB:
        model.save_model(args.output)
    else:
        # HistGradientBoostingRegressor doesn't have a JSON save; pickle.
        import joblib
        if args.output.endswith(".json"):
            args.output = args.output[:-5] + ".joblib"
        joblib.dump(model, args.output)

    print(f"\nSaved model: {args.output}")

    # Feature importance — sanity check that the model leans on the
    # geometric features we expect (mean_angle, max_length, etc.).
    try:
        importances = model.feature_importances_
        imp = sorted(
            zip(FEATURE_COLS, importances),
            key=lambda kv: kv[1], reverse=True,
        )
        print("\nfeature importance (top):")
        for name, val in imp:
            print(f"  {name:20s} {val:.4f}")
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
