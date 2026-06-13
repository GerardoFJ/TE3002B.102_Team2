"""Pure-numpy XGBoost-regression tree predictor.

Loads an XGBoost-JSON model and walks the trees in Python. Lets the
Jetson run inference without installing the xgboost Python package
(handy when the device has no internet / no wheel available for its
specific aarch64 + Python combo).

Supports models trained with `objective="reg:squarederror"` and a
gbtree booster — exactly what our `train_xgboost.py` produces.

Tested against `xgboost.Booster.predict` — outputs match to within
float32 rounding.
"""

import json
from typing import List

import numpy as np


class TinyXGBPredictor:
    """Load an XGBoost JSON model and predict in pure numpy."""

    def __init__(self, model_path: str):
        with open(model_path) as f:
            m = json.load(f)

        learner = m["learner"]
        # XGBoost stores base_score as a JSON-serialized string of a 1-element
        # array, e.g. "[-2.2988129E-1]".
        base_raw = learner["learner_model_param"]["base_score"]
        if isinstance(base_raw, str):
            base_raw = base_raw.strip("[]")
        self.base_score = float(base_raw)

        booster = learner["gradient_booster"]["model"]
        trees = booster["trees"]

        # Compile each tree into parallel numpy arrays for fast access.
        self._lefts: List[np.ndarray] = []
        self._rights: List[np.ndarray] = []
        self._splits: List[np.ndarray] = []
        self._conds: List[np.ndarray] = []
        self._weights: List[np.ndarray] = []
        self._default_lefts: List[np.ndarray] = []
        for tree in trees:
            self._lefts.append(np.asarray(tree["left_children"], dtype=np.int32))
            self._rights.append(np.asarray(tree["right_children"], dtype=np.int32))
            self._splits.append(np.asarray(tree["split_indices"], dtype=np.int32))
            self._conds.append(np.asarray(tree["split_conditions"], dtype=np.float32))
            self._weights.append(np.asarray(tree["base_weights"], dtype=np.float32))
            self._default_lefts.append(
                np.asarray(tree["default_left"], dtype=bool)
            )
        self.n_trees = len(trees)

    def predict_one(self, features: np.ndarray) -> float:
        """Predict a single sample. features is a 1D numpy array."""
        total = self.base_score
        for k in range(self.n_trees):
            left = self._lefts[k]
            right = self._rights[k]
            splits = self._splits[k]
            conds = self._conds[k]
            weights = self._weights[k]
            default_left = self._default_lefts[k]

            node = 0
            # Walk until leaf. XGBoost marks leaves with left_children == -1.
            while left[node] != -1:
                feat = features[splits[node]]
                if np.isnan(feat):
                    node = left[node] if default_left[node] else right[node]
                else:
                    # XGBoost's split is "feature < split_condition → left".
                    if feat < conds[node]:
                        node = left[node]
                    else:
                        node = right[node]
            total += weights[node]
        return float(total)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a batch. X has shape (n_samples, n_features)."""
        return np.asarray(
            [self.predict_one(X[i]) for i in range(X.shape[0])],
            dtype=np.float32,
        )
