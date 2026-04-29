"""
LightGBM classifier for synthetic speech detection.
Operates on tabular MFCC/LFCC statistics (mean + std per file).
"""

import numpy as np
import joblib
import lightgbm as lgb
from typing import Optional, Dict, Any


DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}


class LightGBMClassifier:
    """
    Wrapper around LGBMClassifier with early stopping on a validation set.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_names: Optional[list] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMClassifier":
        """
        Train LightGBM. If validation data is provided, uses early stopping.
        """
        self.model = lgb.LGBMClassifier(**self.params)

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not trained."
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Model not trained."
        return self.model.predict_proba(X)

    def feature_importance(self) -> np.ndarray:
        assert self.model is not None, "Model not trained."
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        assert self.model is not None, "No model to save."
        joblib.dump({"model": self.model, "params": self.params}, path)
        print(f"LightGBM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LightGBMClassifier":
        data = joblib.load(path)
        obj = cls(params=data["params"])
        obj.model = data["model"]
        return obj
