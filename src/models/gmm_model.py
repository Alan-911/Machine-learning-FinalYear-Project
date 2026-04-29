"""
GMM-based binary classifier for synthetic speech detection.
Trains one GMM per class (bonafide, spoof) and classifies via log-likelihood ratio.
This mirrors the classic i-vector/GMM-UBM paradigm from speaker verification.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.mixture import GaussianMixture
from typing import Optional, Tuple


class GMMClassifier:
    """
    Binary classifier using a pair of GMMs (one per class).
    Score = log P(x | GMM_bonafide) - log P(x | GMM_spoof)
    Positive score → predicted bonafide, negative → spoof.
    """

    def __init__(
        self,
        n_components: int = 256,
        covariance_type: str = "diag",
        max_iter: int = 100,
        n_init: int = 1,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.gmm_bonafide: Optional[GaussianMixture] = None
        self.gmm_spoof: Optional[GaussianMixture] = None
        self.threshold: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMClassifier":
        """
        Train two GMMs on bonafide (y=1) and spoof (y=0) samples.
        X: (n_samples, n_features) — flat feature vectors (mean+std of frames)
        y: (n_samples,) — binary labels
        """
        X_bonafide = X[y == 1]
        X_spoof = X[y == 0]

        print(f"Training GMM bonafide on {len(X_bonafide)} samples...")
        self.gmm_bonafide = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        ).fit(X_bonafide)

        print(f"Training GMM spoof on {len(X_spoof)} samples...")
        self.gmm_spoof = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        ).fit(X_spoof)

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return log-likelihood ratio scores. Higher = more likely bonafide."""
        assert self.gmm_bonafide is not None and self.gmm_spoof is not None, \
            "Model not trained. Call fit() first."
        llr = self.gmm_bonafide.score_samples(X) - self.gmm_spoof.score_samples(X)
        return llr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft probability estimates via sigmoid on LLR scores."""
        llr = self.score_samples(X)
        prob_bonafide = 1.0 / (1.0 + np.exp(-llr))
        return np.stack([1 - prob_bonafide, prob_bonafide], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.score_samples(X) > self.threshold).astype(int)

    def save(self, path: str) -> None:
        joblib.dump(
            {"gmm_bonafide": self.gmm_bonafide, "gmm_spoof": self.gmm_spoof,
             "threshold": self.threshold, "n_components": self.n_components},
            path,
        )
        print(f"GMM model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "GMMClassifier":
        data = joblib.load(path)
        model = cls(n_components=data["n_components"])
        model.gmm_bonafide = data["gmm_bonafide"]
        model.gmm_spoof = data["gmm_spoof"]
        model.threshold = data["threshold"]
        return model
