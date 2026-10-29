"""Machine Learning Model for Cardiac Arrhythmia Classification on Edge Devices.

Implements lightweight, efficient classifiers designed for low-memory microprocessors.
Distinguishes between Normal beats (Sinus Rhythm) and Ventricular Ectopic Beats (PVC).
"""

import os
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "pre_rr_ms",
    "post_rr_ms",
    "rr_ratio",
    "local_relative_rr",
    "qrs_amplitude",
    "qrs_energy",
    "qrs_variance",
    "qrs_width_ms",
]


class EdgeArrhythmiaClassifier:
    """Lightweight classifier wrapper optimized for edge deployment."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 8):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=1  # Sequential processing matches single-threaded edge devices
        )
        self.classes_ = ["Normal", "Ventricular_Arrhythmia"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, filepath: str = "models/edge_classifier.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "classes": self.classes_}, filepath)
        print(f"Model saved successfully to {filepath} ({os.path.getsize(filepath) / 1024:.1f} KB)")

    @classmethod
    def load(cls, filepath: str = "models/edge_classifier.joblib") -> "EdgeArrhythmiaClassifier":
        checkpoint = joblib.load(filepath)
        instance = cls()
        instance.model = checkpoint["model"]
        instance.scaler = checkpoint["scaler"]
        instance.classes_ = checkpoint["classes"]
        return instance
