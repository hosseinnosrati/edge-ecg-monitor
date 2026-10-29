"""Training and Evaluation Script for Edge ECG Arrhythmia Classifier.

Processes MIT-BIH records through the DSP pipeline, extracts features,
trains the lightweight model, and evaluates classification performance.
"""

import os
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.data_loader import load_record, download_sample_mitbih_records
from src.dsp.filters import clean_ecg
from src.features.hrv import extract_beat_features
from src.models.classifier import EdgeArrhythmiaClassifier, FEATURE_NAMES


def build_dataset_from_records(records=("100", "119"), data_dir="data/mitbih") -> Tuple[np.ndarray, np.ndarray]:
    download_sample_mitbih_records(target_dir=data_dir, records=list(records))

    all_features = []
    all_labels = []

    for rec in records:
        print(f"Processing MIT-BIH record {rec}...")
        raw_signal, ann_samples, ann_symbols, fs = load_record(rec, data_dir=data_dir)

        # 1. Clean signal via DSP pipeline
        cleaned_signal = clean_ecg(raw_signal, fs=fs, powerline_freq=60.0)

        # 2. Extract beat features using cardiologist-annotated R-peaks
        beat_features = extract_beat_features(cleaned_signal, ann_samples, fs=fs)

        # Map each extracted beat to its annotation symbol
        for bf in beat_features:
            beat_idx = bf["beat_index"]
            sym = ann_symbols[beat_idx]

            # Binary labeling: Normal vs Ventricular Ectopic (PVC)
            if sym in ["N", "L", "R"]:
                label = 0  # Normal
            elif sym in ["V", "E"]:
                label = 1  # Ventricular Arrhythmia
            else:
                continue  # Skip other minor classes for clean binary benchmark

            feature_vector = [bf[feat] for feat in FEATURE_NAMES]
            all_features.append(feature_vector)
            all_labels.append(label)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    print(f"Dataset constructed: {X.shape[0]} total beats ({np.sum(y == 0)} Normal, {np.sum(y == 1)} Arrhythmia).")
    return X, y


def train():
    X, y = build_dataset_from_records(records=("100", "119"))

    # Train / Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} beats | Test set: {X_test.shape[0]} beats")

    classifier = EdgeArrhythmiaClassifier(n_estimators=50, max_depth=8)
    classifier.fit(X_train, y_train)

    # Evaluation
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("       MODEL EVALUATION (MIT-BIH BENCHMARK)")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Normal (N)", "Ventricular Arrhythmia (V)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {auc:.4f}")
    print("=" * 50)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_path = "models/edge_classifier.joblib"
    classifier.save(model_path)


if __name__ == "__main__":
    train()

