"""Interactive Visual Demo: Real-Time ECG Stream, Filtering, and Arrhythmia Detection.

Generates a clear visual figure showing:
1. Raw Noisy ECG (with simulated powerline hum & baseline wander)
2. DSP Denoised Signal + Pan-Tompkins Peak Detection
3. AI Classification (Green = Normal, Red = Ventricular Arrhythmia / PVC)
"""

import os
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import load_record
from src.dsp.filters import clean_ecg
from src.dsp.pan_tompkins import PanTompkinsDetector
from src.features.hrv import extract_beat_features
from src.models.classifier import EdgeArrhythmiaClassifier, FEATURE_NAMES


def run_demo():
    print("Running Edge ECG Arrhythmia Demo...")
    data_dir = "data/mitbih"
    record_name = "119"  # Record 119 has prominent PVC arrhythmias

    raw_signal, _, _, fs = load_record(record_name, data_dir=data_dir)

    # Pick a 10-second segment (samples 0 to 3600)
    duration_sec = 10
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    segment_raw = raw_signal[:n_samples]

    # Inject realistic real-world interference (60Hz powerline hum + 0.2Hz baseline respiration wander)
    powerline_noise = 0.25 * np.sin(2 * np.pi * 60.0 * t)
    baseline_wander = 0.40 * np.sin(2 * np.pi * 0.25 * t)
    noisy_ecg = segment_raw + powerline_noise + baseline_wander

    # 1. Apply DSP filtering
    cleaned_ecg = clean_ecg(noisy_ecg, fs=fs, powerline_freq=60.0)

    # 2. Pan-Tompkins Detection
    detector = PanTompkinsDetector(fs=fs)
    r_peaks, mwi_signal = detector.detect_peaks(cleaned_ecg)

    # 3. Extract Features
    beat_features = extract_beat_features(cleaned_ecg, r_peaks, fs=fs)

    # 4. Load AI Model & Predict
    classifier = EdgeArrhythmiaClassifier.load("models/edge_classifier.joblib")
    X = np.array([[bf[f] for f in FEATURE_NAMES] for bf in beat_features], dtype=np.float32)
    predictions = classifier.predict(X)

    # 5. Plotting publication-quality figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # Subplot 1: Noisy Signal
    axes[0].plot(t, noisy_ecg, color="#d9534f", linewidth=1.1, label="Raw ECG + 60Hz Powerline + Baseline Drift")
    axes[0].set_title("Stage 1: Raw Real-World ECG Input (Noise & Respiration Artifacts)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    # Subplot 2: Pan-Tompkins MWI Envelope
    axes[1].plot(t, mwi_signal, color="#0275d8", linewidth=1.2, label="Moving Window Integration (MWI) Envelope")
    axes[1].set_title("Stage 2: Pan-Tompkins DSP Pipeline (Bandpass 5-15Hz -> Derivative -> Squaring -> MWI)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Energy (a.u.)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    # Subplot 3: Cleaned ECG + AI Arrhythmia Classification
    axes[2].plot(t, cleaned_ecg, color="#292b2c", linewidth=1.2, label="Cleaned ECG Signal")

    # Annotate classified beats
    for idx, (bf, pred) in enumerate(zip(beat_features, predictions)):
        peak_sample = bf["r_peak_sample"]
        peak_time = peak_sample / fs
        peak_val = cleaned_ecg[peak_sample]

        if pred == 0:
            color = "#28a745"  # Green
            label = "Normal" if idx == 0 else ""
            axes[2].scatter(peak_time, peak_val, color=color, s=80, zorder=5, label=label)
        else:
            color = "#dc3545"  # Red
            label = "Ventricular Arrhythmia (PVC)" if "PVC" not in [l.get_label() for l in axes[2].lines] else ""
            axes[2].scatter(peak_time, peak_val, color=color, s=120, marker="^", edgecolors="black", zorder=6, label=label)
            axes[2].annotate(
                "ARRHYTHMIA (PVC)",
                xy=(peak_time, peak_val),
                xytext=(peak_time - 0.25, peak_val + 0.35),
                arrowprops=dict(arrowstyle="->", color="#dc3545", lw=1.5),
                fontsize=9,
                fontweight="bold",
                color="#dc3545",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffebee", edgecolor="#dc3545", alpha=0.8)
            )

    axes[2].set_title("Stage 3: Denoised ECG with AI-Driven Beat Classification (Green: Normal, Red: PVC)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Time (seconds)", fontsize=11)
    axes[2].set_ylabel("Amplitude (mV)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Unique legend handles
    handles, labels = axes[2].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[2].legend(by_label.values(), by_label.keys(), loc="upper right")

    os.makedirs("assets", exist_ok=True)
    out_path = "assets/demo_ecg_detection.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Demo plot successfully saved to {out_path}!")
    print(f"Total Beats Analyzed: {len(beat_features)}")
    print(f"Normal Beats:         {np.sum(predictions == 0)}")
    print(f"Arrhythmia (PVC):     {np.sum(predictions == 1)}")


if __name__ == "__main__":
    run_demo()

