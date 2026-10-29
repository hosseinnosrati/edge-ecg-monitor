"""Unit test for DSP filtering and Pan-Tompkins peak detector."""

import numpy as np
from src.dsp.filters import clean_ecg, butter_bandpass_filter, notch_filter
from src.dsp.pan_tompkins import PanTompkinsDetector


def test_filters_and_peak_detector():
    fs = 360.0
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Synthetic ECG-like signal with simulated heartbeat at ~1.2 Hz (72 bpm)
    # Add R-peaks
    ecg = np.zeros_like(t)
    heartbeat_period = int(fs / 1.2)
    expected_peaks = []
    for idx in range(heartbeat_period // 2, len(t), heartbeat_period):
        # QRS spike
        ecg[idx] = 1.5
        expected_peaks.append(idx)

    # Add 50 Hz powerline interference
    noise_50hz = 0.4 * np.sin(2 * np.pi * 50 * t)
    # Add baseline wander (0.2 Hz)
    baseline_wander = 0.5 * np.sin(2 * np.pi * 0.2 * t)

    noisy_signal = ecg + noise_50hz + baseline_wander

    # Test clean_ecg
    cleaned = clean_ecg(noisy_signal, fs=fs, powerline_freq=50.0)
    assert len(cleaned) == len(noisy_signal)
    assert np.var(cleaned) < np.var(noisy_signal), "Filtering should reduce noise variance"

    # Test Pan-Tompkins
    detector = PanTompkinsDetector(fs=fs)
    r_peaks, mwi = detector.detect_peaks(cleaned)

    print(f"Expected ~{len(expected_peaks)} peaks, detected {len(r_peaks)} peaks.")
    assert len(r_peaks) >= len(expected_peaks) - 1, "Should detect virtually all synthetic beats"
    print("All DSP tests passed successfully!")


if __name__ == "__main__":
    test_filters_and_peak_detector()

