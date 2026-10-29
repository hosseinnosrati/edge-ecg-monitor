"""Feature Extraction Module for ECG Beat Analysis and Arrhythmia Detection.

Extracts temporal (RR intervals, HRV) and morphological features
for each segmented cardiac cycle (beat).
"""

from typing import List, Dict, Any
import numpy as np


def extract_beat_features(
    ecg_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float = 360.0
) -> List[Dict[str, Any]]:
    """Segment ECG signal around R-peaks and extract temporal & morphological features.

    Parameters
    ----------
    ecg_signal : np.ndarray
        Cleaned 1D ECG array.
    r_peaks : np.ndarray
        Array of indices for detected R-peaks.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    List[Dict[str, Any]]
        List of feature dictionaries, one for each valid beat.
    """
    if len(r_peaks) < 3:
        return []

    # Window around R-peak: 100ms before, 150ms after (standard QRS complex)
    pre_samples = int(0.100 * fs)
    post_samples = int(0.150 * fs)

    # Compute RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000.0  # in ms
    mean_rr = np.mean(rr_intervals)

    features = []

    for i in range(1, len(r_peaks) - 1):
        r_idx = r_peaks[i]

        # Boundary check
        if r_idx - pre_samples < 0 or r_idx + post_samples >= len(ecg_signal):
            continue

        beat_window = ecg_signal[r_idx - pre_samples : r_idx + post_samples]

        # 1. Temporal Features (Intervals & Local Dynamics)
        pre_rr = rr_intervals[i - 1]  # ms between peak(i-1) and peak(i)
        post_rr = rr_intervals[i]     # ms between peak(i) and peak(i+1)
        rr_ratio = pre_rr / (post_rr + 1e-6)
        local_relative_rr = pre_rr / (mean_rr + 1e-6)

        # 2. Morphological Features
        qrs_amplitude = np.max(beat_window) - np.min(beat_window)
        qrs_energy = np.sum(np.square(beat_window))
        qrs_variance = np.var(beat_window)

        # Estimate QRS width: duration where amplitude exceeds 30% of peak amplitude
        thresh = np.min(beat_window) + 0.3 * qrs_amplitude
        above_thresh = np.where(beat_window > thresh)[0]
        qrs_width_ms = (len(above_thresh) / fs) * 1000.0 if len(above_thresh) > 0 else 0.0

        features.append({
            "beat_index": i,
            "r_peak_sample": int(r_idx),
            "pre_rr_ms": float(pre_rr),
            "post_rr_ms": float(post_rr),
            "rr_ratio": float(rr_ratio),
            "local_relative_rr": float(local_relative_rr),
            "qrs_amplitude": float(qrs_amplitude),
            "qrs_energy": float(qrs_energy),
            "qrs_variance": float(qrs_variance),
            "qrs_width_ms": float(qrs_width_ms),
        })

    return features
