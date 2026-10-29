"""Frequency-Domain Spectral Analysis of Heart Rate Variability (HRV).

Computes Power Spectral Density (PSD) via Welch's method to extract
clinical autonomic nervous system metrics:
- VLF (Very Low Frequency): < 0.04 Hz
- LF (Low Frequency): 0.04 - 0.15 Hz (Sympathetic & Baroreflex modulation)
- HF (High Frequency): 0.15 - 0.40 Hz (Parasympathetic / Vagal tone)
- LF/HF Ratio: Clinical Sympathovagal Balance Index

Reference:
Task Force of the European Society of Cardiology and the North American
Society of Pacing and Electrophysiology (1996). Heart rate variability:
standards of measurement, physiological interpretation, and clinical use.
Circulation, 93(5), 1043-1065.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import signal
from scipy import interpolate


def compute_hrv_spectral_metrics(
    r_peaks: np.ndarray,
    fs: float = 360.0,
    interpolation_rate: float = 4.0
) -> Dict[str, Any]:
    """Compute frequency-domain HRV metrics using Welch's Power Spectral Density.

    Parameters
    ----------
    r_peaks : np.ndarray
        Indices of detected R-peaks in sample counts.
    fs : float
        Sampling frequency of the raw ECG signal (Hz).
    interpolation_rate : float
        Uniform resampling frequency for the non-uniform RR intervals (standard 4.0 Hz).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing VLF, LF, HF power, total power, LF/HF ratio,
        normalized LF/HF units, and the frequency/PSD vectors for plotting.
    """
    if len(r_peaks) < 4:
        return {
            "vlf_power": 0.0,
            "lf_power": 0.0,
            "hf_power": 0.0,
            "total_power": 0.0,
            "lf_hf_ratio": 0.0,
            "lf_nu": 0.0,
            "hf_nu": 0.0,
            "frequencies": np.array([]),
            "psd": np.array([])
        }

    # Time locations of R-peaks in seconds
    peak_times = r_peaks / fs
    # RR intervals in seconds
    rr_intervals = np.diff(peak_times)
    # Time points for RR intervals (assign to midpoint between beats)
    rr_times = peak_times[1:]

    # Minimum and maximum time bounds
    t_start = rr_times[0]
    t_end = rr_times[-1]

    # Uniform time grid for interpolation
    num_samples = int((t_end - t_start) * interpolation_rate)
    if num_samples < 8:
        # Fallback if window is too short
        num_samples = 16

    t_uniform = np.linspace(t_start, t_end, num_samples)

    # Cubic spline interpolation of non-uniformly sampled RR intervals
    if len(rr_times) > 3:
        spline = interpolate.interp1d(rr_times, rr_intervals, kind="cubic", fill_value="extrapolate")
    else:
        spline = interpolate.interp1d(rr_times, rr_intervals, kind="linear", fill_value="extrapolate")

    rr_uniform = spline(t_uniform)

    # Remove DC component (mean) to focus on spectral variations
    rr_detrended = rr_uniform - np.mean(rr_uniform)

    # Compute Power Spectral Density (PSD) using Welch's method
    nperseg = min(len(rr_detrended), 256)
    if nperseg < 16:
        nperseg = len(rr_detrended)

    freqs, psd = signal.welch(
        rr_detrended,
        fs=interpolation_rate,
        window="hann",
        nperseg=nperseg,
        scaling="density"
    )

    # Frequency band masks
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

    # Trapezoidal integration for band powers (converted to ms^2 by multiplying by 1e6)
    # Since RR is in seconds, PSD is in s^2/Hz. Multiply by 1e6 gives ms^2.
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    vlf_power = float(np.sum(psd[vlf_mask]) * df * 1e6)
    lf_power = float(np.sum(psd[lf_mask]) * df * 1e6)
    hf_power = float(np.sum(psd[hf_mask]) * df * 1e6)
    total_power = vlf_power + lf_power + hf_power

    # Sympathovagal balance (LF / HF)
    lf_hf_ratio = float(lf_power / (hf_power + 1e-8))

    # Normalized units (nu)
    lf_nu = float((lf_power / (lf_power + hf_power + 1e-8)) * 100.0)
    hf_nu = float((hf_power / (lf_power + hf_power + 1e-8)) * 100.0)

    return {
        "vlf_power_ms2": vlf_power,
        "lf_power_ms2": lf_power,
        "hf_power_ms2": hf_power,
        "total_power_ms2": total_power,
        "lf_hf_ratio": lf_hf_ratio,
        "lf_nu": lf_nu,
        "hf_nu": hf_nu,
        "frequencies": freqs,
        "psd": psd * 1e6  # in ms^2/Hz
    }

