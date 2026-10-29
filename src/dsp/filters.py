"""Digital Signal Processing (DSP) Filters for Biomedical ECG Signals.

Designed for real-time and edge processing with minimal computational overhead.
Complies with AAMI recommendations for ECG bandwidth (0.5 Hz - 40 Hz).
"""

import numpy as np
from scipy import signal


def notch_filter(data: np.ndarray, fs: float = 360.0, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Apply an IIR notch filter to remove powerline interference (50 Hz or 60 Hz).

    Parameters
    ----------
    data : np.ndarray
        Raw ECG signal.
    fs : float
        Sampling frequency in Hz (MIT-BIH standard is 360 Hz).
    freq : float
        Frequency to reject (default 50.0 Hz).
    q : float
        Quality factor determining filter bandwidth.

    Returns
    -------
    np.ndarray
        Filtered signal with notch rejection applied.
    """
    b, a = signal.iirnotch(w0=freq, Q=q, fs=fs)
    # Using filtfilt for zero-phase distortion on batches/windows
    filtered = signal.filtfilt(b, a, data)
    return filtered


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 360.0,
    order: int = 3
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to isolate ECG bandwidth and eliminate

    baseline wander (<0.5 Hz) and high-frequency electromyographic (EMG) noise (>40 Hz).

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (3rd order gives steep roll-off with low complexity).

    Returns
    -------
    np.ndarray
        Bandpass-filtered ECG signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, data)
    return filtered


def clean_ecg(
    raw_signal: np.ndarray,
    fs: float = 360.0,
    powerline_freq: float = 50.0
) -> np.ndarray:
    """Complete baseline DSP denoising pipeline.

    Applies powerline notch filter followed by bandpass filter.
    """
    notched = notch_filter(raw_signal, fs=fs, freq=powerline_freq)
    cleaned = butter_bandpass_filter(notched, lowcut=0.5, highcut=40.0, fs=fs, order=3)
    return cleaned

