"""Pan-Tompkins Algorithm for Real-Time QRS Detection in ECG Signals.

Reference:
Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
IEEE Transactions on Biomedical Engineering, (3), 230-236.
"""

from typing import List, Tuple
import numpy as np
from scipy import signal


class PanTompkinsDetector:
    """Real-time capable implementation of the Pan-Tompkins QRS detection algorithm.

    Optimized for low-resource microprocessors and edge computing platforms.
    """

    def __init__(self, fs: float = 360.0):
        """Parameters

        ----------
        fs : float
            Sampling rate of the ECG signal in Hz (standard MIT-BIH is 360 Hz).
        """
        self.fs = fs
        # Moving window integration length ~150ms
        self.integration_window = int(0.150 * fs)
        # Refractory period: minimum time between consecutive R-peaks ~200ms
        self.refractory_samples = int(0.200 * fs)

    def bandpass_stage(self, ecg: np.ndarray) -> np.ndarray:
        """Stage 1: Bandpass filter (5 - 15 Hz) to isolate QRS frequency band."""
        nyq = 0.5 * self.fs
        low = 5.0 / nyq
        high = 15.0 / nyq
        b, a = signal.butter(1, [low, high], btype="band")
        return signal.filtfilt(b, a, ecg)

    def derivative_stage(self, ecg: np.ndarray) -> np.ndarray:
        """Stage 2: 5-point derivative to obtain slope information.

        Transfer function: H(z) = (1/8T) * (-2z^-2 - z^-1 + z^1 + 2z^2)
        Approximated digitally using centered 5-point stencil: [-2, -1, 0, 1, 2] / 8.
        """
        kernel = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / (8.0 / self.fs)
        return signal.convolve(ecg, kernel, mode="same")

    def squaring_stage(self, ecg: np.ndarray) -> np.ndarray:
        """Stage 3: Pointwise squaring to amplify large slopes and enforce positivity."""
        return np.square(ecg)

    def moving_window_integration(self, ecg: np.ndarray) -> np.ndarray:
        """Stage 4: Moving Window Integrator (MWI) to extract waveform envelope."""
        kernel = np.ones(self.integration_window) / self.integration_window
        return signal.convolve(ecg, kernel, mode="same")

    def detect_peaks(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Execute full Pan-Tompkins pipeline and return R-peak indices.

        Parameters
        ----------
        ecg_signal : np.ndarray
            Cleaned or raw 1D ECG array.

        Returns
        -------
        r_peaks : np.ndarray
            Indices of detected R-peaks mapped back to the original signal.
        integrated_signal : np.ndarray
            The MWI envelope signal for visualization and diagnostics.
        """
        # 1. Bandpass filter (5-15 Hz)
        filtered = self.bandpass_stage(ecg_signal)

        # 2. Derivative
        derivative = self.derivative_stage(filtered)

        # 3. Squaring
        squared = self.squaring_stage(derivative)

        # 4. Moving Window Integration
        integrated = self.moving_window_integration(squared)

        # 5. Adaptive Thresholding
        # Initial estimate based on mean signal power
        noise_level = np.mean(integrated)
        signal_level = np.max(integrated) * 0.33
        threshold = noise_level + 0.25 * (signal_level - noise_level)

        # Find local peaks in the integrated signal
        mwi_peaks, _ = signal.find_peaks(
            integrated,
            height=threshold,
            distance=self.refractory_samples
        )

        # Map MWI peaks to true R-peaks in original signal (look in local window)
        search_radius = int(0.100 * self.fs)  # 100ms search window
        true_r_peaks: List[int] = []

        for p in mwi_peaks:
            start = max(0, p - search_radius)
            end = min(len(ecg_signal), p + search_radius)
            if start < end:
                # Find the local maximum in the original signal window
                local_peak = start + np.argmax(ecg_signal[start:end])
                if not true_r_peaks or (local_peak - true_r_peaks[-1] >= self.refractory_samples):
                    true_r_peaks.append(int(local_peak))

        return np.array(true_r_peaks, dtype=int), integrated

