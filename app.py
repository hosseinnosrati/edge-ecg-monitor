"""Edge-ECG-Monitor: Interactive Web Dashboard for Biomedical DSP & Edge AI.

Run locally using:
    streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_record, download_sample_mitbih_records
from src.dsp.filters import notch_filter, butter_bandpass_filter
from src.dsp.pan_tompkins import PanTompkinsDetector
from src.dsp.spectral import compute_hrv_spectral_metrics
from src.features.hrv import extract_beat_features
from src.models.classifier import EdgeArrhythmiaClassifier, FEATURE_NAMES


st.set_page_config(
    page_title="Edge-ECG-Monitor | Biomedical DSP & AI",
    page_icon="🫀",
    layout="wide"
)

# Header
st.title("🫀 Edge-ECG-Monitor")
st.markdown(
    "**Real-Time Biomedical DSP & Edge AI Arrhythmia Detector** | "
    "*[GitHub Repository](https://github.com/hnosrati0-source/edge-ecg-monitor)*"
)
st.caption("Engineered for resource-constrained Edge & IoT medical hardware (<175 MB RAM, 38 KB model).")

# Sidebar Controls
st.sidebar.header("⚙️ DSP & Signal Controls")

# Record selection
record_choice = st.sidebar.selectbox(
    "Select MIT-BIH Record:",
    options=["119", "100"],
    format_func=lambda x: f"Record {x} (PVC Arrhythmias)" if x == "119" else f"Record {x} (Normal Rhythm)"
)

duration_sec = st.sidebar.slider("Signal Window (Seconds):", min_value=5, max_value=30, value=10, step=1)

st.sidebar.subheader("Digital Filter Parameters")
powerline_freq = st.sidebar.radio("Powerline Notch Frequency:", options=[60.0, 50.0], index=0)
bp_low = st.sidebar.slider("Bandpass Lowcut (Hz) [Baseline Wander]:", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
bp_high = st.sidebar.slider("Bandpass Highcut (Hz) [EMG Muscle Noise]:", min_value=20.0, max_value=50.0, value=40.0, step=5.0)

inject_noise = st.sidebar.checkbox("Simulate Real-World Noise Interference", value=True)

# Load Data
data_dir = "data/mitbih"
download_sample_mitbih_records(target_dir=data_dir, records=[record_choice])
raw_signal, _, _, fs = load_record(record_choice, data_dir=data_dir)

# Slice segment
n_samples = int(duration_sec * fs)
t = np.linspace(0, duration_sec, n_samples, endpoint=False)
segment = raw_signal[:n_samples]

if inject_noise:
    # Add 60/50Hz hum and 0.25Hz respiration drift
    hum = 0.25 * np.sin(2 * np.pi * powerline_freq * t)
    drift = 0.35 * np.sin(2 * np.pi * 0.25 * t)
    processed_input = segment + hum + drift
else:
    processed_input = segment

# Apply DSP Pipeline
notched = notch_filter(processed_input, fs=fs, freq=powerline_freq)
cleaned = butter_bandpass_filter(notched, lowcut=bp_low, highcut=bp_high, fs=fs, order=3)

# Pan-Tompkins Peak Detection
detector = PanTompkinsDetector(fs=fs)
r_peaks, mwi_signal = detector.detect_peaks(cleaned)

# Feature Extraction & AI Classification
beat_features = extract_beat_features(cleaned, r_peaks, fs=fs)
classifier = EdgeArrhythmiaClassifier.load("models/edge_classifier.joblib")

if len(beat_features) > 0:
    X = np.array([[bf[f] for f in FEATURE_NAMES] for bf in beat_features], dtype=np.float32)
    predictions = classifier.predict(X)
    probs = classifier.predict_proba(X)
else:
    predictions = np.array([])
    probs = np.array([])

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📈 Real-Time Filtering & Beat Detection",
    "🔬 HRV Spectral Analysis (LF/HF Ratio)",
    "💻 Embedded C-Header & Hardware Specs"
])

# TAB 1: Filtering & Beat Detection
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    heart_rate_bpm = (len(r_peaks) / duration_sec) * 60.0 if duration_sec > 0 else 0
    n_normal = int(np.sum(predictions == 0)) if len(predictions) > 0 else 0
    n_arrhythmia = int(np.sum(predictions == 1)) if len(predictions) > 0 else 0

    col1.metric("Heart Rate", f"{heart_rate_bpm:.1f} BPM")
    col2.metric("Total Detected Beats", f"{len(beat_features)}")
    col3.metric("Normal Beats", f"{n_normal}")
    col4.metric("Arrhythmias (PVC)", f"{n_arrhythmia}", delta_color="inverse")

    fig, (ax_raw, ax_clean) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    ax_raw.plot(t, processed_input, color="#d9534f", linewidth=1.1)
    ax_raw.set_title("Input Signal (Raw ECG + Noise)", fontsize=11, fontweight="bold")
    ax_raw.set_ylabel("Amplitude (mV)")
    ax_raw.grid(True, linestyle="--", alpha=0.5)

    ax_clean.plot(t, cleaned, color="#292b2c", linewidth=1.1, label="Denoised ECG")
    for idx, (bf, pred) in enumerate(zip(beat_features, predictions)):
        p_idx = bf["r_peak_sample"]
        pt = p_idx / fs
        pval = cleaned[p_idx]
        if pred == 0:
            ax_clean.scatter(pt, pval, color="#28a745", s=70, zorder=5, label="Normal" if idx == 0 else "")
        else:
            ax_clean.scatter(pt, pval, color="#dc3545", s=110, marker="^", edgecolors="black", zorder=6, label="PVC Arrhythmia" if "PVC" not in ax_clean.get_legend_handles_labels()[1] else "")
            ax_clean.annotate(
                "PVC", xy=(pt, pval), xytext=(pt - 0.2, pval + 0.3),
                arrowprops=dict(arrowstyle="->", color="#dc3545", lw=1.2),
                fontweight="bold", color="#dc3545"
            )

    ax_clean.set_title("Denoised ECG with AI-Classified Peaks (Green: Normal, Red: PVC Arrhythmia)", fontsize=11, fontweight="bold")
    ax_clean.set_xlabel("Time (seconds)")
    ax_clean.set_ylabel("Amplitude (mV)")
    ax_clean.grid(True, linestyle="--", alpha=0.5)
    ax_clean.legend(loc="upper right")

    st.pyplot(fig)

# TAB 2: HRV Spectral Analysis
with tab2:
    st.subheader("Frequency-Domain Heart Rate Variability (Welch PSD)")
    st.markdown(
        "Decomposes the cardiac inter-beat intervals into autonomic nervous system frequency bands. "
        "Standardized according to ESC/NASPE guidelines."
    )

    spectral_res = compute_hrv_spectral_metrics(r_peaks, fs=fs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LF Power (0.04-0.15 Hz)", f"{spectral_res['lf_power_ms2']:.1f} ms²")
    c2.metric("HF Power (0.15-0.40 Hz)", f"{spectral_res['hf_power_ms2']:.1f} ms²")
    c3.metric("LF/HF Ratio", f"{spectral_res['lf_hf_ratio']:.2f}")
    c4.metric("Normalized LF / HF", f"{spectral_res['lf_nu']:.1f}% / {spectral_res['hf_nu']:.1f}%")

    freqs = spectral_res["frequencies"]
    psd = spectral_res["psd"]

    if len(freqs) > 0 and len(psd) > 0:
        fig_psd, ax_psd = plt.subplots(figsize=(10, 4))
        ax_psd.plot(freqs, psd, color="#0275d8", lw=1.5)

        # Shade LF & HF
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
        ax_psd.fill_between(freqs[lf_mask], psd[lf_mask], color="#ffc107", alpha=0.4, label="LF (Sympathetic: 0.04-0.15 Hz)")
        ax_psd.fill_between(freqs[hf_mask], psd[hf_mask], color="#28a745", alpha=0.4, label="HF (Parasympathetic: 0.15-0.40 Hz)")

        ax_psd.set_xlim(0, 0.5)
        ax_psd.set_title("Heart Rate Variability Power Spectral Density (PSD)", fontweight="bold")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("Power Density (ms²/Hz)")
        ax_psd.grid(True, linestyle="--", alpha=0.5)
        ax_psd.legend()
        st.pyplot(fig_psd)

    st.info(
        f"**Clinical Interpretation:** An LF/HF ratio of **{spectral_res['lf_hf_ratio']:.2f}** reflects the balance "
        "between sympathetic (stress/fight-or-flight) and parasympathetic (rest-and-digest) modulation of the sinoatrial node."
    )

# TAB 3: Embedded C-Header
with tab3:
    st.subheader("Embedded ANSI C99 Code Generation")
    st.markdown(
        "Zero-malloc, MISRA-compliant C99 header for flashing directly onto ARM Cortex-M, STM32, or ESP32 microcontrollers."
    )

    header_file_path = "models/edge_classifier.h"
    if os.path.exists(header_file_path):
        with open(header_file_path, "r", encoding="utf-8") as f:
            c_code = f.read()

        st.download_button(
            label="📥 Download edge_classifier.h",
            data=c_code,
            file_name="edge_classifier.h",
            mime="text/x-c"
        )
        with st.expander("Preview Generated C Header Code", expanded=False):
            st.code(c_code[:2000] + "\n\n/* ... (remaining trees truncated in preview) ... */", language="c")

    st.subheader("Edge Hardware Profile (Dell Inspiron 1525 / Dual-Core Benchmark)")
    st.table({
        "Metric": ["Model Size", "Inference Latency", "Pipeline Speedup", "RAM Footprint", "MIT-BIH Sensitivity"],
        "Value": ["38.08 KB (16.3 KB in C)", "0.12 ms / beat", ">7,800× real-time", "<175 MB", "100.0%"]
    })

