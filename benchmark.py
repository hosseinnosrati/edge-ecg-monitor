"""Edge Hardware Performance Benchmarking Suite.

Profiles latency, memory footprint, and computational throughput
on resource-constrained devices.
"""

import time
import os
import psutil
import numpy as np

from src.dsp.filters import clean_ecg
from src.dsp.pan_tompkins import PanTompkinsDetector
from src.features.hrv import extract_beat_features
from src.models.classifier import EdgeArrhythmiaClassifier, FEATURE_NAMES
from src.data_loader import load_record


def run_benchmark():
    process = psutil.Process(os.getpid())
    initial_mem_mb = process.memory_info().rss / (1024 * 1024)

    print("=" * 60)
    print("      EDGE HARDWARE BENCHMARK (REAL-TIME DSP + AI)")
    print("=" * 60)
    print(f"Initial Memory Footprint: {initial_mem_mb:.2f} MB")

    # Load 60 seconds of real ECG from MIT-BIH 119
    raw_signal, _, _, fs = load_record("119", data_dir="data/mitbih")
    test_duration_sec = 60
    samples_count = int(test_duration_sec * fs)
    test_signal = raw_signal[:samples_count]

    # Load Model
    classifier = EdgeArrhythmiaClassifier.load("models/edge_classifier.joblib")
    detector = PanTompkinsDetector(fs=fs)

    # 1. Benchmark DSP Filtering Stage
    t0 = time.perf_counter()
    cleaned = clean_ecg(test_signal, fs=fs, powerline_freq=60.0)
    dsp_filter_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Benchmark Pan-Tompkins Peak Detection
    t0 = time.perf_counter()
    r_peaks, _ = detector.detect_peaks(cleaned)
    peak_det_time_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Benchmark Feature Extraction
    t0 = time.perf_counter()
    beat_features = extract_beat_features(cleaned, r_peaks, fs=fs)
    feat_time_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Benchmark ML Classification
    X = np.array([[bf[f] for f in FEATURE_NAMES] for bf in beat_features], dtype=np.float32)
    t0 = time.perf_counter()
    predictions = classifier.predict(X)
    ml_time_ms = (time.perf_counter() - t0) * 1000.0

    total_pipeline_time_ms = dsp_filter_time_ms + peak_det_time_ms + feat_time_ms + ml_time_ms
    num_beats = len(beat_features)
    latency_per_beat_ms = total_pipeline_time_ms / max(1, num_beats)

    final_mem_mb = process.memory_info().rss / (1024 * 1024)
    model_file_size_kb = os.path.getsize("models/edge_classifier.joblib") / 1024.0

    print(f"Processed Signal Duration: {test_duration_sec} seconds ({samples_count} samples)")
    print(f"Detected Beats:            {num_beats} beats")
    print(f"Detected Arrhythmias:      {np.sum(predictions == 1)} beats")
    print("-" * 60)
    print(f"DSP Filter Stage:          {dsp_filter_time_ms:.2f} ms")
    print(f"Pan-Tompkins Detection:    {peak_det_time_ms:.2f} ms")
    print(f"Feature Extraction:        {feat_time_ms:.2f} ms")
    print(f"ML Classifier Inference:   {ml_time_ms:.2f} ms")
    print("-" * 60)
    print(f"Total Pipeline Latency:    {total_pipeline_time_ms:.2f} ms (for {test_duration_sec}s of audio/signal)")
    print(f"Real-Time Speedup Factor:  {(test_duration_sec * 1000.0) / total_pipeline_time_ms:.1f}x faster than real-time!")
    print(f"Average Latency per Beat:  {latency_per_beat_ms:.3f} ms / beat")
    print(f"Peak Memory Usage:         {final_mem_mb:.2f} MB")
    print(f"Compiled Model Size:       {model_file_size_kb:.2f} KB")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()

