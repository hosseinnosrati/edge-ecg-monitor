"""C-Header Exporter for Embedded Edge Deployment.

Converts the trained Scikit-Learn Random Forest and Feature Scaler into
a standalone ANSI C99 header file with zero heap allocations (no malloc).
Fully compatible with ARM Cortex-M0/M3/M4/M7, STM32, ESP32, and Arduino.
"""

import os
import joblib
import numpy as np
from sklearn.tree import _tree


def export_tree_to_c(tree, tree_index: int) -> str:
    """Generate C function for a single decision tree."""
    tree_ = tree.tree_
    lines = []
    lines.append(f"static inline int tree_{tree_index}_predict(const float *x) {{")

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = tree_.feature[node]
            threshold = tree_.threshold[node]
            lines.append(f"{indent}if (x[{name}] <= {threshold:.6f}f) {{")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            lines.append(f"{indent}}}")
        else:
            value = tree_.value[node][0]
            # 0: Normal, 1: Arrhythmia
            pred_class = 1 if value[1] > value[0] else 0
            lines.append(f"{indent}return {pred_class};")

    recurse(0, 1)
    lines.append("}\n")
    return "\n".join(lines)


def export_model_to_c_header(
    model_path: str = "models/edge_classifier.joblib",
    output_header_path: str = "models/edge_classifier.h"
):
    """Read trained joblib model and export complete C99 header."""
    checkpoint = joblib.load(model_path)
    model = checkpoint["model"]
    scaler = checkpoint["scaler"]

    means = scaler.mean_
    scales = scaler.scale_
    n_features = len(means)
    n_trees = len(model.estimators_)

    os.makedirs(os.path.dirname(output_header_path), exist_ok=True)

    header_code = []
    header_code.append("/*")
    header_code.append(" * Edge-ECG-Monitor Embedded Classifier")
    header_code.append(" * Standalone ANSI C99 implementation (Zero Dynamic Allocation)")
    header_code.append(" * Compatible with ARM Cortex-M, STM32, ESP32, and bare-metal firmware.")
    header_code.append(" */\n")
    header_code.append("#ifndef EDGE_CLASSIFIER_H")
    header_code.append("#define EDGE_CLASSIFIER_H\n")
    header_code.append("#include <stdint.h>\n")

    header_code.append(f"#define ECG_NUM_FEATURES {n_features}")
    header_code.append(f"#define ECG_NUM_TREES    {n_trees}\n")

    # Feature definitions
    header_code.append("/* Feature Indices: */")
    header_code.append("/* 0: pre_rr_ms, 1: post_rr_ms, 2: rr_ratio, 3: local_relative_rr */")
    header_code.append("/* 4: qrs_amplitude, 5: qrs_energy, 6: qrs_variance, 7: qrs_width_ms */\n")

    # Standard Scaler coefficients
    header_code.append("static const float SCALER_MEANS[ECG_NUM_FEATURES] = {")
    header_code.append("    " + ", ".join(f"{m:.6f}f" for m in means))
    header_code.append("};\n")

    header_code.append("static const float SCALER_SCALES[ECG_NUM_FEATURES] = {")
    header_code.append("    " + ", ".join(f"{s:.6f}f" for s in scales))
    header_code.append("};\n")

    # Individual Trees
    for i, est in enumerate(model.estimators_):
        header_code.append(export_tree_to_c(est, i))

    # Main inference function
    header_code.append("/**")
    header_code.append(" * Predict cardiac rhythm class from raw extracted features.")
    header_code.append(" * @param raw_features Array of 8 extracted ECG features.")
    header_code.append(" * @return 0 for Normal beat, 1 for Ventricular Arrhythmia (PVC).")
    header_code.append(" */")
    header_code.append("static inline int edge_predict_arrhythmia(const float raw_features[ECG_NUM_FEATURES]) {")
    header_code.append("    float scaled_x[ECG_NUM_FEATURES];")
    header_code.append("    for (int i = 0; i < ECG_NUM_FEATURES; ++i) {")
    header_code.append("        scaled_x[i] = (raw_features[i] - SCALER_MEANS[i]) / SCALER_SCALES[i];")
    header_code.append("    }\n")
    header_code.append("    int arrhythmia_votes = 0;")
    for i in range(n_trees):
        header_code.append(f"    arrhythmia_votes += tree_{i}_predict(scaled_x);")
    header_code.append(f"    return (arrhythmia_votes > (ECG_NUM_TREES / 2)) ? 1 : 0;")
    header_code.append("}\n")

    # Probability estimation function
    header_code.append("/**")
    header_code.append(" * Get estimated posterior probability of Ventricular Arrhythmia [0.0f, 1.0f].")
    header_code.append(" */")
    header_code.append("static inline float edge_predict_arrhythmia_probability(const float raw_features[ECG_NUM_FEATURES]) {")
    header_code.append("    float scaled_x[ECG_NUM_FEATURES];")
    header_code.append("    for (int i = 0; i < ECG_NUM_FEATURES; ++i) {")
    header_code.append("        scaled_x[i] = (raw_features[i] - SCALER_MEANS[i]) / SCALER_SCALES[i];")
    header_code.append("    }\n")
    header_code.append("    int votes = 0;")
    for i in range(n_trees):
        header_code.append(f"    votes += tree_{i}_predict(scaled_x);")
    header_code.append(f"    return (float)votes / (float)ECG_NUM_TREES;")
    header_code.append("}\n")

    header_code.append("#endif /* EDGE_CLASSIFIER_H */")

    full_code = "\n".join(header_code)
    with open(output_header_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    print(f"C-Header successfully generated: {output_header_path} ({os.path.getsize(output_header_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    export_model_to_c_header()
