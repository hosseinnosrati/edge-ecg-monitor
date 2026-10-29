/*
 * Edge-ECG-Monitor Embedded Classifier
 * Standalone ANSI C99 implementation (Zero Dynamic Allocation)
 * Compatible with ARM Cortex-M, STM32, ESP32, and bare-metal firmware.
 */

#ifndef EDGE_CLASSIFIER_H
#define EDGE_CLASSIFIER_H

#include <stdint.h>

#define ECG_NUM_FEATURES 8
#define ECG_NUM_TREES    50

/* Feature Indices: */
/* 0: pre_rr_ms, 1: post_rr_ms, 2: rr_ratio, 3: local_relative_rr */
/* 4: qrs_amplitude, 5: qrs_energy, 6: qrs_variance, 7: qrs_width_ms */

static const float SCALER_MEANS[ECG_NUM_FEATURES] = {
    837.190471f, 835.019540f, 1.087213f, 1.012858f, 1.995539f, 34.721463f, 0.326779f, 37.149138f
};

static const float SCALER_SCALES[ECG_NUM_FEATURES] = {
    190.680284f, 192.624348f, 0.470275f, 0.216619f, 0.659662f, 52.107829f, 0.416899f, 22.112989f
};

static inline int tree_0_predict(const float *x) {
    if (x[5] <= 1.079847f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_1_predict(const float *x) {
    if (x[7] <= 1.334850f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_2_predict(const float *x) {
    if (x[4] <= 1.211104f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_3_predict(const float *x) {
    if (x[5] <= 0.444248f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_4_predict(const float *x) {
    if (x[7] <= 1.397659f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_5_predict(const float *x) {
    if (x[7] <= 1.460468f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_6_predict(const float *x) {
    if (x[0] <= -1.185646f) {
        if (x[5] <= 0.425028f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[7] <= 1.586085f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_7_predict(const float *x) {
    if (x[0] <= -1.258485f) {
        if (x[3] <= -1.515802f) {
            if (x[3] <= -1.872692f) {
                if (x[4] <= 1.220349f) {
                    return 0;
                } else {
                    return 1;
                }
            } else {
                return 1;
            }
        } else {
            if (x[4] <= 1.301233f) {
                return 0;
            } else {
                return 1;
            }
        }
    } else {
        if (x[7] <= 1.523276f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_8_predict(const float *x) {
    if (x[4] <= 1.254630f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_9_predict(const float *x) {
    if (x[5] <= 0.444248f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_10_predict(const float *x) {
    if (x[1] <= 0.993485f) {
        if (x[7] <= 1.397659f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        return 1;
    }
}

static inline int tree_11_predict(const float *x) {
    if (x[5] <= 1.079847f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_12_predict(const float *x) {
    if (x[1] <= 0.993485f) {
        if (x[5] <= 1.173034f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        return 1;
    }
}

static inline int tree_13_predict(const float *x) {
    if (x[6] <= 0.632155f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_14_predict(const float *x) {
    if (x[3] <= -1.515802f) {
        if (x[5] <= 0.417676f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[6] <= 1.136293f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_15_predict(const float *x) {
    if (x[5] <= 0.460528f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_16_predict(const float *x) {
    if (x[5] <= 1.084164f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_17_predict(const float *x) {
    if (x[6] <= 0.632155f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_18_predict(const float *x) {
    if (x[0] <= -1.222066f) {
        if (x[7] <= 1.209233f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[6] <= 1.136293f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_19_predict(const float *x) {
    if (x[4] <= 1.267218f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_20_predict(const float *x) {
    if (x[5] <= 1.084164f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_21_predict(const float *x) {
    if (x[3] <= -1.404274f) {
        if (x[7] <= 1.209233f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[5] <= 1.694442f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_22_predict(const float *x) {
    if (x[7] <= 1.397659f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_23_predict(const float *x) {
    if (x[3] <= -1.441450f) {
        if (x[7] <= 1.272041f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[7] <= 1.648894f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_24_predict(const float *x) {
    if (x[3] <= -1.404274f) {
        if (x[6] <= 0.563148f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[2] <= -1.065114f) {
            return 1;
        } else {
            if (x[5] <= 1.470050f) {
                return 0;
            } else {
                return 1;
            }
        }
    }
}

static inline int tree_25_predict(const float *x) {
    if (x[5] <= 1.084591f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_26_predict(const float *x) {
    if (x[7] <= 1.397659f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_27_predict(const float *x) {
    if (x[6] <= 0.632155f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_28_predict(const float *x) {
    if (x[5] <= 0.448565f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_29_predict(const float *x) {
    if (x[4] <= 1.277823f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_30_predict(const float *x) {
    if (x[4] <= 1.170780f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_31_predict(const float *x) {
    if (x[1] <= 0.943012f) {
        if (x[4] <= 1.460474f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        return 1;
    }
}

static inline int tree_32_predict(const float *x) {
    if (x[7] <= 1.397659f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_33_predict(const float *x) {
    if (x[0] <= -1.222066f) {
        if (x[5] <= 0.425028f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[6] <= 1.210709f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_34_predict(const float *x) {
    if (x[7] <= 1.397659f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_35_predict(const float *x) {
    if (x[5] <= 0.460528f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_36_predict(const float *x) {
    if (x[4] <= 1.198516f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_37_predict(const float *x) {
    if (x[3] <= -1.515802f) {
        if (x[5] <= 0.425028f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[6] <= 1.127709f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_38_predict(const float *x) {
    if (x[0] <= -1.222066f) {
        if (x[4] <= 1.162103f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[4] <= 1.403721f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_39_predict(const float *x) {
    if (x[4] <= 1.211104f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_40_predict(const float *x) {
    if (x[7] <= 1.334850f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_41_predict(const float *x) {
    if (x[0] <= -1.222066f) {
        if (x[7] <= 1.209233f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[7] <= 1.586085f) {
            return 0;
        } else {
            return 1;
        }
    }
}

static inline int tree_42_predict(const float *x) {
    if (x[5] <= 1.096126f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_43_predict(const float *x) {
    if (x[4] <= 1.227620f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_44_predict(const float *x) {
    if (x[5] <= 1.096126f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_45_predict(const float *x) {
    if (x[7] <= 1.397659f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_46_predict(const float *x) {
    if (x[7] <= 1.334850f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_47_predict(const float *x) {
    if (x[6] <= 1.060541f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_48_predict(const float *x) {
    if (x[5] <= 0.444248f) {
        return 0;
    } else {
        return 1;
    }
}

static inline int tree_49_predict(const float *x) {
    if (x[0] <= -1.309472f) {
        if (x[5] <= 0.417676f) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (x[6] <= 1.210709f) {
            return 0;
        } else {
            return 1;
        }
    }
}

/**
 * Predict cardiac rhythm class from raw extracted features.
 * @param raw_features Array of 8 extracted ECG features.
 * @return 0 for Normal beat, 1 for Ventricular Arrhythmia (PVC).
 */
static inline int edge_predict_arrhythmia(const float raw_features[ECG_NUM_FEATURES]) {
    float scaled_x[ECG_NUM_FEATURES];
    for (int i = 0; i < ECG_NUM_FEATURES; ++i) {
        scaled_x[i] = (raw_features[i] - SCALER_MEANS[i]) / SCALER_SCALES[i];
    }

    int arrhythmia_votes = 0;
    arrhythmia_votes += tree_0_predict(scaled_x);
    arrhythmia_votes += tree_1_predict(scaled_x);
    arrhythmia_votes += tree_2_predict(scaled_x);
    arrhythmia_votes += tree_3_predict(scaled_x);
    arrhythmia_votes += tree_4_predict(scaled_x);
    arrhythmia_votes += tree_5_predict(scaled_x);
    arrhythmia_votes += tree_6_predict(scaled_x);
    arrhythmia_votes += tree_7_predict(scaled_x);
    arrhythmia_votes += tree_8_predict(scaled_x);
    arrhythmia_votes += tree_9_predict(scaled_x);
    arrhythmia_votes += tree_10_predict(scaled_x);
    arrhythmia_votes += tree_11_predict(scaled_x);
    arrhythmia_votes += tree_12_predict(scaled_x);
    arrhythmia_votes += tree_13_predict(scaled_x);
    arrhythmia_votes += tree_14_predict(scaled_x);
    arrhythmia_votes += tree_15_predict(scaled_x);
    arrhythmia_votes += tree_16_predict(scaled_x);
    arrhythmia_votes += tree_17_predict(scaled_x);
    arrhythmia_votes += tree_18_predict(scaled_x);
    arrhythmia_votes += tree_19_predict(scaled_x);
    arrhythmia_votes += tree_20_predict(scaled_x);
    arrhythmia_votes += tree_21_predict(scaled_x);
    arrhythmia_votes += tree_22_predict(scaled_x);
    arrhythmia_votes += tree_23_predict(scaled_x);
    arrhythmia_votes += tree_24_predict(scaled_x);
    arrhythmia_votes += tree_25_predict(scaled_x);
    arrhythmia_votes += tree_26_predict(scaled_x);
    arrhythmia_votes += tree_27_predict(scaled_x);
    arrhythmia_votes += tree_28_predict(scaled_x);
    arrhythmia_votes += tree_29_predict(scaled_x);
    arrhythmia_votes += tree_30_predict(scaled_x);
    arrhythmia_votes += tree_31_predict(scaled_x);
    arrhythmia_votes += tree_32_predict(scaled_x);
    arrhythmia_votes += tree_33_predict(scaled_x);
    arrhythmia_votes += tree_34_predict(scaled_x);
    arrhythmia_votes += tree_35_predict(scaled_x);
    arrhythmia_votes += tree_36_predict(scaled_x);
    arrhythmia_votes += tree_37_predict(scaled_x);
    arrhythmia_votes += tree_38_predict(scaled_x);
    arrhythmia_votes += tree_39_predict(scaled_x);
    arrhythmia_votes += tree_40_predict(scaled_x);
    arrhythmia_votes += tree_41_predict(scaled_x);
    arrhythmia_votes += tree_42_predict(scaled_x);
    arrhythmia_votes += tree_43_predict(scaled_x);
    arrhythmia_votes += tree_44_predict(scaled_x);
    arrhythmia_votes += tree_45_predict(scaled_x);
    arrhythmia_votes += tree_46_predict(scaled_x);
    arrhythmia_votes += tree_47_predict(scaled_x);
    arrhythmia_votes += tree_48_predict(scaled_x);
    arrhythmia_votes += tree_49_predict(scaled_x);
    return (arrhythmia_votes > (ECG_NUM_TREES / 2)) ? 1 : 0;
}

/**
 * Get estimated posterior probability of Ventricular Arrhythmia [0.0f, 1.0f].
 */
static inline float edge_predict_arrhythmia_probability(const float raw_features[ECG_NUM_FEATURES]) {
    float scaled_x[ECG_NUM_FEATURES];
    for (int i = 0; i < ECG_NUM_FEATURES; ++i) {
        scaled_x[i] = (raw_features[i] - SCALER_MEANS[i]) / SCALER_SCALES[i];
    }

    int votes = 0;
    votes += tree_0_predict(scaled_x);
    votes += tree_1_predict(scaled_x);
    votes += tree_2_predict(scaled_x);
    votes += tree_3_predict(scaled_x);
    votes += tree_4_predict(scaled_x);
    votes += tree_5_predict(scaled_x);
    votes += tree_6_predict(scaled_x);
    votes += tree_7_predict(scaled_x);
    votes += tree_8_predict(scaled_x);
    votes += tree_9_predict(scaled_x);
    votes += tree_10_predict(scaled_x);
    votes += tree_11_predict(scaled_x);
    votes += tree_12_predict(scaled_x);
    votes += tree_13_predict(scaled_x);
    votes += tree_14_predict(scaled_x);
    votes += tree_15_predict(scaled_x);
    votes += tree_16_predict(scaled_x);
    votes += tree_17_predict(scaled_x);
    votes += tree_18_predict(scaled_x);
    votes += tree_19_predict(scaled_x);
    votes += tree_20_predict(scaled_x);
    votes += tree_21_predict(scaled_x);
    votes += tree_22_predict(scaled_x);
    votes += tree_23_predict(scaled_x);
    votes += tree_24_predict(scaled_x);
    votes += tree_25_predict(scaled_x);
    votes += tree_26_predict(scaled_x);
    votes += tree_27_predict(scaled_x);
    votes += tree_28_predict(scaled_x);
    votes += tree_29_predict(scaled_x);
    votes += tree_30_predict(scaled_x);
    votes += tree_31_predict(scaled_x);
    votes += tree_32_predict(scaled_x);
    votes += tree_33_predict(scaled_x);
    votes += tree_34_predict(scaled_x);
    votes += tree_35_predict(scaled_x);
    votes += tree_36_predict(scaled_x);
    votes += tree_37_predict(scaled_x);
    votes += tree_38_predict(scaled_x);
    votes += tree_39_predict(scaled_x);
    votes += tree_40_predict(scaled_x);
    votes += tree_41_predict(scaled_x);
    votes += tree_42_predict(scaled_x);
    votes += tree_43_predict(scaled_x);
    votes += tree_44_predict(scaled_x);
    votes += tree_45_predict(scaled_x);
    votes += tree_46_predict(scaled_x);
    votes += tree_47_predict(scaled_x);
    votes += tree_48_predict(scaled_x);
    votes += tree_49_predict(scaled_x);
    return (float)votes / (float)ECG_NUM_TREES;
}

#endif /* EDGE_CLASSIFIER_H */