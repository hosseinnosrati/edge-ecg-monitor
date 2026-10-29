"""Download and parse MIT-BIH Arrhythmia Database records from PhysioNet.

MIT-BIH is the benchmark international standard for cardiac arrhythmia research.
Reference: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
"""

import os
from typing import Tuple, List, Dict
import numpy as np
import wfdb


# AAMI Standard Classification Mapping:
# N: Normal beat (Normal, LBBB, RBBB, etc.)
# V: Ventricular ectopic beat (PVC, VEB)
# S: Supraventricular ectopic beat (APC, etc.)
# F: Fusion beat
# Q: Unknown / unclassifiable beat
AAMI_MAPPING = {
    "N": "Normal",
    "L": "Normal",
    "R": "Normal",
    "V": "Ventricular_Ectopic",
    "E": "Ventricular_Ectopic",
    "A": "Supraventricular",
    "a": "Supraventricular",
    "J": "Supraventricular",
    "S": "Supraventricular",
    "F": "Fusion",
    "/": "Paced",
}


def download_sample_mitbih_records(target_dir: str = "data/mitbih", records: List[str] = None):
    """Download representative MIT-BIH records from PhysioNet if not present locally."""
    if records is None:
        # 100: baseline normal; 119: rich in ventricular arrhythmias (PVC)
        records = ["100", "119"]

    os.makedirs(target_dir, exist_ok=True)
    for rec in records:
        hea_file = os.path.join(target_dir, f"{rec}.hea")
        dat_file = os.path.join(target_dir, f"{rec}.dat")
        if not (os.path.exists(hea_file) and os.path.exists(dat_file)):
            print(f"Downloading MIT-BIH record {rec} from PhysioNet...")
            wfdb.dl_files(
                db="mitdb",
                dl_dir=target_dir,
                files=[f"{rec}.hea", f"{rec}.dat", f"{rec}.atr"]
            )
            print(f"Record {rec} downloaded.")
        else:
            print(f"Record {rec} already exists in {target_dir}.")


def load_record(record_name: str = "100", data_dir: str = "data/mitbih") -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    """Load an ECG record, its lead-0 signal, annotations, and sampling rate.

    Returns
    -------
    signal : np.ndarray (1D lead-0 ECG)
    annotation_samples : np.ndarray (indices of cardiologist-annotated R-peaks)
    annotation_symbols : List[str] (beat type symbols: 'N', 'V', etc.)
    fs : float (sampling rate in Hz)
    """
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    lead0_signal = record.p_signal[:, 0]
    return lead0_signal, annotation.sample, annotation.symbol, float(record.fs)


if __name__ == "__main__":
    download_sample_mitbih_records()

