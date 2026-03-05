"""
Load CWRU Bearing .mat files — Phase 1, Step 1.4
Each file contains: DE (drive end), FE (fan end), BA (base) vibration + RPM.
Variable names follow X###_DE_time (file ID in the number).
"""

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

# Sampling rates (Hz) — CWRU uses 12k for fault data, 48k for normal baseline
SAMPLE_RATE_12K = 12_000
SAMPLE_RATE_48K = 48_000

# Default data directory relative to project root
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _find_de_key(mat: dict) -> Optional[str]:
    """Find the Drive End vibration key (X###_DE_time)."""
    for key in mat.keys():
        if "_DE_time" in key:
            return key
    return None


def load_mat_file(
    filepath: Path | str,
    channel: str = "DE",
) -> tuple[np.ndarray, int, float]:
    """
    Load one CWRU .mat file and return vibration signal + metadata.

    Args:
        filepath: Path to .mat file (e.g. data/Normal_0.mat)
        channel: 'DE' (drive end), 'FE' (fan end), or 'BA' (base)

    Returns:
        signal: 1D numpy array of vibration (amplitude)
        sample_rate: 12_000 or 48_000 Hz
        rpm: Motor speed during test

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If channel not found in file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    mat = loadmat(str(filepath), squeeze_me=True)

    # Find vibration channel (e.g. X097_DE_time, X105_FE_time)
    pattern = f"_{channel}_time"
    key = None
    for k in mat.keys():
        if pattern in k:
            key = k
            break

    if key is None:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"Channel '{channel}' not found. Available: {available}")

    signal = np.asarray(mat[key], dtype=np.float64).ravel()

    # Infer sample rate from length (12k ~10s = 120k samples, 48k ~5s = 240k)
    n = len(signal)
    sample_rate = SAMPLE_RATE_48K if n > 200_000 else SAMPLE_RATE_12K

    # RPM if present (e.g. X097RPM)
    rpm_key = [k for k in mat.keys() if "RPM" in k and not k.startswith("__")]
    rpm = float(mat[rpm_key[0]]) if rpm_key else 0.0

    return signal, sample_rate, rpm


def load_dataset(
    data_dir: Path | str = DATA_DIR,
    channel: str = "DE",
) -> dict[str, tuple[np.ndarray, int, float]]:
    """
    Load all .mat files in data directory.

    Returns:
        Dict mapping filename stem (e.g. 'Normal_0') to (signal, sample_rate, rpm)
    """
    data_dir = Path(data_dir)
    result = {}

    for path in sorted(data_dir.glob("*.mat")):
        try:
            signal, rate, rpm = load_mat_file(path, channel=channel)
            result[path.stem] = (signal, rate, rpm)
        except Exception as e:
            print(f"Warning: skip {path.name}: {e}")

    return result


def inspect_file(filepath: Path | str) -> None:
    """
    Print structure of one .mat file — useful for exploration.
    """
    filepath = Path(filepath)
    mat = loadmat(str(filepath), squeeze_me=True)

    print(f"File: {filepath.name}")
    print(f"Keys (excluding meta): {[k for k in mat.keys() if not k.startswith('__')]}")

    for key in sorted(mat.keys()):
        if key.startswith("__"):
            continue
        v = mat[key]
        shape = getattr(v, "shape", "?")
        dtype = getattr(v, "dtype", "?")
        print(f"  {key}: shape={shape}, dtype={dtype}")


# --- For direct runs: load one file and print summary ---
if __name__ == "__main__":
    import sys

    # Default: load Normal_0 and show summary
    target = DATA_DIR / "Normal_0.mat"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])

    if not target.exists():
        print(f"Usage: python load_data.py [path/to/file.mat]")
        print(f"Default: {target}")
        sys.exit(1)

    print("Inspect structure:")
    inspect_file(target)
    print()

    signal, rate, rpm = load_mat_file(target)
    duration = len(signal) / rate
    print(f"Loaded: {target.name}")
    print(f"  Samples: {len(signal):,}")
    print(f"  Sample rate: {rate:,} Hz")
    print(f"  Duration: {duration:.2f} s")
    print(f"  RPM: {rpm}")
    print(f"  Signal range: [{signal.min():.4f}, {signal.max():.4f}]")
