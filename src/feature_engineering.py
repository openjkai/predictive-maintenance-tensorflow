"""
Feature engineering for CWRU vibration data — Phase 3
Extract time-domain features (RMS, peak, mean, std, kurtosis) per window.
"""

from pathlib import Path

import numpy as np

from src.load_data import DATA_DIR, load_dataset

# Windowing (from Phase 2 exploration)
WINDOW_SAMPLES = 1024
STEP_SAMPLES = 512

# Label mapping: filename prefix -> class name
LABEL_MAP = {
    "Normal": "normal",
    "IR": "inner_race",
    "B": "ball",
    "OR": "outer_race",
}

CLASS_NAMES = ["normal", "inner_race", "ball", "outer_race"]


def get_label(name: str) -> str:
    """Derive label from filename (e.g. IR007_0 -> inner_race)."""
    for prefix, label in LABEL_MAP.items():
        if name.startswith(prefix):
            return label
    return "unknown"


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract time-domain features from one window.
    Phase 3.1: RMS, peak, mean. Phase 3.2: std, kurtosis.

    Returns:
        1D array: [rms, peak, mean, std, kurtosis]
    """
    w = np.asarray(window, dtype=np.float64).ravel()
    rms = np.sqrt(np.mean(w**2))
    peak = np.max(np.abs(w))
    mean = np.mean(w)
    std = np.std(w)
    # Kurtosis: 4th moment / std^4 - 3 (excess kurtosis, 0 for normal)
    if std > 1e-10:
        kurtosis = np.mean(((w - mean) / std) ** 4) - 3.0
    else:
        kurtosis = 0.0
    return np.array([rms, peak, mean, std, kurtosis], dtype=np.float64)


def sliding_windows(
    signal: np.ndarray,
    window_size: int = WINDOW_SAMPLES,
    step: int = STEP_SAMPLES,
) -> list[np.ndarray]:
    """Split signal into overlapping windows."""
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start : start + window_size])
    return windows


def build_dataset(
    data_dir: Path | str = DATA_DIR,
    window_size: int = WINDOW_SAMPLES,
    step: int = STEP_SAMPLES,
    binary: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build ML-ready dataset from all .mat files.

    Args:
        data_dir: Path to data folder
        window_size: Samples per window
        step: Step between windows
        binary: If True, normal=0 / fault=1. If False, 4 classes.

    Returns:
        X: (n_samples, n_features) feature matrix
        y: (n_samples,) integer labels
        feature_names: list of feature column names
    """
    data = load_dataset(data_dir)
    feature_names = ["rms", "peak", "mean", "std", "kurtosis"]

    X_list = []
    y_list = []

    for name, (signal, _rate, _rpm) in sorted(data.items()):
        label = get_label(name)
        if label == "unknown":
            continue

        if binary:
            y_val = 0 if label == "normal" else 1
        else:
            y_val = CLASS_NAMES.index(label)

        for win in sliding_windows(signal, window_size, step):
            feats = extract_features(win)
            X_list.append(feats)
            y_list.append(y_val)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    return X, y, feature_names


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/val split (preserves class proportions).

    Returns:
        X_train, X_val, y_train, y_val
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(n * val_frac)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    return (
        X[train_idx],
        X[val_idx],
        y[train_idx],
        y[val_idx],
    )


if __name__ == "__main__":
    print("Phase 3 — Feature Engineering\n")

    X, y, names = build_dataset(binary=False)
    print(f"Features: {names}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {np.unique(y)} -> {CLASS_NAMES}")

    X_train, X_val, y_train, y_val = train_val_split(X, y, val_frac=0.2)
    print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}")
    print(f"X_train sample (first row): {X_train[0]}")
