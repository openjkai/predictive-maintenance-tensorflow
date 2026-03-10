"""
Load NASA C-MAPSS dataset for RUL prediction — Phase 7.2
Turbofan engine degradation: train/test + RUL labels.
Data: unit_id, cycle, 3 op settings, 21 sensors (26 columns total).
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Default: data/cmapss/
CMAPSS_DIR = Path(__file__).resolve().parents[1] / "data" / "cmapss"

# Sensor columns (0-based): exclude unit(0), cycle(1), op(2,3,4)
# Use all 21 sensors (cols 5-25) — optionally filter constants later
SENSOR_COLS = list(range(5, 26))

# RUL cap (piecewise linear — standard in literature)
MAX_RUL = 125


class CmapssSplit(NamedTuple):
    """Training or test split with features and RUL labels."""
    X: np.ndarray  # (n_sequences, seq_len, n_features) or (n_samples, n_features)
    y: np.ndarray  # (n_sequences,) or (n_samples,) RUL values
    unit_ids: np.ndarray  # engine unit id per sample
    is_sequence: bool  # True if windowed for LSTM


def load_fd001(
    data_dir: Path | str = CMAPSS_DIR,
    fd: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load FD001 train, test, and true RUL for test.

    Returns:
        train_df: columns [unit, cycle, op1, op2, op3, s1..s21]
        test_df: same structure (ends before failure)
        true_rul: RUL for each test engine (one value per engine)
    """
    data_dir = Path(data_dir)
    train_path = data_dir / f"train_FD00{fd}.txt"
    test_path = data_dir / f"test_FD00{fd}.txt"
    rul_path = data_dir / f"RUL_FD00{fd}.txt"

    if not train_path.exists():
        raise FileNotFoundError(
            f"C-MAPSS FD00{fd} not found. Run: python scripts/download_cmapss.py"
        )

    col_names = (
        ["unit", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"s{i}" for i in range(1, 22)]
    )

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)

    true_rul = np.loadtxt(rul_path, dtype=np.int32) if rul_path.exists() else np.array([])

    return train_df, test_df, true_rul


def compute_train_rul(
    df: pd.DataFrame,
    max_rul: int = MAX_RUL,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute RUL for training data (run-to-failure).
    RUL = cycles until failure; capped at max_rul (piecewise linear).
    """
    rul_list = []
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle")
        max_cycle = unit_df["cycle"].max()
        rul = (max_cycle - unit_df["cycle"]).values
        rul = np.minimum(rul, max_rul)
        rul_list.append(rul)
    y = np.concatenate(rul_list)
    return df, y


def build_sequences(
    df: pd.DataFrame,
    y: np.ndarray,
    sensor_cols: list[int] = SENSOR_COLS,
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding window sequences per engine.
    Each window: (window_size, n_sensors). Label = RUL at last timestep.

    Returns:
        X: (n_sequences, window_size, n_sensors)
        y: (n_sequences,) RUL at end of each window
        unit_ids: (n_sequences,) engine unit
    """
    X_list, y_list, unit_list = [], [], []
    start = 0
    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
        unit_y = y[start : start + len(unit_df)]
        start += len(unit_df)

        arr = unit_df.iloc[:, sensor_cols].values.astype(np.float32)
        if len(arr) < window_size:
            continue

        for i in range(0, len(arr) - window_size + 1, stride):
            X_list.append(arr[i : i + window_size])
            y_list.append(unit_y[i + window_size - 1])
            unit_list.append(unit)

    return np.array(X_list), np.array(y_list, dtype=np.float32), np.array(unit_list)


def prepare_fd001(
    data_dir: Path | str = CMAPSS_DIR,
    window_size: int = 30,
    max_rul: int = MAX_RUL,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full pipeline: load FD001, build sequences, split train/val, scale.

    Returns:
        dict with X_train, X_val, y_train, y_val, scaler, test_X, test_y_true, etc.
    """
    train_df, test_df, true_rul = load_fd001(data_dir, fd=1)
    train_df, train_y = compute_train_rul(train_df, max_rul=max_rul)

    # Drop constant sensors (zero variance)
    sensor_arr = train_df.iloc[:, SENSOR_COLS].values
    variances = np.var(sensor_arr, axis=0)
    used_cols = [SENSOR_COLS[i] for i in range(len(SENSOR_COLS)) if variances[i] > 1e-10]
    if len(used_cols) < 5:
        used_cols = SENSOR_COLS

    X_train, y_train, unit_train = build_sequences(
        train_df, train_y, sensor_cols=used_cols, window_size=window_size
    )

    # Train/val split by engine (no leakage)
    units = np.unique(unit_train)
    rng = np.random.default_rng(random_state)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[:n_val])
    train_mask = ~np.isin(unit_train, list(val_units))
    val_mask = np.isin(unit_train, list(val_units))

    X_tr, X_val = X_train[train_mask], X_train[val_mask]
    y_tr, y_val = y_train[train_mask], y_train[val_mask]

    # Min-max scale (-1 to 1) on train only
    from sklearn.preprocessing import MinMaxScaler

    n_tr, seq_len, n_feat = X_tr.shape
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_tr_flat = X_tr.reshape(-1, n_feat)
    scaler.fit(X_tr_flat)
    X_tr = scaler.transform(X_tr_flat).reshape(n_tr, seq_len, n_feat).astype(np.float32)
    X_val_flat = X_val.reshape(-1, n_feat)
    X_val = scaler.transform(X_val_flat).reshape(X_val.shape[0], seq_len, n_feat).astype(np.float32)

    # Test sequences: for each engine, use last window_size cycles
    test_X_list, test_units = [], []
    start = 0
    for idx, unit in enumerate(test_df["unit"].unique()):
        unit_df = test_df[test_df["unit"] == unit].sort_values("cycle")
        arr = unit_df.iloc[:, used_cols].values.astype(np.float32)
        if len(arr) >= window_size:
            window = arr[-window_size:]
            window_scaled = scaler.transform(window).astype(np.float32)
            test_X_list.append(window_scaled)
            test_units.append(unit)
    test_X = np.array(test_X_list) if test_X_list else np.zeros((0, window_size, len(used_cols)), dtype=np.float32)
    test_y_true = true_rul[: len(test_X_list)] if len(test_X_list) <= len(true_rul) else true_rul

    return {
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": test_X,
        "y_test_true": test_y_true,
        "scaler": scaler,
        "window_size": window_size,
        "n_features": len(used_cols),
        "used_cols": used_cols,
    }
