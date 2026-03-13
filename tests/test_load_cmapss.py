"""Tests for src.load_cmapss (NASA C-MAPSS RUL)."""

import pytest
import numpy as np
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_cmapss import CMAPSS_DIR, load_fd001, compute_train_rul, build_sequences  # noqa: E402


@pytest.mark.parametrize("fd", [1, 2])
def test_load_fd(fd):
    """Load FD001/FD002 returns train, test, RUL."""
    if not (CMAPSS_DIR / f"train_FD00{fd}.txt").exists():
        pytest.skip(f"C-MAPSS FD00{fd} not found. Run: python scripts/download_cmapss.py")

    train_df, test_df, true_rul = load_fd001(CMAPSS_DIR, fd=fd)
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert len(true_rul) == test_df["unit"].nunique()
    assert "unit" in train_df.columns
    assert "cycle" in train_df.columns
    assert train_df.shape[1] == 26


def test_compute_train_rul():
    """RUL is non-negative and capped."""
    import pandas as pd

    # Mini mock: 2 units, 10 cycles each
    df = pd.DataFrame({
        "unit": [1] * 10 + [2] * 10,
        "cycle": list(range(1, 11)) + list(range(1, 11)),
    })
    for c in ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]:
        df[c] = 0.0

    _, y = compute_train_rul(df, max_rul=5)
    assert len(y) == 20
    assert (y >= 0).all()
    assert (y <= 5).all()


def test_build_sequences():
    """Sequences have correct shape."""
    import pandas as pd

    n_cycles = 50
    df = pd.DataFrame({
        "unit": [1] * n_cycles,
        "cycle": range(1, n_cycles + 1),
    })
    for c in ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]:
        df[c] = np.random.randn(n_cycles)

    y = np.arange(n_cycles, 0, -1, dtype=np.float32)  # decreasing RUL
    X, y_out, units = build_sequences(df, y, window_size=10, stride=5)
    assert X.shape[1] == 10
    assert X.shape[2] == 21
    assert len(y_out) == len(units)
    assert len(X) >= 1
