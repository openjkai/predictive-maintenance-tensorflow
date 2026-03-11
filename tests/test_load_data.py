"""Tests for src.load_data (CWRU bearing)."""

import pytest
from pathlib import Path

# Assume project root when running pytest from repo
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_data import DATA_DIR, load_mat_file  # noqa: E402


def test_get_label():
    """Label mapping from filename."""
    from src.feature_engineering import get_label

    assert get_label("Normal_0") == "normal"
    assert get_label("IR007_0") == "inner_race"
    assert get_label("B007_0") == "ball"
    assert get_label("OR007_0") == "outer_race"


def test_load_mat_file():
    """Load one .mat file returns (signal, rate, rpm)."""
    normal_path = DATA_DIR / "Normal_0.mat"
    if not normal_path.exists():
        pytest.skip("CWRU data not found. Run: python scripts/download_cwru.py")

    signal, rate, rpm = load_mat_file(normal_path)
    assert signal.ndim == 1
    assert len(signal) > 1000
    assert rate in (12_000, 48_000)
    assert rpm >= 0
