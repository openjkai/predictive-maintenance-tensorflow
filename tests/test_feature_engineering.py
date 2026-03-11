"""Tests for src.feature_engineering."""

import pytest
import numpy as np
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import (  # noqa: E402
    extract_features,
    sliding_windows,
    train_val_split,
    _spectral_centroid_bandwidth,
    _wavelet_features,
)


def test_extract_features_shape():
    """extract_features returns 9 values."""
    window = np.random.randn(1024).astype(np.float64)
    feats = extract_features(window)
    assert feats.shape == (9,)
    assert feats.dtype == np.float64


def test_extract_features_basic():
    """Time-domain features are non-negative where expected."""
    window = np.ones(1024) * 0.5
    feats = extract_features(window)
    # rms, peak should be >= 0
    assert feats[0] >= 0  # rms
    assert feats[1] >= 0  # peak
    assert feats[2] == 0.5  # mean


def test_sliding_windows():
    """Sliding windows produce correct shape and count."""
    signal = np.random.randn(2048)
    windows = sliding_windows(signal, window_size=1024, step=512)
    assert len(windows) >= 2
    assert all(len(w) == 1024 for w in windows)
    np.testing.assert_array_equal(windows[0], signal[:1024])


def test_train_val_split():
    """Train/val split preserves proportions roughly."""
    X = np.random.randn(100, 5)
    y = np.array([0] * 50 + [1] * 50)  # balanced
    Xtr, Xv, ytr, yv = train_val_split(X, y, val_frac=0.2, random_state=42)
    assert len(ytr) + len(yv) == 100
    assert len(yv) == 20
    assert set(np.unique(ytr)) <= {0, 1}
    assert set(np.unique(yv)) <= {0, 1}


def test_spectral_centroid_bandwidth():
    """Spectral features return floats in [0, 0.5] for normalized freq."""
    window = np.sin(np.linspace(0, 4 * np.pi, 1024))
    c, b = _spectral_centroid_bandwidth(window)
    assert 0 <= c <= 0.5
    assert b >= 0


def test_wavelet_features():
    """Wavelet features return two floats."""
    window = np.random.randn(1024)
    d1, a1 = _wavelet_features(window)
    assert isinstance(d1, float)
    assert isinstance(a1, float)
    assert d1 >= 0
    assert a1 >= 0
