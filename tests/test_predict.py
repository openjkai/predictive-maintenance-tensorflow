"""Tests for src.predict (bearing fault + RUL)."""

import pytest
import numpy as np
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_predict_single_needs_model():
    """predict_single fails gracefully when model missing."""
    from src.predict import predict_single, DEFAULT_MODEL_PATH

    if DEFAULT_MODEL_PATH.exists():
        # If model exists, run a quick prediction
        feats = np.random.randn(9).astype(np.float64) * 0.1
        result = predict_single(feats)
        assert "predicted_class" in result
        assert "probability" in result
        assert "class_names" in result
        assert result["predicted_class"] in result["class_names"]
    else:
        with pytest.raises(FileNotFoundError):
            predict_single([0.1] * 9)


def test_predict_from_file_needs_mat():
    """predict_from_file fails on missing file."""
    from src.predict import predict_from_file

    with pytest.raises(FileNotFoundError):
        predict_from_file("/nonexistent/file.mat")


def test_predict_from_file_with_data():
    """predict_from_file returns dict with expected keys."""
    from src.predict import predict_from_file
    from src.load_data import DATA_DIR

    mat_path = DATA_DIR / "IR007_0.mat"
    if not mat_path.exists():
        pytest.skip("CWRU data not found")

    # Need trained model
    from src.predict import DEFAULT_MODEL_PATH
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("Feature model not found. Run: python scripts/train.py")

    result = predict_from_file(mat_path, n_windows=3)
    assert "predicted_class" in result
    assert "probability" in result
    assert "all_probs" in result
    assert "class_names" in result
    assert "n_windows" in result
