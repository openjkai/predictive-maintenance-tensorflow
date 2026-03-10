"""
Predict bearing fault from features — Phase 4.4
Load trained model and run inference.
"""

from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "fault_classifier.keras"


def load_model_and_meta(model_path: Path | str = DEFAULT_MODEL_PATH):
    """Load trained model and normalization metadata."""
    from tensorflow import keras

    model_path = Path(model_path)
    meta_path = model_path.with_suffix(".npz")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run training first: python scripts/train.py"
        )

    model = keras.models.load_model(model_path)
    meta = np.load(meta_path, allow_pickle=True)
    mean = meta["mean"]
    std = meta["std"]
    class_names = list(meta["class_names"])

    return model, mean, std, class_names


def predict(
    X: np.ndarray,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> tuple[np.ndarray, list[str]]:
    """
    Predict fault class for feature matrix X.

    Args:
        X: (n_samples, n_features) — time-domain + FFT + wavelet features

    Returns:
        pred_classes: (n_samples,) integer predictions
        class_names: list of class labels
    """
    model, mean, std, class_names = load_model_and_meta(model_path)

    X_norm = (X - mean) / std
    probs = model.predict(X_norm, verbose=0)
    pred_classes = np.argmax(probs, axis=1)

    return pred_classes, class_names


def predict_single(
    features: list[float] | np.ndarray,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Predict for one sample. Convenience for demo/CLI.

    Args:
        features: [rms, peak, mean, std, kurtosis, spectral_centroid,
                   spectral_bandwidth, wavelet_energy_d1, wavelet_energy_a1]

    Returns:
        dict with predicted_class, probability, class_names
    """
    X = np.array([features], dtype=np.float64).reshape(1, -1)
    model, mean, std, class_names = load_model_and_meta(model_path)
    X_norm = (X - mean) / std
    probs = model.predict(X_norm, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "predicted_class": class_names[pred_idx],
        "probability": float(probs[pred_idx]),
        "all_probs": dict(zip(class_names, map(float, probs))),
        "class_names": class_names,
    }


def predict_from_file(
    mat_path: Path | str,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    n_windows: int = 10,
) -> dict:
    """
    Predict fault from a .mat file. Extracts features from first n_windows,
    predicts each, returns majority vote + mean confidence.

    Args:
        mat_path: Path to CWRU .mat file
        model_path: Path to trained model
        n_windows: Number of windows to sample (default 10)

    Returns:
        dict with predicted_class, probability, all_probs, class_names
    """
    from src.feature_engineering import extract_features, sliding_windows
    from src.load_data import load_mat_file

    signal, _rate, _rpm = load_mat_file(mat_path)
    windows = sliding_windows(signal)[:n_windows]
    if not windows:
        raise ValueError(f"Signal too short for windows: {len(signal)} samples")

    X = np.array([extract_features(w) for w in windows], dtype=np.float64)
    pred_classes, class_names = predict(X, model_path)

    # Majority vote
    from collections import Counter

    majority = Counter(pred_classes).most_common(1)[0][0]
    prob = (pred_classes == majority).mean()

    model, mean, std, _ = load_model_and_meta(model_path)
    X_norm = (X - mean) / std
    probs = model.predict(X_norm, verbose=0)
    mean_probs = probs.mean(axis=0)

    return {
        "predicted_class": class_names[majority],
        "probability": float(mean_probs[majority]),
        "all_probs": dict(zip(class_names, map(float, mean_probs))),
        "class_names": class_names,
        "n_windows": len(windows),
    }


RAW_MODEL_PATH = MODELS_DIR / "fault_classifier_raw.keras"


def load_raw_model_and_meta(model_path: Path | str = RAW_MODEL_PATH):
    """Load raw-signal model (1D-CNN/LSTM) and metadata."""
    from tensorflow import keras

    model_path = Path(model_path)
    meta_path = model_path.with_suffix(".npz")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Raw model not found: {model_path}. Run: python scripts/train_raw.py"
        )

    model = keras.models.load_model(model_path)
    meta = np.load(meta_path, allow_pickle=True)
    mean = meta["mean"]
    std = meta["std"]
    class_names = list(meta["class_names"])
    window_size = int(meta.get("window_size", 1024))

    return model, mean, std, class_names, window_size


def predict_from_file_raw(
    mat_path: Path | str,
    model_path: Path | str = RAW_MODEL_PATH,
    n_windows: int = 10,
) -> dict:
    """
    Predict from .mat file using raw-signal model (1D-CNN/LSTM).
    Loads signal, creates windows, runs through raw model.
    """
    from src.feature_engineering import sliding_windows
    from src.load_data import load_mat_file

    mat_path = Path(mat_path)
    signal, _rate, _rpm = load_mat_file(mat_path)

    model, mean, std, class_names, window_size = load_raw_model_and_meta(model_path)
    windows = sliding_windows(signal, window_size=window_size)[:n_windows]
    if not windows:
        raise ValueError(f"Signal too short: {len(signal)} samples, need {window_size}")

    X = np.array(windows, dtype=np.float32)[..., np.newaxis]
    X_norm = (X - mean) / np.where(std < 1e-8, 1.0, std)
    probs = model.predict(X_norm, verbose=0)
    mean_probs = probs.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))

    return {
        "predicted_class": class_names[pred_idx],
        "probability": float(mean_probs[pred_idx]),
        "all_probs": dict(zip(class_names, map(float, mean_probs))),
        "class_names": class_names,
        "n_windows": len(windows),
        "mat_path": str(mat_path),
    }


# --- RUL (Phase 7.2) ---

RUL_MODEL_PATH = MODELS_DIR / "rul_predictor.keras"


def load_rul_model_and_meta(model_path: Path | str = RUL_MODEL_PATH):
    """Load RUL model and scaler metadata."""
    from tensorflow import keras

    model_path = Path(model_path)
    meta_path = model_path.with_suffix(".npz")

    if not model_path.exists():
        raise FileNotFoundError(
            f"RUL model not found: {model_path}. Run: python scripts/train_rul.py"
        )

    model = keras.models.load_model(model_path)
    meta = np.load(meta_path, allow_pickle=True)
    scaler_min = meta["scaler_min"]
    scaler_max = meta["scaler_max"]
    window_size = int(meta["window_size"])
    n_features = int(meta["n_features"])
    used_cols = meta["used_cols"]

    return model, scaler_min, scaler_max, window_size, n_features, used_cols


def predict_rul(
    X: np.ndarray,
    model_path: Path | str = RUL_MODEL_PATH,
) -> np.ndarray:
    """
    Predict RUL for sequence(s). X: (n, seq_len, n_features), normalized.
    Returns RUL in cycles (may clip to 0).
    """
    model, scaler_min, scaler_max, _, _, _ = load_rul_model_and_meta(model_path)
    # Scale: (X - min) / (max - min) * 2 - 1 -> [-1, 1]
    scale = scaler_max - scaler_min
    scale[scale < 1e-10] = 1.0
    X_norm = (X - scaler_min) / scale * 2.0 - 1.0
    pred = model.predict(X_norm.astype(np.float32), verbose=0).ravel()
    return np.maximum(pred, 0)


def predict_rul_from_test_engine(
    engine_df: "pd.DataFrame",
    model_path: Path | str = RUL_MODEL_PATH,
) -> dict:
    """
    Predict RUL for one test engine. engine_df has columns [unit, cycle, op1..3, s1..s21].
    Uses last window_size cycles.
    """
    import pandas as pd

    model, scaler_min, scaler_max, window_size, n_features, used_cols = load_rul_model_and_meta(
        model_path
    )
    used_cols = list(used_cols)

    arr = engine_df.sort_values("cycle").iloc[:, used_cols].values.astype(np.float32)
    if len(arr) < window_size:
        raise ValueError(f"Engine has {len(arr)} cycles, need at least {window_size}")

    window = arr[-window_size:]
    scale = scaler_max - scaler_min
    scale = np.where(scale < 1e-10, 1.0, scale)
    window_norm = (window - scaler_min) / scale * 2.0 - 1.0
    rul = float(model.predict(window_norm[np.newaxis, ...].astype(np.float32), verbose=0)[0, 0])
    rul = max(0, rul)

    return {"rul": rul, "unit": int(engine_df["unit"].iloc[0])}


if __name__ == "__main__":
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Demo: predict on a random sample (for testing)
    from src.feature_engineering import build_dataset
    from src.load_data import DATA_DIR

    X, y, _ = build_dataset(DATA_DIR, binary=False)
    sample = X[0]

    print("Predict from sample features:", sample)
    result = predict_single(sample)
    print(f"Predicted: {result['predicted_class']} ({result['probability']:.2%})")
    print("All probs:", result["all_probs"])
