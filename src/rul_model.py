"""
RUL (Remaining Useful Life) prediction — Phase 7.2
LSTM model for NASA C-MAPSS turbofan engine degradation.
"""

from pathlib import Path

import numpy as np

from src.load_cmapss import CMAPSS_DIR, prepare_fd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "rul_predictor.keras"


def _model_path_for_fd(fd: int) -> Path:
    """Model path for given FD (1–4)."""
    return MODELS_DIR / f"rul_predictor_fd00{fd}.keras"


def build_rul_lstm(
    seq_len: int = 30,
    n_features: int = 21,
    lstm_units: int = 64,
) -> "keras.Model":
    """LSTM for RUL regression: sequence in, scalar RUL out."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(seq_len, n_features)),
            keras.layers.LSTM(lstm_units, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(lstm_units // 2, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ],
        name="rul_predictor",
    )
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )
    return model


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error — standard metric for RUL."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _compute_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA scoring function: heavier penalty for late predictions (underestimation).
    score = sum(d) where d = exp(alpha*(y_pred - y_true)) - 1 if pred < true else exp(beta*(y_true - y_pred)) - 1
    alpha=1/13, beta=1/10
    """
    diff = y_pred - y_true
    alpha, beta = 1 / 13, 1 / 10
    d = np.where(diff < 0, np.exp(-beta * diff) - 1, np.exp(alpha * diff) - 1)
    return float(np.sum(d))


def _plot_curves(history: dict, out_path: Path) -> None:
    """Save loss and MAE curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_range = range(1, len(history["loss"]) + 1)

    ax1.plot(epochs_range, history["loss"], label="train")
    ax1.plot(epochs_range, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.set_title("RUL — Training loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["mae"], label="train")
    ax2.plot(epochs_range, history["val_mae"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.set_title("RUL — MAE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Training curves saved: {out_path}")


def train(
    data_dir: Path | str = CMAPSS_DIR,
    fd: int = 1,
    window_size: int = 30,
    max_rul: int = 125,
    lstm_units: int = 64,
    epochs: int = 50,
    batch_size: int = 64,
    val_frac: float = 0.2,
    random_state: int = 42,
    model_path: Path | str | None = None,
) -> dict:
    """
    Train RUL predictor on FD001 or FD002 (Phase 8.3).

    Returns:
        dict with val_rmse, val_mae, test_rmse, test_score, history, etc.
    """
    from tensorflow import keras

    if model_path is None:
        model_path = _model_path_for_fd(fd)

    data = prepare_fd(
        data_dir=data_dir,
        fd=fd,
        window_size=window_size,
        max_rul=max_rul,
        val_frac=val_frac,
        random_state=random_state,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test_true = data["y_test_true"]
    scaler = data["scaler"]
    n_features = data["n_features"]

    model = build_rul_lstm(
        seq_len=window_size,
        n_features=n_features,
        lstm_units=lstm_units,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    # Validation
    y_val_pred = model.predict(X_val, verbose=0).ravel()
    val_rmse = _compute_rmse(y_val, y_val_pred)
    val_mae = float(np.mean(np.abs(y_val - y_val_pred)))
    print(f"\nVal RMSE: {val_rmse:.2f} cycles")
    print(f"Val MAE:  {val_mae:.2f} cycles")

    # Test (if we have test data)
    test_rmse, test_score = None, None
    if len(X_test) > 0 and len(y_test_true) > 0:
        y_test_pred = model.predict(X_test, verbose=0).ravel()
        # Align lengths
        n_test = min(len(y_test_pred), len(y_test_true))
        test_rmse = _compute_rmse(y_test_true[:n_test], y_test_pred[:n_test])
        test_score = _compute_score(y_test_true[:n_test], y_test_pred[:n_test])
        print(f"Test RMSE: {test_rmse:.2f} cycles")
        print(f"Test score (NASA): {test_score:.2f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    model.save(model_path)

    # Save metadata for prediction (min-max scale to [-1,1])
    meta_path = Path(model_path).with_suffix(".npz")
    np.savez(
        meta_path,
        scaler_min=scaler.data_min_.astype(np.float32),
        scaler_max=scaler.data_max_.astype(np.float32),
        window_size=window_size,
        n_features=n_features,
        used_cols=np.array(data["used_cols"]),
        max_rul=max_rul,
        fd=fd,
    )

    # Training curves
    figs_dir = Path(__file__).resolve().parents[1] / "notebooks"
    figs_dir.mkdir(parents=True, exist_ok=True)
    curve_name = f"training_curves_rul_fd00{fd}.png"
    _plot_curves(history.history, figs_dir / curve_name)

    return {
        "history": history.history,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "test_rmse": test_rmse,
        "test_score": test_score,
        "meta_path": str(meta_path),
    }
