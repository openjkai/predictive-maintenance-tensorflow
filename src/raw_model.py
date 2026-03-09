"""
Raw signal models — 1D-CNN and LSTM for vibration windows.
Takes raw (window_size,) input instead of hand-crafted features.
"""

from pathlib import Path

import numpy as np

from src.feature_engineering import (
    CLASS_NAMES,
    build_raw_dataset,
    train_val_split,
)
from src.load_data import DATA_DIR

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "fault_classifier_raw.keras"


def build_1d_cnn(
    window_size: int = 1024,
    n_classes: int = 4,
) -> "keras.Model":
    """1D-CNN for raw vibration: Conv1D layers + pooling."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window_size, 1)),
            keras.layers.Conv1D(32, 64, activation="relu", padding="same"),
            keras.layers.MaxPooling1D(4),
            keras.layers.Conv1D(64, 32, activation="relu", padding="same"),
            keras.layers.MaxPooling1D(4),
            keras.layers.Conv1D(64, 16, activation="relu", padding="same"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="fault_classifier_1dcnn",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(
    window_size: int = 1024,
    n_classes: int = 4,
    lstm_units: int = 64,
) -> "keras.Model":
    """LSTM for raw vibration time series."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(window_size, 1)),
            keras.layers.LSTM(lstm_units, return_sequences=True),
            keras.layers.LSTM(lstm_units // 2, return_sequences=False),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="fault_classifier_lstm",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _plot_training_curves(history: dict, out_path: Path) -> None:
    """Save loss and accuracy curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_range = range(1, len(history["loss"]) + 1)

    ax1.plot(epochs_range, history["loss"], label="train")
    ax1.plot(epochs_range, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training loss (raw model)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["accuracy"], label="train")
    ax2.plot(epochs_range, history["val_accuracy"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training accuracy (raw model)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Training curves saved: {out_path}")


def _compute_class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
    """Phase 7.3: Balanced class weights for imbalanced data."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(n_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def train(
    data_dir: Path | str = DATA_DIR,
    model_type: str = "1dcnn",
    window_size: int = 1024,
    step: int = 512,
    binary: bool = False,
    epochs: int = 30,
    batch_size: int = 64,
    val_frac: float = 0.2,
    random_state: int = 42,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    use_class_weights: bool = True,
) -> dict:
    """
    Train raw-signal model (1D-CNN or LSTM).

    Args:
        model_type: "1dcnn" or "lstm"
        window_size: Samples per window
        step: Step between windows

    Returns:
        dict with history, val_accuracy, val_loss, etc.
    """
    from tensorflow import keras

    # Build raw dataset: X (n, window_size, 1), y (n,)
    X, y = build_raw_dataset(
        data_dir, window_size=window_size, step=step, binary=binary
    )
    n_classes = len(np.unique(y))
    class_names = ["normal", "fault"] if binary else CLASS_NAMES[:n_classes]

    X_train, X_val, y_train, y_val = train_val_split(
        X, y, val_frac=val_frac, random_state=random_state
    )

    # Normalize per channel (simple z-score on each window)
    mean, std = X_train.mean(axis=(0, 1)), X_train.std(axis=(0, 1))
    std[std < 1e-8] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Build model
    if model_type.lower() == "lstm":
        model = build_lstm(window_size=window_size, n_classes=n_classes)
    else:
        model = build_1d_cnn(window_size=window_size, n_classes=n_classes)

    # Phase 7.3: Class weights
    class_weight = None
    if use_class_weights:
        class_weight = _compute_class_weights(y_train, n_classes)
        print(f"Class weights (balanced): {class_weight}")

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            ),
        ],
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # Confusion matrix
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for true_idx, pred_idx in zip(y_val, y_pred):
        cm[true_idx, pred_idx] += 1
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("       ", " ".join(f"{c[:4]:>5}" for c in class_names))
    for i, name in enumerate(class_names):
        print(f"{name[:6]:>6}", " ".join(f"{cm[i, j]:>5}" for j in range(n)))

    # Per-class metrics
    recall = np.zeros(n)
    precision = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        recall[i] = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        precision[i] = tp / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name:12} recall={recall[i]:.2%}  precision={precision[i]:.2%}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    model.save(model_path)

    meta_path = model_path.with_suffix(".npz")
    np.savez(
        meta_path,
        mean=mean,
        std=std,
        class_names=np.array(class_names, dtype=object),
        window_size=window_size,
        model_type=model_type,
    )

    # Training curves
    figs_dir = Path(__file__).resolve().parents[1] / "notebooks"
    figs_dir.mkdir(parents=True, exist_ok=True)
    _plot_training_curves(history.history, figs_dir / "training_curves_raw.png")

    return {
        "history": history.history,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "class_names": class_names,
        "y_val": y_val,
        "y_pred": y_pred,
        "meta_path": str(meta_path),
    }
