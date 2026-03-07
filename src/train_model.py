"""
Train fault classifier — Phase 4
Simple Dense model on time-domain features (RMS, peak, mean, std, kurtosis).
"""

from pathlib import Path

import numpy as np

from src.feature_engineering import (
    CLASS_NAMES,
    build_dataset,
    train_val_split,
)
from src.load_data import DATA_DIR

# Model save path
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "fault_classifier.keras"


def _plot_training_curves(history: dict, out_path: Path) -> None:
    """Save loss and accuracy curves to file."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs_range = range(1, len(history["loss"]) + 1)

    ax1.plot(epochs_range, history["loss"], label="train")
    ax1.plot(epochs_range, history["val_loss"], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["accuracy"], label="train")
    ax2.plot(epochs_range, history["val_accuracy"], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Training curves saved: {out_path}")


def build_model(n_classes: int, input_dim: int = 5) -> "keras.Model":
    """Build simple Dense classifier (Phase 4.1)."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ],
        name="fault_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    data_dir: Path | str = DATA_DIR,
    binary: bool = False,
    epochs: int = 50,
    batch_size: int = 64,
    val_frac: float = 0.2,
    random_state: int = 42,
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> dict:
    """
    Build dataset, train model, evaluate, save.

    Returns:
        dict with history, val_accuracy, val_loss, class_names
    """
    from tensorflow import keras

    # Data
    X, y, _ = build_dataset(data_dir, binary=binary)
    n_classes = len(np.unique(y))
    class_names = ["normal", "fault"] if binary else CLASS_NAMES[:n_classes]

    X_train, X_val, y_train, y_val = train_val_split(
        X, y, val_frac=val_frac, random_state=random_state
    )

    # Normalize features (helps training)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # Model
    model = build_model(n_classes=n_classes, input_dim=X.shape[1])

    # Train
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

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

    # Confusion matrix (Phase 6.4)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for true_idx, pred_idx in zip(y_val, y_pred):
        cm[true_idx, pred_idx] += 1
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("       ", " ".join(f"{c[:4]:>5}" for c in class_names))
    for i, name in enumerate(class_names):
        print(f"{name[:6]:>6}", " ".join(f"{cm[i, j]:>5}" for j in range(n)))

    # Per-class recall and precision (Phase 5)
    recall = np.zeros(n)
    precision = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        recall[i] = tp / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        precision[i] = tp / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    print("\nPer-class metrics:")
    for i, name in enumerate(class_names):
        print(f"  {name:12} recall={recall[i]:.2%}  precision={precision[i]:.2%}")

    # Save training curves (Phase 6.4)
    figs_dir = Path(__file__).resolve().parents[1] / "notebooks"
    figs_dir.mkdir(parents=True, exist_ok=True)
    _plot_training_curves(history.history, figs_dir / "training_curves.png")

    # Save model + metadata for predict.py
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)
    model.save(model_path)

    # Save normalization params and class names
    meta_path = model_path.with_suffix(".npz")
    np.savez(
        meta_path,
        mean=mean,
        std=std,
        class_names=np.array(class_names, dtype=object),
    )

    return {
        "history": history.history,
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "class_names": class_names,
        "y_val": y_val,
        "y_pred": y_pred,
        "meta_path": str(meta_path),
    }


if __name__ == "__main__":
    print("Phase 4 — Train fault classifier\n")

    result = train(binary=False, epochs=50)

    print(f"\nVal accuracy: {result['val_accuracy']:.2%}")
    print(f"Val loss: {result['val_loss']:.4f}")
    print(f"Model saved: {DEFAULT_MODEL_PATH}")
    print(f"Metadata: {result['meta_path']}")
