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
