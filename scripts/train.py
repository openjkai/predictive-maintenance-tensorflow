#!/usr/bin/env python3
"""
Train predictive maintenance model — Phase 4.
Builds features, trains Dense classifier, saves model.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_data import DATA_DIR
from src.train_model import train

DEFAULT_MODEL = ROOT / "models" / "fault_classifier.keras"


def main():
    print("Phase 4 — Model Training\n")

    result = train(
        data_dir=DATA_DIR,
        binary=False,
        epochs=50,
        batch_size=64,
        val_frac=0.2,
        random_state=42,
        model_path=DEFAULT_MODEL,
    )

    print(f"\nVal accuracy: {result['val_accuracy']:.2%}")
    print(f"Val loss: {result['val_loss']:.4f}")
    print(f"Model saved: {DEFAULT_MODEL}")
    print(f"Metadata: {result['meta_path']}")
    print("Phase 4 complete — run predict.py for inference.")


if __name__ == "__main__":
    main()
