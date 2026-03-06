#!/usr/bin/env python3
"""
Train predictive maintenance model — Phase 4.
Builds features, trains Dense classifier, saves model.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_data import DATA_DIR
from src.train_model import train

DEFAULT_MODEL = ROOT / "models" / "fault_classifier.keras"


def main():
    parser = argparse.ArgumentParser(description="Train fault classifier")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--binary", action="store_true", help="Binary (normal vs fault)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Output model path")
    args = parser.parse_args()

    print("Phase 4 — Model Training\n")

    result = train(
        data_dir=DATA_DIR,
        binary=args.binary,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        random_state=42,
        model_path=args.model,
    )

    print(f"\nVal accuracy: {result['val_accuracy']:.2%}")
    print(f"Val loss: {result['val_loss']:.4f}")
    print(f"Model saved: {args.model}")
    print(f"Metadata: {result['meta_path']}")
    print("Phase 4 complete — run scripts/demo.py for inference demo.")


if __name__ == "__main__":
    main()
