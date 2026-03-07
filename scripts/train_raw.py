#!/usr/bin/env python3
"""
Train raw-signal model (1D-CNN or LSTM) — Future improvement.
Uses raw vibration windows instead of hand-crafted features.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_raw_dataset, train_val_split
from src.load_data import DATA_DIR
from src.raw_model import train

DEFAULT_MODEL = ROOT / "models" / "fault_classifier_raw.keras"


def main():
    parser = argparse.ArgumentParser(description="Train raw-signal fault classifier")
    parser.add_argument("--arch", choices=["1dcnn", "lstm"], default="1dcnn", help="Model: 1dcnn or lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--out", type=str, default=str(DEFAULT_MODEL), help="Output model path")
    args = parser.parse_args()

    print("Raw-signal model — 1D-CNN / LSTM\n")
    print(f"Architecture: {args.arch}")

    result = train(
        data_dir=DATA_DIR,
        model_type=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        model_path=Path(args.out),
    )

    print(f"\nVal accuracy: {result['val_accuracy']:.2%}")
    print(f"Model saved: {args.out}")
    print("Run: python scripts/demo_raw.py [data/file.mat]")


if __name__ == "__main__":
    main()
