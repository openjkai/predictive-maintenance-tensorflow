#!/usr/bin/env python3
"""
Train RUL (Remaining Useful Life) predictor — Phase 7.2
NASA C-MAPSS FD001 turbofan engine degradation.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_cmapss import CMAPSS_DIR
from src.rul_model import train

DEFAULT_MODEL = ROOT / "models" / "rul_predictor.keras"


def main():
    parser = argparse.ArgumentParser(description="Train RUL predictor on NASA C-MAPSS FD001")
    parser.add_argument("--data-dir", type=Path, default=CMAPSS_DIR, help="C-MAPSS data directory")
    parser.add_argument("--window-size", type=int, default=30, help="Sequence length (cycles)")
    parser.add_argument("--max-rul", type=int, default=125, help="RUL cap (piecewise linear)")
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL, help="Output model path")
    args = parser.parse_args()

    print("NASA C-MAPSS — RUL Prediction (FD001)\n")

    result = train(
        data_dir=args.data_dir,
        window_size=args.window_size,
        max_rul=args.max_rul,
        lstm_units=args.lstm_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        model_path=args.out,
    )

    print(f"\nModel saved: {args.out}")
    print(f"Val RMSE: {result['val_rmse']:.2f} cycles")
    if result.get("test_rmse") is not None:
        print(f"Test RMSE: {result['test_rmse']:.2f} cycles")
        print(f"Test score (NASA): {result['test_score']:.2f}")
    print("\nRun: python scripts/demo_rul.py")


if __name__ == "__main__":
    main()
