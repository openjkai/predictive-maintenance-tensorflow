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
from src.rul_model import train, _model_path_for_fd

DEFAULT_MODEL = ROOT / "models" / "rul_predictor.keras"


def main():
    parser = argparse.ArgumentParser(description="Train RUL predictor on NASA C-MAPSS FD001/FD002")
    parser.add_argument("--fd", type=int, choices=[1, 2, 3, 4], default=1,
                        help="FD001/002/003/004 (1=1op, 2=6op, 3=1op+2fault, 4=6op+2fault)")
    parser.add_argument("--data-dir", type=Path, default=CMAPSS_DIR, help="C-MAPSS data directory")
    parser.add_argument("--window-size", type=int, default=30, help="Sequence length (cycles)")
    parser.add_argument("--max-rul", type=int, default=125, help="RUL cap (piecewise linear)")
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=None, help="Output model path (default: rul_predictor_fd00N.keras)")
    args = parser.parse_args()

    out_path = args.out or _model_path_for_fd(args.fd)
    print(f"NASA C-MAPSS — RUL Prediction (FD00{args.fd})\n")

    result = train(
        data_dir=args.data_dir,
        fd=args.fd,
        window_size=args.window_size,
        max_rul=args.max_rul,
        lstm_units=args.lstm_units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        model_path=out_path,
    )

    print(f"\nModel saved: {out_path}")
    print(f"Val RMSE: {result['val_rmse']:.2f} cycles")
    if result.get("test_rmse") is not None:
        print(f"Test RMSE: {result['test_rmse']:.2f} cycles")
        print(f"Test score (NASA): {result['test_score']:.2f}")
    print("\nRun: python scripts/demo_rul.py")


if __name__ == "__main__":
    main()
