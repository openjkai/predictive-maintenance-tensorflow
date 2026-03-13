#!/usr/bin/env python3
"""
Demo: RUL (Remaining Useful Life) prediction — Phase 7.2 / 8.3
NASA C-MAPSS FD001 or FD002 — predict cycles until failure.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_cmapss import load_fd001, CMAPSS_DIR
from src.predict import load_rul_model_and_meta
from src.rul_model import _model_path_for_fd

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="RUL demo on FD001 or FD002")
    parser.add_argument("--fd", type=int, choices=[1, 2], default=1)
    parser.add_argument("-n", "--engines", type=int, default=5, help="Number of test engines to show")
    args = parser.parse_args()

    model_path = _model_path_for_fd(args.fd)
    if not model_path.exists():
        print(f"RUL model for FD00{args.fd} not found. Run first:")
        print("  python scripts/download_cmapss.py")
        print(f"  python scripts/train_rul.py --fd {args.fd}")
        sys.exit(1)

    print(f"NASA C-MAPSS — RUL Demo (FD00{args.fd})\n")

    train_df, test_df, true_rul = load_fd001(CMAPSS_DIR, fd=args.fd)

    model, scaler_min, scaler_max, window_size, n_features, used_cols = load_rul_model_and_meta(
        model_path
    )
    used_cols = list(used_cols)

    test_engines = test_df["unit"].unique()
    predictions = []
    for unit in test_engines[: args.engines]:
        unit_df = test_df[test_df["unit"] == unit].sort_values("cycle")
        arr = unit_df.iloc[:, used_cols].values.astype(np.float32)
        if len(arr) < window_size:
            continue
        window = arr[-window_size:]
        scale = scaler_max - scaler_min
        scale = np.where(scale < 1e-10, 1.0, scale)
        window_norm = (window - scaler_min) / scale * 2.0 - 1.0
        rul = float(model.predict(window_norm[np.newaxis, ...].astype(np.float32), verbose=0)[0, 0])
        rul = max(0, rul)
        predictions.append((unit, rul))

    # Show results (compare with true RUL for first 5)
    print("Engine  | Predicted RUL | True RUL | Error")
    print("-" * 45)
    for i, (unit, pred_rul) in enumerate(predictions):
        true = true_rul[i] if i < len(true_rul) else "?"
        err = abs(pred_rul - true) if i < len(true_rul) else ""
        print(f"  {unit:3d}  | {pred_rul:12.1f} | {str(true):>8} | {err}")

    print("\nRUL = Remaining Useful Life (cycles until failure)")
    print("Lower RUL = closer to failure. Use --fd 2 for FD002 (6 op conditions).")


if __name__ == "__main__":
    main()
