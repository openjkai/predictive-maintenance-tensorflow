#!/usr/bin/env python3
"""
Run prediction — from sample features or from a .mat file.
Usage:
  python scripts/run_predict.py                    # Demo with sample from dataset
  python scripts/run_predict.py data/IR007_0.mat # Predict from .mat file
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_dataset
from src.load_data import DATA_DIR
from src.predict import predict_from_file, predict_single

MODEL_PATH = ROOT / "models" / "fault_classifier.keras"


def health_score(probability: float, is_normal: bool) -> float:
    """0–100 health score. Higher = healthier."""
    return (probability * 100) if is_normal else ((1 - probability) * 100)


def recommendation(score: float) -> str:
    if score >= 90:
        return "No maintenance required"
    if score >= 70:
        return "Monitor — schedule inspection soon"
    if score >= 50:
        return "Maintenance recommended"
    return "Maintenance required — inspect immediately"


def main():
    parser = argparse.ArgumentParser(description="Predict bearing fault from features or .mat file")
    parser.add_argument(
        "mat_file",
        nargs="?",
        default=None,
        help="Path to .mat file (optional; uses sample if omitted)",
    )
    parser.add_argument(
        "--machine-id",
        default="sample_001",
        help="Machine ID for output (default: sample_001)",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print("Model not found. Run training first:")
        print("  python scripts/train.py")
        sys.exit(1)

    if args.mat_file:
        mat_path = Path(args.mat_file)
        if not mat_path.exists():
            print(f"File not found: {mat_path}")
            sys.exit(1)
        result = predict_from_file(mat_path, model_path=MODEL_PATH)
        print(f"Predict from: {mat_path.name} ({result['n_windows']} windows)\n")
    else:
        X, _, _ = build_dataset(DATA_DIR, binary=False)
        result = predict_single(X[0], model_path=MODEL_PATH)

    pred = result["predicted_class"]
    prob = result["probability"]
    is_normal = pred == "normal"
    score = health_score(prob, is_normal)

    print(f"Machine ID: {args.machine_id}")
    print(f"Predicted: {pred}")
    print(f"Confidence: {prob:.1%}")
    print(f"Health score: {score:.0f}%")
    print(f"Recommendation: {recommendation(score)}")
    print(f"\nAll class probabilities: {result['all_probs']}")


if __name__ == "__main__":
    main()
