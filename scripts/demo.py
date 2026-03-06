#!/usr/bin/env python3
"""
Demo: health score and maintenance recommendation — Phase 6.3
Predicts fault from sample data and outputs human-readable recommendation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_dataset
from src.load_data import DATA_DIR
from src.predict import predict_single

MODEL_PATH = ROOT / "models" / "fault_classifier.keras"


def health_score(probability: float, is_normal: bool) -> float:
    """Convert prediction to 0–100 health score. Higher = healthier."""
    if is_normal:
        return probability * 100
    return (1 - probability) * 100


def recommendation(health_score: float) -> str:
    """Maintenance recommendation based on health score."""
    if health_score >= 90:
        return "No maintenance required"
    if health_score >= 70:
        return "Monitor — schedule inspection soon"
    if health_score >= 50:
        return "Maintenance recommended"
    return "Maintenance required — inspect immediately"


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run training first:")
        print("  python scripts/train.py")
        sys.exit(1)

    print("Predictive Maintenance — Demo\n")

    # Use a sample from the dataset (in production: real sensor features)
    X, y, _ = build_dataset(DATA_DIR, binary=False)
    sample = X[0]

    result = predict_single(sample, model_path=MODEL_PATH)
    pred = result["predicted_class"]
    prob = result["probability"]
    is_normal = pred == "normal"

    score = health_score(prob, is_normal)

    print(f"Machine ID: sample_001")
    print(f"Predicted: {pred}")
    print(f"Confidence: {prob:.1%}")
    print(f"Health score: {score:.0f}%")
    print(f"Recommendation: {recommendation(score)}")
    print(f"\nAll class probabilities: {result['all_probs']}")


if __name__ == "__main__":
    main()
