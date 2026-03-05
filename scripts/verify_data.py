#!/usr/bin/env python3
"""
Verify CWRU data — Phase 1, Step 1.5
Load data, print summary, plot a short segment (healthy vs fault).
Run from project root: python scripts/verify_data.py
"""

import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.load_data import DATA_DIR, load_dataset


def plot_segment(name: str, signal: np.ndarray, rate: int, duration_s: float = 0.05, ax=None):
    """Plot first `duration_s` seconds of vibration signal."""
    n_samples = int(rate * duration_s)
    segment = signal[:n_samples]
    time = np.arange(len(segment)) / rate
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(time, segment, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{name} — first {duration_s}s")
    ax.grid(True, alpha=0.3)
    return ax


def main():
    print("CWRU Data Verification — Phase 1.5\n")

    # Load all data
    data = load_dataset(DATA_DIR)
    if not data:
        print("No .mat files found in", DATA_DIR)
        print("Run: python scripts/download_cwru.py")
        sys.exit(1)

    # Summary
    print(f"Loaded {len(data)} files\n")
    print(f"{'File':<20} {'Samples':>10} {'Rate':>8} {'Duration':>10}")
    print("-" * 52)

    for name, (signal, rate, rpm) in sorted(data.items()):
        dur = len(signal) / rate
        print(f"{name:<20} {len(signal):>10,} {rate:>8,} {dur:>8.2f}s")

    # Plot healthy vs fault (first 0.05s)
    normal_key = [k for k in data.keys() if k.startswith("Normal")][0]
    fault_key = [k for k in data.keys() if k.startswith("IR")][0]

    signal_n, rate_n, _ = data[normal_key]
    signal_f, rate_f, _ = data[fault_key]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    plot_segment(normal_key, signal_n, rate_n, duration_s=0.05, ax=ax1)
    plot_segment(fault_key, signal_f, rate_f, duration_s=0.05, ax=ax2)
    fig.tight_layout()
    out_path = PROJECT_ROOT / "notebooks" / "verify_segment.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()

    print(f"\nPlot saved: {out_path}")
    print("Phase 1.5 complete — data loadable and verifiable.")


if __name__ == "__main__":
    main()
