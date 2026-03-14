#!/usr/bin/env python3
"""
Run full pipeline: download data → train models → demos.
Phase 9.1 — One command to set up and run everything.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], desc: str) -> bool:
    """Run command, return success."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full predictive maintenance pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--skip-bearing", action="store_true", help="Skip bearing models")
    parser.add_argument("--skip-rul", action="store_true", help="Skip RUL models")
    parser.add_argument("--rul-fd", type=int, choices=[1, 2, 3, 4], default=1, help="RUL dataset (1-4)")
    parser.add_argument("--quick", action="store_true", help="Quick run: fewer epochs")
    args = parser.parse_args()

    venv_py = ROOT / "venv" / "bin" / "python"
    python = str(venv_py) if venv_py.exists() else "python"
    scripts = ROOT / "scripts"

    print("\nPredictive Maintenance — Full Pipeline\n")

    if not args.skip_download:
        if not run([python, str(scripts / "download_cwru.py")], "1. Download CWRU bearing data"):
            print("CWRU download failed. Exiting.")
            sys.exit(1)
        if not args.skip_rul:
            if not run([python, str(scripts / "download_cmapss.py")], "2. Download NASA C-MAPSS"):
                print("C-MAPSS download failed. Continuing without RUL...")
                args.skip_rul = True
    else:
        print("\nSkipping download (--skip-download)")

    if not args.skip_bearing:
        epochs = "5" if args.quick else "50"
        if not run(
            [python, str(scripts / "train.py"), "--epochs", epochs],
            "3. Train bearing feature model",
        ):
            print("Bearing train failed.")
        run(
            [python, str(scripts / "demo.py")],
            "4. Bearing demo",
        )
    else:
        print("\nSkipping bearing (--skip-bearing)")

    if not args.skip_rul:
        epochs = "5" if args.quick else "30"
        if not run(
            [python, str(scripts / "train_rul.py"), "--fd", str(args.rul_fd), "--epochs", epochs],
            f"5. Train RUL (FD00{args.rul_fd})",
        ):
            print("RUL train failed.")
        run(
            [python, str(scripts / "demo_rul.py"), "--fd", str(args.rul_fd), "-n", "5"],
            "6. RUL demo",
        )
    else:
        print("\nSkipping RUL (--skip-rul)")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Dashboard: streamlit run scripts/dashboard.py")
    print("  Tests: python -m pytest tests/ -v")
    print("=" * 60)


if __name__ == "__main__":
    main()
