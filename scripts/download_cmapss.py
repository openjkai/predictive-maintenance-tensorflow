#!/usr/bin/env python3
"""
Download NASA C-MAPSS dataset for RUL (Remaining Useful Life) prediction.
Phase 7.2 — Turbofan engine degradation data.
Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
"""

import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cmapss"
ZIP_NAME = "CMAPSSData.zip"

# Multiple mirrors in case one is down
DOWNLOAD_URLS = [
    "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
    "https://github.com/edwardzjl/CMAPSSData/archive/refs/heads/master.zip",
]

# If using GitHub fallback, the structure is CMAPSSData-master/train_FD001.txt etc.
GITHUB_PREFIX = "CMAPSSData-master/"


def download_from_nasa() -> bool:
    """Download from NASA legacy URL."""
    url = DOWNLOAD_URLS[0]
    out_path = DATA_DIR / ZIP_NAME
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        print(f"  Downloaded: {ZIP_NAME} from NASA")
        return True
    except Exception as e:
        print(f"  NASA download failed: {e}")
        return False


# GitHub raw files for FD001 (simplest dataset)
GITHUB_RAW = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
FD001_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]


def download_from_github() -> bool:
    """Download FD001 from GitHub raw (reliable fallback)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for fname in FD001_FILES:
        url = f"{GITHUB_RAW}/{fname}"
        out_path = DATA_DIR / fname
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=120) as resp:
                out_path.write_bytes(resp.read())
            print(f"  Downloaded: {fname}")
            ok += 1
        except Exception as e:
            print(f"  Failed {fname}: {e}")
    return ok == len(FD001_FILES)


def extract_nasa_zip() -> bool:
    """Extract NASA CMAPSSData.zip."""
    zip_path = DATA_DIR / ZIP_NAME
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
            for name in zf.namelist():
                print(f"  Extracted: {name}")
        return True
    except Exception as e:
        print(f"  Extract failed: {e}")
        return False


def main():
    print("NASA C-MAPSS — RUL Dataset (FD001)\n")
    print(f"Target: {DATA_DIR}")

    # Check if already present
    train_fd001 = DATA_DIR / "train_FD001.txt"
    if train_fd001.exists():
        print("  FD001 already present. Skip download.")
        print("  Run: python scripts/train_rul.py")
        return

    # Try GitHub first (reliable); NASA zip as optional
    print("Downloading from GitHub...")
    if download_from_github():
        print("\nDone. Run: python scripts/train_rul.py")
        return

    print("\nTrying NASA...")
    if download_from_nasa() and extract_nasa_zip():
        print("\nDone. Run: python scripts/train_rul.py")
        return

    print("\nManual download:")
    print("  1. Get files from https://github.com/edwardzjl/CMAPSSData")
    print("  2. Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in:", DATA_DIR)


if __name__ == "__main__":
    main()
