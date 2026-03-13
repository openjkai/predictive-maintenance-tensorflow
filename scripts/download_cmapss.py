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


# GitHub raw files — FD001 + FD002 (Phase 8.3)
GITHUB_RAW = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
FD_FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
]


def download_from_github(fd: int | None = None) -> bool:
    """Download FD001 (and FD002 if fd=2 or None) from GitHub raw."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    to_download = (
        [f for f in FD_FILES if f"FD00{fd}" in f] if fd in (1, 2)
        else FD_FILES
    )
    ok = 0
    for fname in to_download:
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
    return ok == len(to_download)


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
    import argparse
    parser = argparse.ArgumentParser(description="Download NASA C-MAPSS data")
    parser.add_argument("--fd", type=int, choices=[1, 2], default=None,
                        help="Download only FD001 or FD002 (default: both)")
    args = parser.parse_args()

    print("NASA C-MAPSS — RUL Dataset\n")
    print(f"Target: {DATA_DIR}")

    # Check what's needed
    want_fd = (1, 2) if args.fd is None else (args.fd,)
    needed = [fd for fd in want_fd if not (DATA_DIR / f"train_FD00{fd}.txt").exists()]
    if not needed:
        print("  Data already present. Skip download.")
        print("  Run: python scripts/train_rul.py [--fd 1|2]")
        return

    print(f"  Downloading FD00{','.join(map(str, needed))}...")
    success = download_from_github(needed[0] if len(needed) == 1 else None)
    if success:
        print("\nDone. Run: python scripts/train_rul.py [--fd 1|2]")
        return

    print("\nTrying NASA (full zip)...")
    if download_from_nasa() and extract_nasa_zip():
        print("\nDone. Run: python scripts/train_rul.py [--fd 1|2]")
        return

    print("\nManual: get files from https://github.com/edwardzjl/CMAPSSData")


if __name__ == "__main__":
    main()
