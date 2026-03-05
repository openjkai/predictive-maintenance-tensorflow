#!/usr/bin/env python3
"""
Download CWRU Bearing dataset (12k Drive End) — Phase 1, Step 1.3
Source: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
"""

import os
import urllib.request
from pathlib import Path

# Output directory
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Minimal starter set: Normal + 3 fault types (0.007" @ 0 HP, 1797 RPM)
# Enough for binary (normal vs fault) and later multi-class
FILES = {
    # Normal baseline
    "Normal_0.mat": "97",
    # Inner race fault
    "IR007_0.mat": "105",
    # Ball fault
    "B007_0.mat": "118",
    # Outer race fault (centered @ 6:00)
    "OR007@6_0.mat": "130",
}

BASE_URL = "https://engineering.case.edu/sites/default/files/{}.mat"


def download_file(name: str, file_id: str) -> bool:
    """Download one .mat file from CWRU."""
    url = BASE_URL.format(file_id)
    out_path = DATA_DIR / name
    if out_path.exists():
        print(f"  Skip (exists): {name}")
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        out_path.write_bytes(data)
        print(f"  Downloaded: {name}")
        return True
    except Exception as e:
        print(f"  Failed {name}: {e}")
        return False


def main():
    print("CWRU Bearing Data — Downloading to", DATA_DIR)
    success = 0
    for name, fid in FILES.items():
        if download_file(name, fid):
            success += 1
    print(f"\nDone: {success}/{len(FILES)} files")
    if success == len(FILES):
        print("Ready for Phase 1, Step 1.4 (load_data.py)")
    else:
        print("Some downloads failed. Check URLs or try manual download from:")
        print("  https://engineering.case.edu/bearingdatacenter/download-data-file")


if __name__ == "__main__":
    main()
