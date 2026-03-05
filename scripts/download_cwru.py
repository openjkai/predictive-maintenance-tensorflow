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

# Extended set: more data for better model accuracy (Tier 2)
# - Normal at all loads (0-3 HP)
# - 3 fault types at 0.007", 0.014", 0.021" @ 0 HP
# Source: https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
FILES = {
    # Normal baseline — all loads
    "Normal_0.mat": "97",
    "Normal_1.mat": "98",
    "Normal_2.mat": "99",
    "Normal_3.mat": "100",
    # 0.007" faults @ 0 HP
    "IR007_0.mat": "105",
    "B007_0.mat": "118",
    "OR007@6_0.mat": "130",
    # 0.014" faults @ 0 HP (larger = more severe)
    "IR014_0.mat": "169",
    "B014_0.mat": "185",
    "OR014@6_0.mat": "197",
    # 0.021" faults @ 0 HP
    "IR021_0.mat": "209",
    "B021_0.mat": "222",
    "OR021@6_0.mat": "234",
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
        print("Ready for Phase 1.5 — run: python scripts/verify_data.py")
    else:
        print("Some downloads failed. Check URLs or try manual download from:")
        print("  https://engineering.case.edu/bearingdatacenter/download-data-file")


if __name__ == "__main__":
    main()
