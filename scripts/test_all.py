#!/usr/bin/env python3
"""Run each market's tests in isolation to avoid legacy top-level import collisions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKETS = ("cn", "hk", "us")


def main() -> int:
    failures: list[str] = []
    for market in MARKETS:
        cwd = ROOT / "systems" / market
        print(f"\n=== {market.upper()} test suite ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=cwd,
            check=False,
        )
        if result.returncode:
            failures.append(market)

    if failures:
        print("\nFailed suites: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nAll market test suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
