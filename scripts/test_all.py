#!/usr/bin/env python3
"""Run portfolio and market tests with the isolation each suite requires."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ("portfolio", ROOT, ("portfolio/tests",)),
    ("cn", ROOT / "systems/cn", ()),
    ("cn_lab", ROOT / "systems/cn_lab", ()),
    ("hk", ROOT / "systems/hk", ()),
    ("us", ROOT / "systems/us", ()),
)


def main() -> int:
    failures: list[str] = []
    for name, cwd, targets in SUITES:
        print(f"\n=== {name.upper()} test suite ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *targets],
            cwd=cwd,
            check=False,
        )
        if result.returncode:
            failures.append(name)

    if failures:
        print("\nFailed suites: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nAll portfolio, strategy-lab and market test suites passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
