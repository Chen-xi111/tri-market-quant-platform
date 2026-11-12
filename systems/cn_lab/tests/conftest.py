"""Make the isolated strategy-lab suite importable from any working directory."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[3]
CN_DIR = PROJECT_DIR / "systems" / "cn"
for import_path in (PROJECT_DIR, CN_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
