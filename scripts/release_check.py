#!/usr/bin/env python3
"""Validate that the repository is safe and internally consistent for a public release."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from run_all import active_services

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "Dockerfile",
    "docker-compose.yml",
    "run_public.sh",
    "demo_data/manifest.json",
)
FORBIDDEN_TRACKED_PARTS = {
    ".market_cache",
    ".mpl_cache",
    ".runtime",
    "secrets.toml",
    ".user_stocks.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [Path(value) for value in result.stdout.decode().split("\0") if value]


def validate_fixtures(errors: list[str]) -> None:
    manifest_path = ROOT / "demo_data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markets = manifest.get("markets", {})
    if set(markets) != {"cn", "hk", "us"}:
        errors.append("demo manifest must contain cn, hk and us markets")

    for market_key, market in markets.items():
        symbols = market.get("symbols", {})
        if len(symbols) != 3:
            errors.append(f"{market_key}: expected three demo instruments")
        for symbol, metadata in symbols.items():
            fixture_path = ROOT / "demo_data" / metadata["file"]
            if not fixture_path.is_file():
                errors.append(f"{market_key}/{symbol}: fixture is missing")
                continue
            if sha256(fixture_path) != metadata.get("sha256"):
                errors.append(f"{market_key}/{symbol}: SHA-256 does not match manifest")
            with fixture_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = sum(1 for _ in reader)
                expected_columns = {"date", "open", "high", "low", "close", "volume"}
                if set(reader.fieldnames or ()) != expected_columns:
                    errors.append(f"{market_key}/{symbol}: unexpected CSV columns")
            if rows != metadata.get("rows"):
                errors.append(f"{market_key}/{symbol}: row count does not match manifest")


def validate_tracked_files(errors: list[str]) -> None:
    for tracked in tracked_files():
        if tracked.name == ".env" or any(part in FORBIDDEN_TRACKED_PARTS for part in tracked.parts):
            errors.append(f"private runtime path is tracked: {tracked}")


def main() -> int:
    errors: list[str] = []
    if not (sys.version_info.major == 3 and 11 <= sys.version_info.minor < 14):
        errors.append("Python 3.11, 3.12 or 3.13 is required")

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            errors.append(f"required release file is missing: {relative_path}")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    validate_fixtures(errors)
    validate_tracked_files(errors)

    public_services = active_services({"QUANT_PUBLIC_MODE": "true"})
    if [(service.name, service.port) for service in public_services] != [("portal", 8500)]:
        errors.append("public launcher must start only the portal on port 8500")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1

    fixture_count = sum(
        len(market["symbols"])
        for market in json.loads((ROOT / "demo_data/manifest.json").read_text(encoding="utf-8"))[
            "markets"
        ].values()
    )
    print(f"Release preflight passed for v{project['version']} ({fixture_count} fixtures).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
