#!/usr/bin/env python3
"""Export compact, non-sensitive public demo fixtures from local provider caches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "demo_data"

DATASETS = {
    "cn": {
        "label": "China A-shares",
        "currency": "CNY",
        "cost_bps": 12,
        "symbols": {
            "600519.SS": ("Kweichow Moutai", "systems/cn/.market_cache/600519_SS_history.pkl"),
            "300750.SZ": ("CATL", "systems/cn/.market_cache/300750_SZ_history.pkl"),
            "600019.SS": ("Baoshan Iron & Steel", "systems/cn/.market_cache/600019_SS_history.pkl"),
        },
    },
    "hk": {
        "label": "Hong Kong equities",
        "currency": "HKD",
        "cost_bps": 15,
        "symbols": {
            "0700.HK": ("Tencent", "systems/hk/.market_cache/0700_HK_history.pkl"),
            "9988.HK": ("Alibaba", "systems/hk/.market_cache/9988_HK_history.pkl"),
            "2318.HK": ("Ping An", "systems/hk/.market_cache/2318_HK_history.pkl"),
        },
    },
    "us": {
        "label": "US equities",
        "currency": "USD",
        "cost_bps": 7,
        "symbols": {
            "SPY": ("SPDR S&P 500 ETF", "systems/us/.market_cache/SPY_history.pkl"),
            "AAPL": ("Apple", "systems/us/.market_cache/AAPL_history.pkl"),
            "NVDA": ("NVIDIA", "systems/us/.market_cache/NVDA_history.pkl"),
        },
    },
}


def read_source_metadata(cache_path: Path) -> dict:
    sidecar = cache_path.with_suffix(".json")
    if not sidecar.exists():
        return {"source": "public provider cache", "saved_at": None}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "description": (
            "Fixed public-market snapshots for the portfolio demo. "
            "They are examples, not current quotes or investment advice."
        ),
        "markets": {},
    }

    for market_key, market in DATASETS.items():
        market_manifest = {
            "label": market["label"],
            "currency": market["currency"],
            "default_cost_bps": market["cost_bps"],
            "symbols": {},
        }
        for symbol, (name, relative_cache_path) in market["symbols"].items():
            cache_path = ROOT / relative_cache_path
            if not cache_path.exists():
                raise FileNotFoundError(f"Missing source cache: {cache_path}")

            frame = pd.read_pickle(cache_path)
            required = ["open", "high", "low", "close", "volume"]
            missing = [column for column in required if column not in frame]
            if missing:
                raise ValueError(f"{cache_path} is missing columns: {missing}")

            fixture = frame[required].copy()
            fixture.index = pd.to_datetime(fixture.index).tz_localize(None).normalize()
            fixture = fixture[~fixture.index.duplicated(keep="last")].sort_index()
            fixture = fixture.replace([float("inf"), float("-inf")], pd.NA).dropna()
            fixture = fixture.tail(640)
            fixture.index.name = "date"

            file_name = f"{market_key}_{symbol.replace('.', '_').lower()}.csv"
            fixture.to_csv(OUTPUT / file_name, float_format="%.6f")
            source_metadata = read_source_metadata(cache_path)
            market_manifest["symbols"][symbol] = {
                "name": name,
                "file": file_name,
                "rows": len(fixture),
                "start": fixture.index.min().date().isoformat(),
                "end": fixture.index.max().date().isoformat(),
                "source": source_metadata.get("source", "public provider cache"),
                "snapshot_saved_at": source_metadata.get("saved_at"),
            }
        manifest["markets"][market_key] = market_manifest

    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
