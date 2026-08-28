"""Streamlit 服务存活期间的港股收盘后扫描调度器。"""

from __future__ import annotations

from datetime import datetime, time, timedelta
import json
import os
from pathlib import Path
import threading

from .opportunity_scanner import OpportunityScanner
from .pipeline import HONG_KONG_TZ, MarketScanPipeline, SnapshotStore


DEFAULT_SCAN_TIME = time(16, 20)


def should_scan_today(now: datetime, available_dates: list[str], scan_time: time = DEFAULT_SCAN_TIME) -> bool:
    current = now.astimezone(HONG_KONG_TZ) if now.tzinfo else now.replace(tzinfo=HONG_KONG_TZ)
    return (
        current.weekday() < 5
        and current.time() >= scan_time
        and current.date().isoformat() not in available_dates
    )


class DailyMarketScanScheduler:
    def __init__(
        self,
        pipeline: MarketScanPipeline | None = None,
        opportunity_scanner: OpportunityScanner | None = None,
        scan_time: time = DEFAULT_SCAN_TIME,
        check_seconds: int = 60,
    ):
        self.pipeline = pipeline or MarketScanPipeline()
        self.store: SnapshotStore = self.pipeline.store
        self.opportunity_scanner = opportunity_scanner or OpportunityScanner()
        self.scan_time = scan_time
        self.check_seconds = max(15, check_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_retry_at: datetime | None = None

    @property
    def status_path(self) -> Path:
        return self.store.data_dir / "scheduler_status.json"

    def _write_status(self, state: str, **extra) -> None:
        self.store.data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": state,
            "updated_at": datetime.now(HONG_KONG_TZ).isoformat(timespec="seconds"),
            "scan_time": self.scan_time.strftime("%H:%M"),
            **extra,
        }
        temporary = self.status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.status_path)

    def start(self) -> "DailyMarketScanScheduler":
        if self._thread and self._thread.is_alive():
            return self
        self._thread = threading.Thread(target=self._run, name="hk-daily-market-scan", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        self._write_status("waiting", message="等待下一个港股交易日收盘扫描")
        while not self._stop_event.wait(self.check_seconds):
            now = datetime.now(HONG_KONG_TZ)
            if self._next_retry_at and now < self._next_retry_at:
                continue
            if not should_scan_today(now, self.store.dates(), self.scan_time):
                continue
            try:
                self._write_status("running", message="正在执行港股全市场扫描")
                snapshot = self.pipeline.run()
                opportunity_error = None
                recommended_count = 0
                try:
                    result = self.opportunity_scanner.run(snapshot)
                    recommended_count = int(result.metadata.get("recommended_count", 0))
                except Exception as exc:
                    opportunity_error = f"{type(exc).__name__}: {exc}"
                self._next_retry_at = None
                self._write_status(
                    "success",
                    message=(
                        f"今日扫描已保存，强信心候选 {recommended_count} 只"
                        if opportunity_error is None
                        else "市场快照已保存，候选策略评分待重试"
                    ),
                    snapshot_date=snapshot.snapshot_date,
                    data_date=snapshot.metadata.get("data_date"),
                    opportunity_error=opportunity_error,
                    recommended_count=recommended_count,
                )
            except Exception as exc:
                self._next_retry_at = now + timedelta(minutes=30)
                self._write_status(
                    "error",
                    message="扫描失败，30 分钟后自动重试",
                    error=f"{type(exc).__name__}: {exc}",
                )


def read_scheduler_status(store: SnapshotStore | None = None) -> dict:
    path = (store or SnapshotStore()).data_dir / "scheduler_status.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"state": "not_started", "message": "自动扫描器尚未启动"}
