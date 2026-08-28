"""可由 cron、launchd 或 CI 调用的 A 股资金流日扫描入口。"""

from fund_flow import FundFlowPipeline, OpportunityScanner


def progress(label: str, completed: int, total: int) -> None:
    print(f"[{completed:02d}/{total:02d}] {label}", flush=True)


if __name__ == "__main__":
    snapshot = FundFlowPipeline().run(progress=progress)
    opportunities = OpportunityScanner().run(snapshot, progress=progress)
    print(
        f"saved snapshot={snapshot.snapshot_date} "
        f"data_date={snapshot.metadata.get('data_date')} "
        f"partial={bool(snapshot.errors)} "
        f"recommended={opportunities.metadata.get('recommended_count', 0)}",
        flush=True,
    )
