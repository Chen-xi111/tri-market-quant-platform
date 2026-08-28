"""A 股资金流扫描、快照与调度。"""

from .pipeline import (
    AKShareFundFlowProvider,
    FundFlowPipeline,
    FundFlowUnavailable,
    ScanSnapshot,
    SnapshotStore,
)
from .opportunity_scanner import (
    OpportunityScanner,
    OpportunitySnapshot,
    OpportunityStore,
    build_candidate_pool,
)

__all__ = [
    "AKShareFundFlowProvider",
    "FundFlowPipeline",
    "FundFlowUnavailable",
    "ScanSnapshot",
    "SnapshotStore",
    "OpportunityScanner",
    "OpportunitySnapshot",
    "OpportunityStore",
    "build_candidate_pool",
]
