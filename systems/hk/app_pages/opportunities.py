"""港股热度候选的 7 策略与样本外验证页面。"""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import streamlit as st

from market_scan import OpportunityScanner, OpportunitySnapshot, OpportunityStore, SnapshotStore


SCAN_STORE = SnapshotStore()
OPPORTUNITY_STORE = OpportunityStore(SCAN_STORE.data_dir)


@st.cache_data(ttl=30, max_entries=20, show_spinner=False)
def load_opportunities(snapshot_date: str) -> OpportunitySnapshot | None:
    return OpportunityStore().load(snapshot_date)


def run_scoring(snapshot_date: str) -> None:
    snapshot = SCAN_STORE.load(snapshot_date)
    if snapshot is None:
        st.error("找不到扫描快照，请先到“港股全市场扫描”。")
        return
    progress_bar = st.progress(0, text="正在组建港股候选池…")

    def update(label: str, completed: int, total: int) -> None:
        progress_bar.progress(completed / max(total, 1), text=f"{label} · {completed}/{total}")

    try:
        result = OpportunityScanner(store=OPPORTUNITY_STORE).run(snapshot, progress=update)
    except Exception as exc:
        progress_bar.empty()
        st.error(f"批量评分未完成：{type(exc).__name__}: {exc}")
        return
    progress_bar.empty()
    load_opportunities.clear()
    st.session_state["hk_opportunity_flash"] = (
        f"已评估 {result.metadata.get('candidate_count', 0)} 只，"
        f"强信心候选 {result.metadata.get('recommended_count', 0)} 只。"
    )
    st.rerun()


def display_table(frame: pd.DataFrame, key: str) -> None:
    mappings = {
        "opportunity_rank": "排名", "code": "代码", "name": "名称",
        "opportunity_score": "综合机会分", "heat_score": "市场热度",
        "liquidity_score": "流动性", "strategy_score": "7 策略分",
        "strategy_confidence": "策略信心", "buy_votes": "买入票", "sell_votes": "卖出票",
        "trend_quality": "趋势质量", "research_probability": "模型上涨概率",
        "research_quality": "样本外证据质量", "research_status": "研究状态",
        "selection_path": "入选路径", "change_pct": "当日涨跌幅",
        "turnover": "成交额", "discovery_source": "发现来源", "latest_close": "最新收盘",
        "stop_loss": "策略止损", "take_profit": "策略止盈",
        "reward_risk": "现价盈亏比", "stop_distance_pct": "止损距离",
        "evaluation_status": "评分状态", "rejection_reasons": "未入选原因",
    }
    table = pd.DataFrame(index=frame.index)
    for source, label in mappings.items():
        if source in frame:
            table[label] = frame[source]
    st.dataframe(
        table,
        hide_index=True,
        key=key,
        height=min(650, 42 + max(1, len(table)) * 35),
        column_config={
            "排名": st.column_config.NumberColumn(format="%d", pinned=True),
            "代码": st.column_config.TextColumn(pinned=True),
            "名称": st.column_config.TextColumn(pinned=True),
            "综合机会分": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "市场热度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "流动性": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "趋势质量": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "样本外证据质量": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "7 策略分": st.column_config.NumberColumn(format="%+.1f / 10"),
            "策略信心": st.column_config.NumberColumn(format="percent"),
            "模型上涨概率": st.column_config.NumberColumn(format="percent"),
            "当日涨跌幅": st.column_config.NumberColumn(format="%+.2f%%"),
            "成交额": st.column_config.NumberColumn(format="HKD %.0f"),
            "最新收盘": st.column_config.NumberColumn(format="%.3f"),
            "策略止损": st.column_config.NumberColumn(format="%.3f"),
            "策略止盈": st.column_config.NumberColumn(format="%.3f"),
            "现价盈亏比": st.column_config.NumberColumn(format="%.2fR"),
            "止损距离": st.column_config.NumberColumn(format="percent"),
        },
    )


st.title("每日强信心候选")
st.caption("从全市场热度榜、成交额前列、平台预设股和用户自选股中发现候选，再执行 7 策略与防泄漏样本外验证。")
flash = st.session_state.pop("hk_opportunity_flash", None)
if flash:
    st.success(flash)
dates = SCAN_STORE.dates()
with st.container(horizontal=True, vertical_alignment="bottom"):
    selected_date = st.selectbox(
        "扫描快照日期", dates or ["暂无快照"], disabled=not dates, key="hk_opportunity_date"
    )
    if st.button(
        "重新批量跑分", icon=":material/model_training:", type="primary", disabled=not dates
    ):
        run_scoring(selected_date)
if not dates:
    st.info("请先到“港股全市场扫描”建立快照。")
    st.stop()
result = load_opportunities(selected_date)
if result is None:
    st.info("该快照尚未跑策略评分。点击“重新批量跑分”生成候选名单。")
    st.stop()

metadata = result.metadata
with st.container(horizontal=True):
    st.metric("多入口候选池", f"{metadata.get('candidate_count', 0)} 只", border=True)
    st.metric("完成 7 策略", f"{metadata.get('evaluated_count', 0)} 只", border=True)
    st.metric("强信心候选", f"{metadata.get('recommended_count', 0)} 只", border=True)
    st.metric("数据交易日", str(metadata.get("data_date", "—")), border=True)
st.caption(
    f"评分时间 {metadata.get('scored_at', '—')} · 综合机会分 = 7 策略 60% + 热度 15% + "
    f"策略信心 10% + 趋势质量 15% · 研究成本假设 {metadata.get('round_trip_cost_bps', '—')} bp/往返"
)

recommended = result.recommended
st.subheader("当日强信心买入候选")
if recommended.empty:
    st.info("当前没有股票同时通过 7 策略强买、流动性、趋势、止损止盈与样本外证据。系统不会为了凑数推荐。")
    completed = result.evaluated[
        result.evaluated.get("evaluation_status", pd.Series(index=result.evaluated.index, dtype=str)).eq("已完成")
    ].head(8)
    if not completed.empty:
        st.caption("以下是综合排序靠前的观察股，均未通过全部门槛。")
        display_table(completed, "hk_nearest_opportunities")
else:
    display_table(recommended, "hk_recommended_opportunities")
    selected_code = st.selectbox("选择候选股进一步复核", recommended["code"].astype(str).tolist())
    st.link_button(
        "在个股策略工作台打开",
        f"/?ticker={quote(selected_code)}",
        icon=":material/open_in_new:",
        type="primary",
    )
    st.download_button(
        "导出强信心候选",
        data=recommended.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"港股强信心候选_{selected_date}.csv",
        mime="text/csv",
        icon=":material/download:",
    )
with st.expander(f"查看全部 {len(result.evaluated)} 只评分与未入选原因"):
    display_table(result.evaluated, "hk_all_scored_opportunities")
with st.expander("港股强信心门槛"):
    st.markdown(
        "候选需满足：综合机会分 ≥ 68、7 策略为强买且评分 ≥ +5、买入票 ≥ 3、"
        "卖出票 ≤ 1、信心 ≥ 43%、趋势质量 ≥ 70、热度 ≥ 55、流动性 ≥ 45、"
        "日成交额 ≥ 500 万港元、止损止盈有效、现价盈亏比 ≥ 1.30R、"
        "止损距离 ≤ 12%，并通过扩展窗口样本外验证。"
    )
st.warning(
    "“强信心”只表示通过当前规则，不代表收益保证或个性化投资建议。下单前仍需复核公告、"
    "买卖单位、价差、盘中流动性和富途订单状态。",
    icon=":material/warning:",
)
