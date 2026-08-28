"""资金热点、策略触发与样本外量化证据融合的每日候选。"""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import streamlit as st

from fund_flow import OpportunityScanner, OpportunitySnapshot, OpportunityStore, SnapshotStore


FLOW_STORE = SnapshotStore()
OPPORTUNITY_STORE = OpportunityStore(FLOW_STORE.data_dir)


@st.cache_data(ttl=30, max_entries=20, show_spinner=False)
def load_opportunities(snapshot_date: str) -> OpportunitySnapshot | None:
    return OpportunityStore().load(snapshot_date)


def run_scoring(snapshot_date: str) -> None:
    snapshot = FLOW_STORE.load(snapshot_date)
    if snapshot is None:
        st.error("找不到对应的资金快照，请先在“全市场资金雷达”扫描。")
        return
    progress_bar = st.progress(0, text="正在组建热门股候选池…")

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
    st.session_state["opportunity_flash"] = (
        f"已评估 {result.metadata.get('candidate_count', 0)} 只，"
        f"强信心候选 {result.metadata.get('recommended_count', 0)} 只。"
    )
    st.rerun()


def display_table(frame: pd.DataFrame, key: str) -> None:
    table = pd.DataFrame(index=frame.index)
    mappings = {
        "opportunity_rank": "排名",
        "code": "代码",
        "name": "名称",
        "opportunity_score": "前置机会分",
        "research_probability": "模型上涨概率",
        "research_quality": "样本外证据质量",
        "research_status": "样本外状态",
        "research_signal": "当前研究信号",
        "oos_auc": "样本外AUC",
        "brier_skill": "概率技能分",
        "signal_count": "历史高概率样本",
        "signal_hit_rate": "历史命中率",
        "signal_avg_net_return": "历史成本后均值",
        "flow_score": "资金强度",
        "strategy_score": "原策略分",
        "strategy_confidence": "策略信心",
        "buy_votes": "买入票",
        "sell_votes": "卖出票",
        "trend_quality": "趋势质量",
        "selection_path": "入选路径",
        "change_pct_today": "当日涨跌幅",
        "capital_signal": "资金状态",
        "discovery_source": "入选来源",
        "latest_close": "最新收盘",
        "stop_loss": "策略止损",
        "take_profit": "策略止盈",
        "evaluation_status": "评分状态",
        "rejection_reasons": "未入选原因",
    }
    for source, label in mappings.items():
        if source in frame:
            table[label] = frame[source]
    st.dataframe(
        table,
        hide_index=True,
        key=key,
        height=min(620, 42 + max(1, len(table)) * 35),
        column_config={
            "排名": st.column_config.NumberColumn(format="%d", pinned=True),
            "代码": st.column_config.TextColumn(pinned=True),
            "名称": st.column_config.TextColumn(pinned=True),
            "前置机会分": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
            "模型上涨概率": st.column_config.NumberColumn(format="percent"),
            "样本外证据质量": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
            "样本外AUC": st.column_config.NumberColumn(format="%.3f"),
            "概率技能分": st.column_config.NumberColumn(format="%.3f"),
            "历史命中率": st.column_config.NumberColumn(format="percent"),
            "历史成本后均值": st.column_config.NumberColumn(format="percent"),
            "资金强度": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
            "趋势质量": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.0f"
            ),
            "原策略分": st.column_config.NumberColumn(format="%+.1f / 10"),
            "策略信心": st.column_config.NumberColumn(format="percent"),
            "当日涨跌幅": st.column_config.NumberColumn(format="%.2f%%"),
            "最新收盘": st.column_config.NumberColumn(format="%.3f"),
            "策略止损": st.column_config.NumberColumn(format="%.3f"),
            "策略止盈": st.column_config.NumberColumn(format="%.3f"),
        },
    )


st.title("每日强信心候选")
st.caption(
    "同时扫描全市资金榜、热门板块领涨股、平台预设股和用户自选股；"
    "7 策略只负责触发候选，样本外量化研究是最终推荐的硬门槛。"
)

flash = st.session_state.pop("opportunity_flash", None)
if flash:
    st.success(flash)

dates = FLOW_STORE.dates()
with st.container(horizontal=True, vertical_alignment="bottom"):
    selected_date = st.selectbox(
        "资金快照日期",
        dates or ["暂无快照"],
        disabled=not dates,
        key="opportunity_snapshot_date",
    )
    if st.button(
        "重新批量跑分",
        icon=":material/model_training:",
        type="primary",
        disabled=not dates,
    ):
        run_scoring(selected_date)

if not dates:
    st.info("请先到“全市场资金雷达”建立每日快照。")
    st.stop()

result = load_opportunities(selected_date)
if result is None:
    st.info("该资金快照尚未跑原策略评分。点击“重新批量跑分”生成候选名单。")
    st.stop()

metadata = result.metadata
research_metadata = metadata.get("quant_research", {})
with st.container(horizontal=True):
    st.metric("多入口候选池", f"{metadata.get('candidate_count', 0)} 只", border=True)
    st.metric(
        "可做样本外研究",
        f"{research_metadata.get('eligible_count', 0)} 只",
        border=True,
    )
    st.metric(
        "模型历史验证通过",
        f"{research_metadata.get('validated_count', 0)} 只",
        border=True,
    )
    st.metric("强信心候选", f"{metadata.get('recommended_count', 0)} 只", border=True)

st.caption(
    f"资金数据日 {metadata.get('data_date', '—')} · "
    f"评分时间 {metadata.get('scored_at', '—')} · "
    "前置机会分只用于发现形态；最终还必须通过隔离期样本外验证、成本后收益和当前模型概率。"
)

recommended = result.recommended
st.subheader("当日强信心买入候选")
if recommended.empty:
    st.info(
        "当日没有股票同时通过策略触发、样本外预测增益、成本后历史收益、"
        "当前上涨概率与公共风控。系统不会为了凑数强行推荐。"
    )
    completed = result.evaluated[
        result.evaluated.get(
            "evaluation_status", pd.Series(index=result.evaluated.index, dtype=str)
        ).eq("已完成")
    ].head(5)
    if not completed.empty:
        st.caption("以下是距离门槛最近的观察股，均不属于当日强信心买入候选。")
        display_table(completed, "nearest_opportunities")
else:
    display_table(recommended, "recommended_opportunities")
    selected_code = st.selectbox(
        "选择候选股进一步复核",
        recommended["code"].astype(str).tolist(),
        format_func=lambda code: (
            f"{code}  "
            f"{recommended.loc[recommended['code'].astype(str) == code, 'name'].iloc[0]}"
        ),
    )
    st.link_button(
        "在原个股策略工作台打开",
        f"/?ticker={quote(selected_code)}",
        icon=":material/open_in_new:",
        type="primary",
    )
    st.download_button(
        "导出强信心候选",
        data=recommended.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"A股强信心候选_{selected_date}.csv",
        mime="text/csv",
        icon=":material/download:",
    )

with st.expander(f"查看全部 {len(result.evaluated)} 只评分与未入选原因"):
    display_table(result.evaluated, "all_scored_opportunities")
    if result.errors:
        st.caption(f"其中 {len(result.errors)} 只因行情数据不足未跑完策略。")

with st.expander("强信心门槛"):
    st.markdown(
        "先通过量化研究硬门槛：至少 120 个样本外观测和 4 个扩展窗口，"
        "AUC ≥ 0.52、概率误差优于历史胜率基线、高概率信号成本后平均收益为正、"
        "至少一半验证窗口为正，且当前模型上涨概率 ≥ 55%。再通过公共风控："
        "前置机会分 ≥ 70、止损/止盈仍有效、K 线新鲜、"
        "非 ST/退市且未接近各自板块涨停。然后满足任一路径：\n\n"
        "- **资金·策略共振**：原策略 ≥ +5、买入票 ≥ 2、资金强度 ≥ 75，"
        "3 个周期至少 2 个流入且当日净流入。\n"
        "- **技术先行·资金未连续流出**：原策略 ≥ +5、买入票 ≥ 3、卖出票 ≤ 1、"
        "信心 ≥ 43%、趋势质量 ≥ 75，且资金不得连续流出。\n"
        "- **量化验证主导·资金未恶化**：不要求前置机会分 ≥ 70；策略只需形成买入触发 "
        "（≥ +3、买入票 ≥ 2、卖出票 ≤ 1），趋势质量 ≥ 75，且资金不得连续流出。"
    )

st.warning(
    "模型概率来自有限历史样本，只表示当前特征在本股票历史中的条件概率估计。"
    "“强信心”不代表收益保证或个性化投资建议。"
    "下单前仍需在原工作台复核趋势、回测、止损、仓位与交易状态。",
    icon=":material/warning:",
)
