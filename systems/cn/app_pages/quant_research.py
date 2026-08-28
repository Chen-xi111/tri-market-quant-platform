"""每日候选的防泄漏样本外量化研究面板。"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from fund_flow import OpportunitySnapshot, OpportunityStore, SnapshotStore


FLOW_STORE = SnapshotStore()


@st.cache_data(ttl=30, max_entries=20, show_spinner=False)
def load_research(snapshot_date: str) -> OpportunitySnapshot | None:
    return OpportunityStore().load(snapshot_date)


def research_table(frame: pd.DataFrame, key: str) -> None:
    columns = {
        "opportunity_rank": "排名",
        "code": "代码",
        "name": "名称",
        "research_status": "模型历史验证",
        "research_signal": "当前信号",
        "research_probability": "当前上涨概率",
        "research_quality": "证据质量",
        "oos_auc": "样本外AUC",
        "brier_skill": "概率技能分",
        "oos_observations": "样本外观测",
        "fold_count": "验证窗口",
        "signal_count": "历史高概率样本",
        "signal_hit_rate": "历史命中率",
        "signal_avg_net_return": "成本后平均收益",
        "return_ci_low": "收益区间下界",
        "return_ci_high": "收益区间上界",
        "positive_fold_ratio": "正收益窗口占比",
        "research_drivers": "当前主要驱动",
        "research_reasons": "验证结论",
    }
    table = pd.DataFrame(index=frame.index)
    for source, label in columns.items():
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
            "当前上涨概率": st.column_config.NumberColumn(format="percent"),
            "证据质量": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f"
            ),
            "样本外AUC": st.column_config.NumberColumn(format="%.3f"),
            "概率技能分": st.column_config.NumberColumn(format="%.3f"),
            "历史命中率": st.column_config.NumberColumn(format="percent"),
            "成本后平均收益": st.column_config.NumberColumn(format="percent"),
            "收益区间下界": st.column_config.NumberColumn(format="percent"),
            "收益区间上界": st.column_config.NumberColumn(format="percent"),
            "正收益窗口占比": st.column_config.NumberColumn(format="percent"),
        },
    )


st.title("量化研究验证")
st.caption(
    "用扩展窗口样本外验证回答“历史上是否真的有预测增益”，"
    "而不是给 7 个策略再加一张主观权重票。"
)

dates = OpportunityStore().dates()
selected_date = st.selectbox(
    "研究快照日期",
    dates or ["暂无研究结果"],
    disabled=not dates,
    key="quant_research_snapshot_date",
)
if not dates:
    st.info("请先在“每日强信心候选”运行批量评分。")
    st.stop()

result = load_research(selected_date)
if result is None or result.evaluated.empty:
    st.info("该日期还没有可展示的量化研究结果。")
    st.stop()

metadata = result.metadata
research = metadata.get("quant_research", {})
with st.container(horizontal=True):
    st.metric("已研究候选", f"{research.get('eligible_count', 0)} 只", border=True)
    st.metric("历史验证通过", f"{research.get('validated_count', 0)} 只", border=True)
    st.metric("当前概率也通过", f"{research.get('current_gate_count', 0)} 只", border=True)
    st.metric("资金流历史快照", f"{len(FLOW_STORE.dates())} 日", border=True)

st.success(
    "可行：多数沪深股票约有 640 根日线，足以进行带 10 日隔离期的扩展窗口验证。"
)
if len(FLOW_STORE.dates()) < 60:
    st.warning(
        "资金流机器学习暂缓：当前历史快照少于 60 个交易日。"
        "现阶段资金流只作为当日背景和恶化否决因子，避免把今天的排名伪造为历史特征。"
    )

completed = result.evaluated[
    result.evaluated.get(
        "research_probability", pd.Series(index=result.evaluated.index, dtype=float)
    ).notna()
].copy()
if not completed.empty:
    chart = completed.nlargest(12, "research_probability").copy()
    chart["股票"] = chart["name"].astype(str) + " " + chart["code"].astype(str)
    chart["当前上涨概率"] = chart["research_probability"] * 100
    plot_frame = chart[
        ["股票", "当前上涨概率", "research_quality", "research_status"]
    ].copy()
    st.subheader("当前模型上涨概率排名")
    st.caption(
        f"{metadata.get('data_date', '—')} 收盘后状态 · 只显示概率最高的 12 只，"
        "是否推荐仍取决于全部样本外硬门槛。"
    )
    probability_chart = (
        alt.Chart(plot_frame)
        .mark_bar(color="#3B6FB6")
        .encode(
            y=alt.Y(
                "股票:N",
                sort=alt.SortField("当前上涨概率", order="descending"),
                title=None,
            ),
            x=alt.X(
                "当前上涨概率:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="当前上涨概率（%）",
            ),
            tooltip=[
                alt.Tooltip("股票:N", title="股票"),
                alt.Tooltip("当前上涨概率:Q", title="上涨概率", format=".2f"),
                alt.Tooltip("research_quality:Q", title="证据质量", format=".1f"),
                alt.Tooltip("research_status:N", title="历史验证"),
            ],
        )
        .properties(height=max(320, len(plot_frame) * 28))
    )
    st.altair_chart(probability_chart, width="stretch")

st.subheader("样本外验证明细")
research_table(result.evaluated, "quant_research_all")

available_codes = completed["code"].astype(str).tolist()
if available_codes:
    selected_code = st.selectbox(
        "查看单股概率校准",
        available_codes,
        format_func=lambda code: (
            f"{code}  "
            f"{completed.loc[completed['code'].astype(str) == code, 'name'].iloc[0]}"
        ),
        key="quant_research_calibration_code",
    )
    selected = completed[completed["code"].astype(str) == selected_code].iloc[0]
    try:
        points = json.loads(str(selected.get("research_calibration") or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        points = []
    if points:
        calibration = pd.DataFrame(points).sort_values("predicted_probability")
        calibration["模型预测"] = calibration["predicted_probability"] * 100
        calibration["实际命中"] = calibration["realized_hit_rate"] * 100
        calibration["理想校准线"] = calibration["模型预测"]
        st.line_chart(
            calibration,
            x="模型预测",
            y=["实际命中", "理想校准线"],
            color=["#3B6FB6", "#6B7280"],
            x_label="模型预测概率（%）",
            y_label="历史实际命中率（%）",
            height=330,
        )
        st.caption(
            f"每点为约 1/5 的样本外观测；当前主要驱动："
            f"{selected.get('research_drivers', '—')}。"
        )

with st.expander("研究方法与验收门槛"):
    st.markdown(
        f"- **模型**：{research.get('model', 'L2 正则逻辑回归')}；正则强度只在每个训练窗口内部选择。\n"
        f"- **特征**：{research.get('features', 'OHLCV 派生特征')}，全部只使用当日及以前数据。\n"
        f"- **目标**：{research.get('target', '未来 10 日成本后收益为正')}。\n"
        f"- **验证**：{research.get('validation', '扩展窗口样本外验证')}。\n"
        f"- **交易成本假设**：往返约 {research.get('round_trip_cost_bps', '—')} bp。\n"
        "- **历史通过**：样本外观测 ≥ 120、窗口 ≥ 4、AUC ≥ 0.52、Brier 概率误差优于基线、"
        "高概率信号成本后均值为正、至少一半窗口为正、证据质量 ≥ 50。\n"
        "- **当前通过**：历史通过后，当前上涨概率还需 ≥ 55%。\n"
        "- **决策用途**：量化主导路径可越过旧的前置机会分 70 门槛，"
        "但仍需策略买入触发、趋势质量和公共风控。"
    )

st.warning(
    "这是研究与候选过滤工具，不是收益保证。单股历史模型仍可能受制度变化、"
    "停复牌、涨跌停、样本选择和行情源复权质量影响。",
    icon=":material/warning:",
)
