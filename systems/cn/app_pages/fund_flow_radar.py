"""全市场资金雷达页面。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fund_flow import (
    FundFlowPipeline,
    FundFlowUnavailable,
    OpportunityScanner,
    ScanSnapshot,
    SnapshotStore,
)
from fund_flow.scheduler import read_scheduler_status


STORE = SnapshotStore()


@st.cache_data(ttl=30, max_entries=20, show_spinner=False)
def load_snapshot(snapshot_date: str) -> ScanSnapshot | None:
    return SnapshotStore().load(snapshot_date)


def format_amount(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    amount = float(value)
    if abs(amount) >= 100_000_000:
        return f"{amount / 100_000_000:+,.1f} 亿"
    if abs(amount) >= 10_000:
        return f"{amount / 10_000:+,.0f} 万"
    return f"{amount:+,.0f} 元"


def coverage_label(snapshot: ScanSnapshot) -> str:
    coverage = snapshot.metadata.get("coverage", {})
    labels = []
    if coverage.get("market"):
        labels.append("大盘")
    for key, name in (("industry", "行业"), ("concept", "概念"), ("stock", "个股")):
        horizons = coverage.get(key) or []
        if horizons:
            labels.append(f"{name} {len(horizons)}/3")
    return " · ".join(labels) or "无可用分区"


def run_manual_scan() -> None:
    progress_bar = st.progress(0, text="准备连接资金流接口…")

    def update(label: str, completed: int, total: int) -> None:
        progress_bar.progress(
            completed / max(total, 1),
            text=f"{label} · {completed}/{total}",
        )

    try:
        snapshot = FundFlowPipeline(store=STORE).run(progress=update)
    except FundFlowUnavailable as exc:
        progress_bar.empty()
        st.error(f"本次所有接口均不可用，已保留原快照。{exc}")
        return
    except Exception as exc:
        progress_bar.empty()
        st.error(f"扫描未完成，已保留原快照：{type(exc).__name__}: {exc}")
        return
    opportunity_note = ""
    try:
        opportunities = OpportunityScanner().run(snapshot, progress=update)
        opportunity_note = (
            f"强信心候选 "
            f"{opportunities.metadata.get('recommended_count', 0)} 只。"
        )
    except Exception as exc:
        opportunity_note = f"批量策略评分未完成：{type(exc).__name__}: {exc}"
    progress_bar.empty()
    load_snapshot.clear()
    st.session_state["fund_flow_flash"] = (
        f"已保存 {snapshot.snapshot_date} 快照；"
        f"数据交易日 {snapshot.metadata.get('data_date')}。{opportunity_note}"
    )
    st.rerun()


def market_metrics(snapshot: ScanSnapshot) -> None:
    market = snapshot.market
    latest = market.iloc[-1] if not market.empty else pd.Series(dtype="float64")
    available_days = min(5, len(market))
    five_day = market.tail(available_days)["main_net"].sum() if "main_net" in market else float("nan")
    cumulative_label = "近 5 日累计" if available_days == 5 else f"可用 {available_days} 日累计"
    industry_in = (
        int(snapshot.industries.get("main_net_today", pd.Series(dtype=float)).gt(0).sum())
        if not snapshot.industries.empty
        else 0
    )
    stock_in = (
        int(snapshot.stocks.get("main_net_today", pd.Series(dtype=float)).gt(0).sum())
        if not snapshot.stocks.empty
        else 0
    )
    with st.container(horizontal=True):
        st.metric(
            "全市场主力净流入",
            format_amount(latest.get("main_net")),
            border=True,
        )
        st.metric(cumulative_label, format_amount(five_day), border=True)
        st.metric("当日净流入行业", f"{industry_in} 个", border=True)
        st.metric("当日净流入个股", f"{stock_in} 只", border=True)


def render_market(snapshot: ScanSnapshot) -> None:
    market = snapshot.market.copy()
    if market.empty:
        st.warning("本次大盘资金接口不可用；板块和个股分区仍可独立查看。")
        return
    market["日期"] = pd.to_datetime(market["date"], errors="coerce")
    series_columns = {
        "main_net": "主力净流入（亿）",
        "ultra_net": "超大单净流入（亿）",
        "big_net": "大单净流入（亿）",
        "small_net": "小单净流入（亿）",
    }
    chart = pd.DataFrame({"日期": market["日期"]})
    for source, label in series_columns.items():
        if source in market:
            chart[label] = market[source] / 100_000_000
    with st.container(border=True):
        st.subheader("资金进出趋势")
        st.line_chart(chart, x="日期", y=list(chart.columns[1:]), y_label="亿元")
        st.caption("正值代表净流入，负值代表净流出；不同单量存在统计口径差异。")

    table = pd.DataFrame({"日期": market["date"]})
    table["上证涨跌幅"] = market.get("sh_change_pct")
    table["深证涨跌幅"] = market.get("sz_change_pct")
    table["主力净流入"] = market.get("main_net")
    table["主力净占比"] = market.get("main_ratio")
    st.dataframe(
        table.tail(30).iloc[::-1],
        hide_index=True,
        key="market_flow_history",
        column_config={
            "上证涨跌幅": st.column_config.NumberColumn(format="%.2f%%"),
            "深证涨跌幅": st.column_config.NumberColumn(format="%.2f%%"),
            "主力净流入": st.column_config.NumberColumn(format="%.0f 元"),
            "主力净占比": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )


def ranking_table(frame: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    table = pd.DataFrame(index=frame.index)
    table["排名"] = frame.get("rank")
    if entity_type == "stock":
        table["代码"] = frame.get("code")
    table["名称"] = frame.get("name")
    table["资金强度"] = frame.get("flow_score")
    table["资金状态"] = frame.get("capital_signal")
    table["价格热度"] = frame.get("price_heat")
    mappings = {
        "change_pct_today": "今日涨跌幅",
        "main_net_today": "今日主力净流入",
        "main_ratio_today": "今日主力净占比",
        "main_net_d5": "5 日主力净流入",
        "main_ratio_d5": "5 日主力净占比",
        "main_net_d10": "10 日主力净流入",
        "main_ratio_d10": "10 日主力净占比",
        "leader_today": "主力净流入最大股",
    }
    for source, label in mappings.items():
        if source in frame:
            table[label] = frame[source]
    return table


def render_flow_bar(frame: pd.DataFrame) -> None:
    if "main_net_today" not in frame or frame["main_net_today"].dropna().empty:
        return
    valid = frame.dropna(subset=["main_net_today"])
    selected = pd.concat(
        [valid.nlargest(10, "main_net_today"), valid.nsmallest(10, "main_net_today")]
    ).drop_duplicates(subset=["name"])
    selected = selected.assign(
        **{
            "当日主力净流入（亿）": selected["main_net_today"] / 100_000_000,
            "方向": selected["main_net_today"].map(lambda value: "净流入" if value >= 0 else "净流出"),
        }
    )
    chart = (
        alt.Chart(selected)
        .mark_bar()
        .encode(
            x=alt.X("当日主力净流入（亿）:Q", title="亿元"),
            y=alt.Y("name:N", title=None, sort="-x"),
            color=alt.Color(
                "方向:N",
                scale=alt.Scale(domain=["净流入", "净流出"], range=["#d62728", "#2ca02c"]),
            ),
            tooltip=[
                alt.Tooltip("name:N", title="名称"),
                alt.Tooltip("当日主力净流入（亿）:Q", title="净流入（亿）", format=".2f"),
                alt.Tooltip("flow_score:Q", title="资金强度", format=".1f"),
            ],
        )
        .properties(height=460)
    )
    st.altair_chart(chart)


def render_ranking(snapshot: ScanSnapshot, view: str) -> None:
    if view == "行业":
        frame, entity_type = snapshot.industries.copy(), "sector"
    elif view == "概念":
        frame, entity_type = snapshot.concepts.copy(), "sector"
    else:
        frame, entity_type = snapshot.stocks.copy(), "stock"
    if frame.empty:
        st.warning(f"本次{view}资金接口不可用，请查看采集告警或保留的其他分区。")
        return

    with st.container(border=True):
        st.subheader(f"{view}资金强弱对照")
        render_flow_bar(frame)

    if entity_type == "stock":
        with st.form("stock_flow_filters", border=False):
            query = st.text_input("按代码或名称筛选", placeholder="例如 600010 或 包钢股份")
            minimum_score = st.slider("最低资金强度", 0, 100, 0, 5)
            submitted = st.form_submit_button("应用筛选", icon=":material/filter_alt:")
        if submitted or query or minimum_score:
            query = query.strip()
            if query:
                frame = frame[
                    frame["code"].astype(str).str.contains(query, case=False, na=False)
                    | frame["name"].astype(str).str.contains(query, case=False, na=False)
                ]
            frame = frame[frame["flow_score"].fillna(0) >= minimum_score]

    table = ranking_table(frame, entity_type)
    st.dataframe(
        table,
        hide_index=True,
        height=560,
        key=f"{view}_fund_flow_rank",
        column_config={
            "排名": st.column_config.NumberColumn(format="%d", pinned=True),
            "代码": st.column_config.TextColumn(pinned=True),
            "名称": st.column_config.TextColumn(pinned=True),
            "资金强度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "今日涨跌幅": st.column_config.NumberColumn(format="%.2f%%"),
            "今日主力净流入": st.column_config.NumberColumn(format="%.0f 元"),
            "今日主力净占比": st.column_config.NumberColumn(format="%.2f%%"),
            "5 日主力净流入": st.column_config.NumberColumn(format="%.0f 元"),
            "5 日主力净占比": st.column_config.NumberColumn(format="%.2f%%"),
            "10 日主力净流入": st.column_config.NumberColumn(format="%.0f 元"),
            "10 日主力净占比": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    st.download_button(
        "导出当前排名",
        data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"A股{view}资金流_{snapshot.snapshot_date}.csv",
        mime="text/csv",
        icon=":material/download:",
    )


st.title("全市场资金雷达")
st.caption("每天收盘后扫描大盘、行业、概念与全市场个股资金流，并保存可回看的日快照。")

flash = st.session_state.pop("fund_flow_flash", None)
if flash:
    st.success(flash)

dates = STORE.dates()
with st.container(horizontal=True, vertical_alignment="bottom"):
    selected_date = st.selectbox(
        "快照日期",
        dates or ["暂无快照"],
        disabled=not dates,
        key="fund_flow_snapshot_date",
    )
    if st.button("立即扫描", icon=":material/refresh:", type="primary"):
        run_manual_scan()

scheduler_status = read_scheduler_status(STORE)
status_text = scheduler_status.get("message", "")
status_time = scheduler_status.get("updated_at", "")
st.caption(f"自动任务：每个工作日 15:20（服务运行期间） · {status_text} {status_time}")

if not dates:
    st.info("尚无资金流快照。点击“立即扫描”建立第一份快照；之后服务会在工作日收盘后自动更新。")
    st.stop()

snapshot = load_snapshot(selected_date)
if snapshot is None:
    st.error("快照文件无法读取，请重新扫描。")
    st.stop()

st.caption(
    f"数据交易日 {snapshot.metadata.get('data_date')} · "
    f"扫描时间 {snapshot.metadata.get('scanned_at')} · "
    f"数据源 {snapshot.metadata.get('provider')} {snapshot.metadata.get('provider_version')} · "
    f"覆盖 {coverage_label(snapshot)}"
)
if snapshot.metadata.get("market_derived_from_stocks"):
    st.info(
        "大盘历史接口本次不可用；页面中的全市场主力净流入由当日个股资金净额汇总得到，"
        "指数涨跌幅会暂时留空。"
    )
if snapshot.errors:
    with st.expander(f"本次有 {len(snapshot.errors)} 项免费接口未成功"):
        st.code("\n".join(snapshot.errors), language=None)
        st.caption("局部失败不会清空其他已成功分区；下次扫描会再次尝试。")

market_metrics(snapshot)
view = st.segmented_control(
    "分析分区",
    ["大盘", "行业", "概念", "个股"],
    default="大盘",
    selection_mode="single",
    key="fund_flow_view",
)
if view == "大盘":
    render_market(snapshot)
else:
    render_ranking(snapshot, view)

with st.expander("评分口径与开源来源"):
    st.markdown(
        "资金强度是同一分区内今日、5 日、10 日主力净流入金额与净占比的加权百分位，"
        "仅用于比较资金持续性，不是涨幅预测。数据接口来自 "
        "[AKShare](https://github.com/akfamily/akshare)，定时运行模式参考 "
        "[bigApush](https://github.com/wei1104/bigApush)；两者均采用 MIT 许可证。"
    )

st.warning(
    "资金流是成交口径下的估算结果，可能受接口延迟、口径变化和短期交易噪声影响；"
    "请结合趋势、估值、流动性与风控验证，不应把排名直接视为买入建议。",
    icon=":material/warning:",
)
