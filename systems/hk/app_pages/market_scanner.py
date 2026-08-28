"""港股全市场热度扫描页面。"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from market_scan import MarketScanPipeline, MarketScanUnavailable, OpportunityScanner, ScanSnapshot, SnapshotStore
from market_scan.scheduler import read_scheduler_status


STORE = SnapshotStore()


@st.cache_data(ttl=30, max_entries=20, show_spinner=False)
def load_snapshot(snapshot_date: str) -> ScanSnapshot | None:
    return SnapshotStore().load(snapshot_date)


def format_hkd(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    amount = float(value)
    if abs(amount) >= 100_000_000:
        return f"HKD {amount / 100_000_000:,.1f} 亿"
    if abs(amount) >= 10_000:
        return f"HKD {amount / 10_000:,.0f} 万"
    return f"HKD {amount:,.0f}"


def run_manual_scan() -> None:
    progress_bar = st.progress(0, text="准备港股股票池…")

    def update(label: str, completed: int, total: int) -> None:
        progress_bar.progress(completed / max(total, 1), text=f"{label} · {completed}/{total}")

    try:
        snapshot = MarketScanPipeline(store=STORE).run(progress=update)
    except MarketScanUnavailable as exc:
        progress_bar.empty()
        st.error(f"本次批量行情不可用，已保留原快照。{exc}")
        return
    except Exception as exc:
        progress_bar.empty()
        st.error(f"扫描未完成，已保留原快照：{type(exc).__name__}: {exc}")
        return
    note = ""
    try:
        result = OpportunityScanner().run(snapshot, progress=update)
        note = f"强信心候选 {result.metadata.get('recommended_count', 0)} 只。"
    except Exception as exc:
        note = f"候选评分未完成：{type(exc).__name__}: {exc}"
    progress_bar.empty()
    load_snapshot.clear()
    st.session_state["hk_scan_flash"] = (
        f"已保存 {snapshot.snapshot_date} 快照；数据交易日 {snapshot.metadata.get('data_date')}。{note}"
    )
    st.rerun()


st.title("港股全市场扫描")
st.caption("批量扫描 00001–09999 普通股，用成交额、动量、换手与收盘位置发现值得进一步研究的候选。")

flash = st.session_state.pop("hk_scan_flash", None)
if flash:
    st.success(flash)

dates = STORE.dates()
with st.container(horizontal=True, vertical_alignment="bottom"):
    selected_date = st.selectbox(
        "快照日期", dates or ["暂无快照"], disabled=not dates, key="hk_scan_snapshot_date"
    )
    if st.button("立即扫描", icon=":material/refresh:", type="primary"):
        run_manual_scan()

status = read_scheduler_status(STORE)
st.caption(
    f"自动任务：工作日 16:20（服务运行期间） · {status.get('message', '')} {status.get('updated_at', '')}"
)
if not dates:
    st.info("尚无港股扫描快照。首次扫描会自动发现股票池，通常比后续扫描稍久。")
    st.stop()

snapshot = load_snapshot(selected_date)
if snapshot is None:
    st.error("快照无法读取，请重新扫描。")
    st.stop()

metadata = snapshot.metadata
st.caption(
    f"数据日 {metadata.get('data_date')} · 最新报价 {metadata.get('latest_quote_at')} · "
    f"扫描时间 {metadata.get('scanned_at')} · {metadata.get('provider')} · "
    f"有效普通股 {metadata.get('universe_count', 0)} 只"
)
if snapshot.errors:
    with st.expander(f"有 {len(snapshot.errors)} 个批次请求未成功"):
        st.code("\n".join(snapshot.errors), language=None)

stocks = snapshot.stocks.copy()
market = snapshot.market.iloc[-1] if not snapshot.market.empty else pd.Series(dtype=float)
with st.container(horizontal=True):
    st.metric("扫描普通股", f"{len(stocks)} 只", border=True)
    st.metric("市场成交额", format_hkd(market.get("turnover")), border=True)
    st.metric("上涨 / 下跌", f"{int(market.get('rising', 0))} / {int(market.get('falling', 0))}", border=True)
    st.metric("上涨家数占比", f"{float(market.get('breadth', 0)):.1f}%", border=True)
    st.metric("热度 ≥ 75", f"{int(market.get('hot_count', 0))} 只", border=True)

with st.container(border=True):
    st.subheader("热度与成交额分布")
    chart_frame = stocks.nlargest(40, "turnover").copy()
    chart_frame["成交额（亿港元）"] = chart_frame["turnover"] / 100_000_000
    chart = (
        alt.Chart(chart_frame)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("change_pct:Q", title="当日涨跌幅（%）"),
            y=alt.Y("成交额（亿港元）:Q", title="成交额（亿港元）"),
            size=alt.Size("heat_score:Q", title="热度"),
            color=alt.Color("market_state:N", title="状态"),
            tooltip=["code:N", "name:N", "change_pct:Q", "成交额（亿港元）:Q", "heat_score:Q"],
        )
        .properties(height=420)
    )
    st.altair_chart(chart)

with st.form("hk_market_filters", border=False):
    query = st.text_input("按代码或名称筛选", placeholder="例如 01164 或 中广核矿业")
    minimum_heat = st.slider("最低热度", 0, 100, 40, 5)
    minimum_turnover = st.number_input("最低成交额（万港元）", min_value=0, value=500, step=100)
    submitted = st.form_submit_button("应用筛选", icon=":material/filter_alt:")
if submitted or query or minimum_heat or minimum_turnover:
    query = query.strip()
    if query:
        stocks = stocks[
            stocks["code"].astype(str).str.contains(query, case=False, na=False)
            | stocks["name"].astype(str).str.contains(query, case=False, na=False)
        ]
    stocks = stocks[
        (stocks["heat_score"].fillna(0) >= minimum_heat)
        & (stocks["turnover"].fillna(0) >= minimum_turnover * 10_000)
    ]

table = stocks.rename(
    columns={
        "rank": "排名", "code": "代码", "name": "名称", "price": "最新价",
        "change_pct": "涨跌幅", "turnover": "成交额", "turnover_rate": "换手指标",
        "heat_score": "市场热度", "liquidity_score": "流动性", "momentum_score": "动量",
        "close_location": "收盘位置", "market_state": "市场状态", "quote_time": "报价时间",
    }
)
columns = [
    "排名", "代码", "名称", "最新价", "涨跌幅", "成交额", "市场热度", "流动性",
    "动量", "收盘位置", "市场状态", "报价时间",
]
st.dataframe(
    table[[column for column in columns if column in table]],
    hide_index=True,
    height=600,
    key="hk_market_scan_rank",
    column_config={
        "排名": st.column_config.NumberColumn(format="%d", pinned=True),
        "代码": st.column_config.TextColumn(pinned=True),
        "名称": st.column_config.TextColumn(pinned=True),
        "最新价": st.column_config.NumberColumn(format="%.3f"),
        "涨跌幅": st.column_config.NumberColumn(format="%+.2f%%"),
        "成交额": st.column_config.NumberColumn(format="HKD %.0f"),
        "市场热度": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "流动性": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "动量": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
        "收盘位置": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
    },
)
st.download_button(
    "导出当前排名",
    data=table.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"港股全市场扫描_{snapshot.snapshot_date}.csv",
    mime="text/csv",
    icon=":material/download:",
)

with st.expander("评分口径与数据说明"):
    st.markdown(
        "市场热度 = 成交活跃度 45% + 当日动量 25% + 换手指标 15% + 收盘位置 15%，"
        "并对极低流动性、仙股价格和单日异常波动降分。腾讯报价用于股票发现与排序；"
        "候选仍要再经过历史 K 线、7 策略和样本外验证。"
    )
st.warning(
    "热度代表市场关注度，不代表资金净流入，也不是收益预测。港股没有 A 股统一涨跌停板，"
    "因此系统使用流动性与异常波动门槛控制追涨和冲击成本。",
    icon=":material/warning:",
)
