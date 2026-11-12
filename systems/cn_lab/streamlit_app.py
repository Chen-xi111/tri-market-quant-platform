"""A 股隔日 / 当日决策策略实验平台。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parents[1]
CN_DIR = PROJECT_DIR / "systems" / "cn"
if str(CN_DIR) not in sys.path:
    sys.path.insert(0, str(CN_DIR))

import config as cfg  # noqa: E402
from data_sources import SOURCES  # noqa: E402
from market_data import fetch_history, fetch_quote, quote_from_history  # noqa: E402
from trading_rules import board_name, price_limit_bounds  # noqa: E402

from action_backtest import BacktestResult, evaluate_strategy  # noqa: E402
from action_strategy import PROFILES, build_action_plan, score_breakdown  # noqa: E402


st.set_page_config(
    page_title="A 股短周期策略实验室",
    page_icon=":material/experiment:",
    layout="wide",
    initial_sidebar_state="expanded",
)


LAB_STOCKS = {
    "600019.SS": "宝钢股份",
    "600519.SS": "贵州茅台",
    "300750.SZ": "宁德时代",
    **cfg.PRESET_STOCKS,
}
DEMO_FILES = {
    "600019.SS": "cn_600019_ss.csv",
    "600519.SS": "cn_600519_ss.csv",
    "300750.SZ": "cn_300750_sz.csv",
}


def _format_stock(value: str) -> str:
    try:
        yahoo_code, _, raw_code = cfg.normalize_ticker(value)
    except ValueError:
        return str(value)
    return f"{raw_code}  {LAB_STOCKS.get(yahoo_code, '自定义')}"


def _demo_history(ticker: str) -> pd.DataFrame | None:
    filename = DEMO_FILES.get(ticker)
    if filename is None:
        return None
    path = PROJECT_DIR / "demo_data" / filename
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


@st.cache_data(ttl="5m", max_entries=32, show_spinner=False, refresh_mode="background")
def load_daily_history(source_key: str, ticker: str, days: int):
    offline = os.environ.get("CN_LAB_OFFLINE") == "1"
    if offline:
        demo = _demo_history(ticker)
        return demo, "demo", ("离线验收模式：使用项目内历史样例",), True
    frame, actual_source, failures = fetch_history(source_key, ticker, days)
    if frame is not None and not frame.empty:
        return frame, actual_source, failures, False
    demo = _demo_history(ticker)
    if demo is not None:
        return demo, "demo", (*failures, "在线源不可用，已使用项目内历史样例"), True
    return frame, actual_source, failures, False


@st.cache_data(ttl=20, max_entries=64, show_spinner=False, refresh_mode="background")
def load_latest_quote(source_key: str, ticker: str):
    if os.environ.get("CN_LAB_OFFLINE") == "1":
        return None, None, ("离线验收模式未请求实时报价",)
    return fetch_quote(
        source_key,
        ticker,
        allow_slow_fallback=(source_key != "eastmoney"),
    )


def _direction_color(level: str) -> str:
    return {
        "buy": "green",
        "sell": "red",
        "reduce": "orange",
        "warning": "orange",
        "hold": "gray",
    }.get(level, "gray")


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _metric_cards(result: BacktestResult) -> None:
    metrics = result.metrics
    low, high = metrics["win_rate_ci"]
    with st.container(horizontal=True):
        st.metric(
            "时间外历史胜率",
            _percent(metrics["win_rate"]),
            border=True,
            help=f"95% Wilson 区间 {_percent(low)}–{_percent(high)}；不是未来胜率保证。",
        )
        st.metric("时间外交易数", f"{metrics['total_trades']} 笔", border=True)
        profit_factor = metrics["profit_factor"]
        st.metric(
            "盈亏因子",
            "∞" if profit_factor == float("inf") else f"{profit_factor:.2f}",
            border=True,
        )
        st.metric("时间外收益", _percent(metrics["total_return"]), border=True)
        st.metric("最大回撤", _percent(metrics["max_drawdown"]), border=True)
    st.caption(
        f"验证状态：{metrics['status']}。胜率区间 {_percent(low)}–{_percent(high)}；"
        "样本少于 12 笔时只展示、不得据此实盘。"
    )


def _render_plan(plan, *, holding_quantity: int, sellable_quantity: int) -> None:
    st.subheader("今日执行清单")
    st.badge(
        plan.action,
        color=_direction_color(plan.action_level),
        icon=":material/rule:",
    )
    st.write(plan.reason)

    with st.container(horizontal=True):
        st.metric("策略评分", f"{plan.score:.1f} / 100", border=True)
        st.metric(
            "计划买入区",
            f"¥{plan.entry_low:.2f}–{plan.entry_high:.2f}",
            border=True,
        )
        st.metric("止损", f"¥{plan.stop_loss:.2f}", border=True)
        st.metric("第一止盈", f"¥{plan.target_1:.2f}", border=True)
        st.metric("第二止盈", f"¥{plan.target_2:.2f}", border=True)

    execution_rows = [
        {
            "顺序": 1,
            "动作": "买入确认",
            "价格/数量": f"¥{plan.entry_low:.2f}–{plan.entry_high:.2f}；最多 {plan.suggested_quantity} 股",
            "触发条件": f"评分达标且价格重新进入买入区；买入单位按 {plan.buy_lot_size} 股保守处理",
        },
        {
            "顺序": 2,
            "动作": "风险退出",
            "价格/数量": f"¥{plan.stop_loss:.2f}；全部可卖数量",
            "触发条件": "触及止损，或收盘趋势失效；今日买入部分受 T+1 限制",
        },
        {
            "顺序": 3,
            "动作": "第一止盈",
            "价格/数量": f"¥{plan.target_1:.2f}；约 50% 可卖仓位",
            "触发条件": f"达到第一目标，计划盈亏比约 {plan.risk_reward_1:.2f}R",
        },
        {
            "顺序": 4,
            "动作": "第二止盈",
            "价格/数量": f"¥{plan.target_2:.2f}；剩余可卖仓位",
            "触发条件": f"达到第二目标，计划盈亏比约 {plan.risk_reward_2:.2f}R",
        },
    ]
    st.dataframe(pd.DataFrame(execution_rows), hide_index=True)
    if holding_quantity > 0:
        st.info(
            f"录入持仓 {holding_quantity} 股，其中今日可卖 {min(holding_quantity, sellable_quantity)} 股；"
            f"当前计划建议卖出 {plan.sell_quantity} 股。"
        )
    else:
        st.warning(
            "A 股股票实行 T+1：今天买入的股票今天不能卖出；所谓“当日操作”不等于日内回转交易。"
        )
    st.caption(plan.invalidation)


def _render_chart(features: pd.DataFrame, result: BacktestResult, plan) -> None:
    visible = features.tail(180).reset_index()
    date_column = visible.columns[0]
    visible = visible.rename(columns={date_column: "日期", "close": "收盘价"})
    line = (
        alt.Chart(visible)
        .mark_line(color="#2563EB", strokeWidth=2)
        .encode(
            x=alt.X("日期:T", title="交易日"),
            y=alt.Y("收盘价:Q", title="价格（CNY）", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("日期:T"), alt.Tooltip("收盘价:Q", format=".2f")],
        )
    )
    trades = result.trades_frame()
    layers = [line]
    if not trades.empty:
        entries = trades[["entry_date", "entry_price"]].rename(
            columns={"entry_date": "日期", "entry_price": "价格"}
        )
        exits = trades[["exit_date", "exit_price", "exit_reason"]].rename(
            columns={"exit_date": "日期", "exit_price": "价格", "exit_reason": "原因"}
        )
        layers.append(
            alt.Chart(entries)
            .mark_point(shape="triangle-up", filled=True, size=90, color="#16A34A")
            .encode(x="日期:T", y="价格:Q", tooltip=["日期:T", alt.Tooltip("价格:Q", format=".2f")])
        )
        layers.append(
            alt.Chart(exits)
            .mark_point(shape="triangle-down", filled=True, size=90, color="#DC2626")
            .encode(
                x="日期:T",
                y="价格:Q",
                tooltip=["日期:T", alt.Tooltip("价格:Q", format=".2f"), "原因:N"],
            )
        )
    levels = pd.DataFrame(
        [
            {"位置": "止损", "价格": plan.stop_loss, "颜色": "#DC2626"},
            {"位置": "第一止盈", "价格": plan.target_1, "颜色": "#F59E0B"},
            {"位置": "第二止盈", "价格": plan.target_2, "颜色": "#16A34A"},
        ]
    )
    layers.append(
        alt.Chart(levels)
        .mark_rule(strokeDash=[5, 4])
        .encode(
            y="价格:Q",
            color=alt.Color(
                "位置:N",
                scale=alt.Scale(domain=levels["位置"].tolist(), range=levels["颜色"].tolist()),
            ),
            tooltip=["位置:N", alt.Tooltip("价格:Q", format=".2f")],
        )
    )
    st.altair_chart(alt.layer(*layers).properties(height=430).interactive())
    st.caption("绿色三角为历史实际模拟买点，红色三角为卖点；虚线为当前计划，不是历史信号。")


def _render_validation(full: BacktestResult, out_of_sample: BacktestResult) -> None:
    st.subheader("验证结论")
    _metric_cards(out_of_sample)
    if out_of_sample.metrics["total_trades"] < 12:
        st.error("时间外交易样本不足 12 笔：当前胜率不能作为实盘依据，应继续扩大股票池和时间窗口。")
    elif out_of_sample.metrics["status"] == "不建议实操":
        st.error("时间外结果未通过基本门槛，不建议按当前参数实操。")
    else:
        st.warning("结果最多支持小额模拟验证；单只股票、单段行情仍有选择偏差。")

    comparison = pd.DataFrame(
        [
            {
                "样本": "全样本（仅作诊断）",
                "交易数": full.metrics["total_trades"],
                "胜率": full.metrics["win_rate"],
                "收益率": full.metrics["total_return"],
                "最大回撤": full.metrics["max_drawdown"],
                "盈亏因子": full.metrics["profit_factor"],
            },
            {
                "样本": "后 35% 时间外（主结论）",
                "交易数": out_of_sample.metrics["total_trades"],
                "胜率": out_of_sample.metrics["win_rate"],
                "收益率": out_of_sample.metrics["total_return"],
                "最大回撤": out_of_sample.metrics["max_drawdown"],
                "盈亏因子": out_of_sample.metrics["profit_factor"],
            },
        ]
    )
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "胜率": st.column_config.NumberColumn(format="percent"),
            "收益率": st.column_config.NumberColumn(format="percent"),
            "最大回撤": st.column_config.NumberColumn(format="percent"),
            "盈亏因子": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if out_of_sample.fold_metrics:
        st.markdown("**分阶段稳定性**")
        st.dataframe(
            pd.DataFrame(out_of_sample.fold_metrics),
            hide_index=True,
            column_config={
                "胜率": st.column_config.NumberColumn(format="percent"),
                "收益率": st.column_config.NumberColumn(format="percent"),
                "最大回撤": st.column_config.NumberColumn(format="percent"),
            },
        )

    with st.expander("回测口径与限制", expanded=False):
        for label, description in out_of_sample.assumptions.items():
            st.write(f"- **{label}**：{description}")
        st.write("- 历史胜率不是未来概率；未计入停牌、ST、新股上市初期、排队成交和冲击成本。")
        st.write("- 参数为固定实验参数，没有针对当前股票做收益最大化调参。")

    trades = out_of_sample.trades_frame()
    if not trades.empty:
        display = trades.rename(
            columns={
                "entry_date": "买入日",
                "exit_date": "卖出日",
                "entry_price": "买入价",
                "exit_price": "卖出价",
                "quantity": "数量",
                "net_pnl": "净盈亏",
                "return_pct": "收益率(%)",
                "holding_bars": "持有交易日",
                "exit_reason": "退出原因",
                "fees": "费用",
            }
        )
        st.dataframe(
            display[
                [
                    "买入日",
                    "卖出日",
                    "买入价",
                    "卖出价",
                    "数量",
                    "净盈亏",
                    "收益率(%)",
                    "持有交易日",
                    "退出原因",
                    "费用",
                ]
            ],
            hide_index=True,
        )
        st.download_button(
            "下载时间外成交记录",
            data=display.to_csv(index=False).encode("utf-8-sig"),
            file_name="cn-lab-oos-trades.csv",
            mime="text/csv",
            icon=":material/download:",
        )


with st.sidebar:
    st.header("A 股短周期实验室")
    st.caption("独立分支平台 · 不连接券商 · 默认端口 8504")
    with st.form("lab_controls"):
        selected = st.selectbox(
            "股票代码",
            list(LAB_STOCKS),
            format_func=_format_stock,
            accept_new_options=True,
            filter_mode="contains",
            placeholder="例如 600019 或 000831",
        )
        mode_label = st.segmented_control(
            "操作模式",
            [PROFILES["overnight"].label, PROFILES["today"].label],
            default=PROFILES["overnight"].label,
            selection_mode="single",
        )
        source_key = st.selectbox(
            "行情源",
            list(SOURCES),
            format_func=lambda key: SOURCES[key][0],
        )
        history_days = st.slider("验证历史天数", 365, 1095, 900, 30)
        capital = st.number_input(
            "实验资金（CNY）", min_value=10_000.0, value=100_000.0, step=10_000.0
        )
        risk_pct = st.slider("单笔风险预算（%）", 0.5, 2.0, 1.0, 0.1) / 100
        max_position_pct = st.slider("单票仓位上限（%）", 10, 50, 30, 5) / 100
        st.markdown("**持仓输入（用于生成卖点）**")
        holding_quantity = int(st.number_input("当前持仓股数", min_value=0, value=0, step=100))
        sellable_quantity = int(st.number_input("今日可卖股数", min_value=0, value=0, step=100))
        average_cost = st.number_input(
            "持仓成本", min_value=0.0, value=0.0, step=0.01, format="%.2f"
        )
        st.form_submit_button("生成/刷新计划", type="primary", icon=":material/calculate:")

try:
    ticker, _, raw_code = cfg.normalize_ticker(selected)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
mode = "today" if mode_label == PROFILES["today"].label else "overnight"
stock_name = LAB_STOCKS.get(ticker, f"自定义 {raw_code}")

st.title(f"{stock_name}（{raw_code}）短周期实验")
with st.container(horizontal=True):
    st.badge(PROFILES[mode].label, color="blue", icon=":material/schedule:")
    st.badge(board_name(ticker), color="gray", icon=":material/account_balance:")
    st.badge("仅模拟研究", color="orange", icon=":material/science:")

if mode == "today":
    st.info("“当日决策”指卖出昨日可卖持仓，或今天买入后隔夜持有；A 股股票不能当日买入后当日卖出。")

result_slot = st.container()
with result_slot.skeleton():
    history, actual_source, history_failures, demo_mode = load_daily_history(
        source_key, ticker, history_days
    )

if history is None or history.empty:
    st.error("未能取得足够历史行情，无法生成策略计划。")
    if history_failures:
        st.code("\n".join(history_failures), language=None)
    st.stop()

quote, quote_source, quote_failures = load_latest_quote(source_key, ticker)
if not quote or quote.get("price") is None:
    quote = quote_from_history(
        history,
        ticker,
        SOURCES.get(actual_source, (actual_source, None, ""))[0]
        if actual_source in SOURCES
        else str(actual_source),
    )
current_price = float(quote["price"])

try:
    features, full_result, oos_result = evaluate_strategy(
        history,
        ticker=ticker,
        mode=mode,
        initial_capital=capital,
        risk_pct=risk_pct,
        max_position_pct=max_position_pct,
    )
    plan = build_action_plan(
        features,
        mode=mode,
        current_price=current_price,
        capital=capital,
        risk_pct=risk_pct,
        max_position_pct=max_position_pct,
        ticker=ticker,
        holding_quantity=holding_quantity,
        sellable_quantity=sellable_quantity,
        average_cost=average_cost,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

previous_close = float(history["close"].iloc[-2]) if len(history) > 1 else current_price
limit_down, limit_up = price_limit_bounds(previous_close, ticker)
st.caption(
    f"数据截至 {pd.Timestamp(history.index[-1]).strftime('%Y-%m-%d')} · "
    f"历史源 {actual_source or '未知'} · 报价源 {quote_source or quote.get('source', '历史回退')} · "
    f"现价 ¥{current_price:.2f} · 常规涨跌停参考 ¥{limit_down:.2f}/¥{limit_up:.2f}"
)
if demo_mode:
    st.warning("当前使用项目内历史样例，仅用于功能演示，不能据此交易。")
data_messages = tuple(history_failures) + tuple(quote_failures)
if data_messages:
    with st.expander("数据源回退信息"):
        st.code("\n".join(data_messages), language=None)

view = st.segmented_control(
    "查看内容",
    ["今日计划", "历史验证", "因子与信号"],
    default="今日计划",
    selection_mode="single",
    label_visibility="collapsed",
)

if view == "今日计划":
    _metric_cards(oos_result)
    _render_plan(plan, holding_quantity=holding_quantity, sellable_quantity=sellable_quantity)
    st.subheader("买卖点与当前风控线")
    _render_chart(features, full_result, plan)
elif view == "历史验证":
    _render_validation(full_result, oos_result)
else:
    st.subheader("评分构成")
    breakdown = score_breakdown(features)
    st.dataframe(
        breakdown,
        hide_index=True,
        column_config={
            "权重": st.column_config.NumberColumn(format="percent"),
            "通过度": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "贡献分": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    recent = features.tail(30).reset_index()
    recent = recent.rename(
        columns={
            recent.columns[0]: "日期",
            "close": "收盘价",
            "score": "评分",
            "entry_signal": "买入信号",
            "entry_low": "买入下沿",
            "entry_high": "买入上沿",
            "stop_loss": "止损",
            "target_1": "第一止盈",
            "target_2": "第二止盈",
        }
    )
    st.dataframe(
        recent[
            [
                "日期",
                "收盘价",
                "评分",
                "买入信号",
                "买入下沿",
                "买入上沿",
                "止损",
                "第一止盈",
                "第二止盈",
            ]
        ],
        hide_index=True,
    )
    st.caption("每行信号只使用该日及更早数据；历史成交发生在信号后的下一交易日开盘。")

st.divider()
st.caption("本平台只用于策略研究和模拟验证，不构成投资建议，也不会连接券商或自动下单。")
