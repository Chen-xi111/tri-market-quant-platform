"""
量化交易平台 — Streamlit 面板
标的: 中广核矿业 01164.HK

功能:
  Tab1: 行情概览 + 实时信号
  Tab2: 策略回测 + 绩效分析
  Tab3: 模拟交易 + 风控面板
"""

import matplotlib
from matplotlib import font_manager
import mplfinance as mpf
import numpy as np
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from data_sources import SOURCES
from market_data import fetch_history, fetch_quote, quote_from_history
from limited_guidance import build_limited_guidance
from market_clock import (
    HONG_KONG_TZ,
    HKMarketPhase,
    hk_market_phase,
    is_hk_continuous_session,
    is_hk_quote_session,
)
from trade_decision import assess_trade_decision, quote_requires_execution_block
from trade_plan import build_trade_plan, plan_order_rows
from strategies import STRATEGIES
from strategies.composite import CompositeStrategy
from strategies.base import SignalType
from backtest.engine import BacktestEngine
from backtest.metrics import metrics_summary
from backtest.validation import assess_readiness, robustness_score, scan_grid
from backtest.research_ui import render_research_validation
from risk.manager import RiskManager
from watchlist import load_user_stocks, remember_user_stock
import config as cfg

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体配置（macOS）。直接注册字体文件，避免独立 MPL 缓存找不到字体。
_chart_font = None
for _font_path in (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
):
    try:
        font_manager.fontManager.addfont(_font_path)
        if _chart_font is None:
            _chart_font = font_manager.FontProperties(fname=_font_path)
    except (FileNotFoundError, OSError, RuntimeError):
        pass
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'PingFang SC', 'Heiti TC', 'SimHei']
if _chart_font is not None:
    plt.rcParams['font.family'] = _chart_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
#  页面
# ============================================================
st.set_page_config(
    page_title=cfg.PAGE_TITLE,
    page_icon="📈",
    layout=cfg.PAGE_LAYOUT,
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .buy  { color: #00E676; font-weight: bold; }
    .sell { color: #FF5252; font-weight: bold; }
    .hold { color: #BDBDBD; }
    .metric-good { color: #00E676; }
    .metric-bad  { color: #FF5252; }
    .metric-warn { color: #FFD740; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  缓存数据加载
# ============================================================
@st.cache_data(
    ttl="5m", max_entries=48, show_spinner=False, refresh_mode="background",
)
def load_history(source_key: str, ticker: str, days: int = 730):
    """历史 K 线变化慢，单独做五分钟缓存，避免交互时反复拉取。"""
    return fetch_history(source_key, ticker, days)


@st.cache_data(
    ttl=15, max_entries=96, show_spinner=False, refresh_mode="background",
)
def load_quote(source_key: str, ticker: str):
    """实时价保持短缓存，不再带着整段历史数据一起过期。"""
    return fetch_quote(
        source_key,
        ticker,
        allow_slow_fallback=(source_key != "eastmoney"),
    )


def load_market_pair(source_key: str, ticker: str, days: int):
    """并发等待两条相互独立的行情链路，避免切股时串行阻塞。"""
    if not is_hk_quote_session():
        history = load_history(source_key, ticker, days)
        return history, (
            None,
            None,
            ("港股已休市，使用最后一根日 K 作为最新可用报价",),
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="market-data") as executor:
        history_future = executor.submit(load_history, source_key, ticker, days)
        quote_future = executor.submit(load_quote, source_key, ticker)
        return history_future.result(), quote_future.result()


@st.cache_data(ttl="10m", max_entries=48, show_spinner=False)
def evaluate_composite(frame: pd.DataFrame):
    """综合策略只随历史 K 线变化，不因界面交互重复训练 ML 模型。"""
    return CompositeStrategy().evaluate(frame.tail(365))


STOCK_DEPENDENT_STATE_KEYS = (
    "backtest_run",
    "bt_result",
    "bt_run",
    "tournament_run",
    "run_tournament",
    "scan_results",
)


def clear_stock_dependent_state():
    """切换股票后移除不能复用的分析结果，保留按股票隔离的模拟账户。"""
    for key in STOCK_DEPENDENT_STATE_KEYS:
        st.session_state.pop(key, None)


# ============================================================
#  侧边栏
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 量化交易平台")

        # ---- 股票选择 ----
        st.subheader("📈 标的选择")

        # 将用户新增代码写入本地自选清单，刷新或重启后仍可从下拉框选择。
        user_stocks = load_user_stocks(cfg.normalize_ticker)
        stock_options = list(dict.fromkeys([*cfg.PRESET_STOCKS, *user_stocks]))

        requested_ticker = st.query_params.get("ticker")
        if isinstance(requested_ticker, list):
            requested_ticker = requested_ticker[0] if requested_ticker else None
        if requested_ticker:
            try:
                requested_yahoo = cfg.normalize_ticker(str(requested_ticker))[0]
                if requested_yahoo not in stock_options:
                    stock_options.append(requested_yahoo)
                st.session_state["stock_selector"] = requested_yahoo
                del st.query_params["ticker"]
            except ValueError as exc:
                st.warning(f"候选股链接无效：{exc}")

        def format_stock_option(value):
            try:
                yahoo_code, _, raw_code = cfg.normalize_ticker(value)
            except ValueError:
                return str(value)
            name = cfg.PRESET_STOCKS.get(yahoo_code)
            return f"{raw_code}  {name}" if name else f"{raw_code}  自定义"

        selected_stock = st.selectbox(
            "选择或输入港股代码",
            stock_options,
            format_func=format_stock_option,
            index=0,
            key="stock_selector",
            accept_new_options=True,
            filter_mode="contains",
            placeholder="例如 00700 或 700.HK",
        )

        try:
            yahoo_code, futu_code, raw_code = cfg.normalize_ticker(selected_stock)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        try:
            _, stock_was_added = remember_user_stock(
                yahoo_code, cfg.PRESET_STOCKS, cfg.normalize_ticker,
            )
        except OSError as exc:
            stock_was_added = False
            st.warning(f"代码可继续查看，但未能保存到自选列表：{exc}")

        stock_name = cfg.PRESET_STOCKS.get(yahoo_code, f"自定义 {raw_code}")

        if st.session_state.get("_active_ticker") != yahoo_code:
            clear_stock_dependent_state()
            st.session_state["_active_ticker"] = yahoo_code

        st.caption(f"当前标的: **{stock_name}** ({raw_code})")
        if stock_was_added:
            st.success(f"已将 {raw_code} 保存到下拉列表。")

        st.divider()
        st.subheader("📊 策略选择")

        strategy_key = st.selectbox(
            "交易策略",
            list(STRATEGIES.keys()),
            format_func=lambda x: STRATEGIES[x][0],
        )

        st.divider()
        st.subheader("💰 资金与风控")

        capital = st.number_input("初始资金 (HKD)", value=cfg.INITIAL_CAPITAL, step=10000.0, format="%.0f")
        max_pos = st.slider("最大仓位 (%)", 5, 100, int(cfg.MAX_POSITION_PCT * 100), 5) / 100
        max_risk = st.slider("单笔风险 (%)", 0.5, 10.0, cfg.MAX_SINGLE_RISK_PCT * 100, 0.5) / 100
        if max_risk > 0.02:
            st.warning("单笔风险高于 2% 会显著放大连续亏损；实操前建议先降到 0.5%–1%。")
        if max_pos > 0.40:
            st.warning("单股仓位高于 40% 会形成集中度风险，市场扫描无法抵消。")
        lot_size = int(
            st.number_input(
                "每手股数",
                min_value=1,
                value=100,
                step=1,
                key=f"hk_lot_size_{yahoo_code}",
                help="港股每只证券的买卖单位可能不同；请按券商当前显示填写。",
            )
        )

        st.divider()
        st.subheader("🔌 数据源")
        source_key = st.selectbox(
            "行情API",
            list(SOURCES.keys()),
            format_func=lambda x: SOURCES[x][0],
        )
        st.caption(SOURCES[source_key][2])

        st.divider()
        st.subheader("📅 回测设置")
        backtest_days = st.slider("回测天数", 60, cfg.BACKTEST_YEARS * 365, 365, 30)

        st.divider()

        if st.button("🔄 刷新数据", width="stretch"):
            load_history.clear()
            load_quote.clear()
            st.rerun()

        return strategy_key, capital, max_pos, max_risk, lot_size, backtest_days, source_key, yahoo_code, futu_code, stock_name, raw_code


# ============================================================
#  主面板
# ============================================================
def main():
    strategy_key, capital, max_pos, max_risk, lot_size, backtest_days, source_key, yahoo_code, futu_code, stock_name, raw_code = render_sidebar()

    st.title(f"📈 {stock_name} ({raw_code})")
    phase_labels = {
        HKMarketPhase.PRE_OPEN: ("开市前时段", "orange"),
        HKMarketPhase.MORNING: ("上午连续交易", "green"),
        HKMarketPhase.LUNCH: ("午间时段", "orange"),
        HKMarketPhase.AFTERNOON: ("下午连续交易", "green"),
        HKMarketPhase.CLOSING_AUCTION: ("收市竞价", "orange"),
        HKMarketPhase.CLOSED: ("休市", "gray"),
    }
    phase_label, phase_color = phase_labels[hk_market_phase()]
    with st.container(horizontal=True):
        st.badge(f"港股 · {phase_label}", color=phase_color, icon=":material/schedule:")
        st.badge(f"HKD · 每手 {lot_size} 股", color="blue")
        st.caption(f"香港时间 {datetime.now(HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M')}")
    metadata_slot = st.empty()

    # 动态页签只执行用户当前看到的部分，避免切股时运行隐藏回测。
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📡 实时盯盘与信号", "🔬 策略回测", "📋 模拟交易与风控", "🏆 策略锦标赛"],
        key="main_tab",
        on_change="rerun",
    )

    # 加载数据
    with st.spinner(f"正在加载 {stock_name}…", show_time=True):
        history_result, quote_result = load_market_pair(
            source_key, yahoo_code, max(backtest_days + 100, 365)
        )
        df, actual_source, failures = history_result
        quote, quote_source, quote_failures = quote_result

    if df is None:
        if quote:
            quote_source_label = SOURCES[quote_source][0]
            metadata_slot.caption(
                f"报价数据: {quote_source_label} | 历史数据: 不可用 | "
                f"更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if tab1.open:
                with tab1:
                    render_realtime_only(quote, stock_name, raw_code, failures)
            else:
                st.warning("历史 K 线不可用，暂时不能计算策略、回测或风控指标。")
        else:
            st.error(f"⚠️ 无法获取 {stock_name} ({raw_code}) 行情数据。")
            diagnostics = [*failures, *quote_failures]
            if diagnostics:
                st.code("\n".join(dict.fromkeys(diagnostics)), language=None)
            st.info("请在终端运行诊断脚本查看原因：")
            st.code(f"cd \"/Users/mi_qi88sohu.com/Documents/宝钢股份\" && source .venv/bin/activate && python3 diagnose.py {raw_code}", language="bash")
        return

    history_is_cached = bool(df.attrs.get("market_data_cached"))
    actual_source_label = SOURCES[actual_source][0]
    if history_is_cached:
        actual_source_label += "（本地缓存）"
    if quote is None:
        quote = quote_from_history(df, yahoo_code, actual_source_label)
        quote_source_label = f"{actual_source_label}(历史回退)"
    else:
        quote_source_label = SOURCES[quote_source][0]
        if quote.get("_cached"):
            quote_source_label += "（本地缓存）"

    quote_is_nonlive = quote_requires_execution_block(quote, quote_source_label)

    metadata_slot.caption(
        f"历史数据: {actual_source_label}"
        + (f"（短历史 {len(df)}/60）" if df.attrs.get("market_data_limited") else "")
        + f" | 报价数据: {quote_source_label} | "
        f"更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if history_is_cached or quote_is_nonlive:
        st.warning(
            "当前至少一项行情来自本地缓存或历史 K 线回退；"
            "可查看数据，但决策卡会禁止新开仓。",
            icon=":material/cloud_off:",
        )
    if actual_source != source_key and source_key != "eastmoney":
        st.warning(
            f"已从 {SOURCES[source_key][0]} 自动切换到 {actual_source_label}。"
            + (f" 原因：{' | '.join(failures)}" if failures else "")
        )
    if df.attrs.get("market_data_limited"):
        st.warning(
            f"只获取到 {len(df)} 根日 K，未达完整七策略所需的 60 根。"
            "已自动切换为短历史观察模式；不生成策略下单仓位。",
            icon=":material/history:",
        )
        if failures:
            with st.expander("数据源回退详情", icon=":material/troubleshoot:"):
                st.code("\n".join(dict.fromkeys(failures)), language=None)

    # 实例化策略与风控
    StrategyClass = STRATEGIES[strategy_key][1]
    strategy = StrategyClass()
    risk_mgr = RiskManager(capital)

    # ================================================================
    #  Tab 1: 实时盯盘与信号
    # ================================================================
    if tab1.open:
        with tab1:
            try:
                render_overview(df, quote, strategy, risk_mgr, capital, max_risk, max_pos, lot_size, stock_name, raw_code)
            except Exception as e:
                st.error(f"Tab1 渲染失败: {e}")
                st.exception(e)

    # ================================================================
    #  Tab 2: 策略回测
    # ================================================================
    if tab2.open:
        with tab2:
            try:
                render_backtest(
                    df,
                    strategy,
                    capital,
                    backtest_days,
                    max_risk,
                    max_pos,
                    lot_size,
                    (
                        yahoo_code, strategy_key, float(capital), int(backtest_days),
                        float(max_risk), float(max_pos), int(lot_size),
                    ),
                )
            except Exception as e:
                st.error(f"Tab2 回测失败: {e}")
                st.exception(e)

    # ================================================================
    #  Tab 3: 模拟交易与风控
    # ================================================================
    if tab3.open:
        with tab3:
            try:
                render_paper_trading(df, quote, strategy, risk_mgr, capital, max_risk, max_pos, lot_size, futu_code, stock_name)
            except Exception as e:
                st.error(f"Tab3 模拟交易失败: {e}")
                st.exception(e)

    # ================================================================
    #  Tab 4: 策略锦标赛
    # ================================================================
    if tab4.open:
        with tab4:
            try:
                render_tournament(
                    df, capital, max_risk, max_pos, lot_size, yahoo_code
                )
            except Exception as e:
                st.error(f"Tab4 锦标赛失败: {e}")
                st.exception(e)


# ============================================================
#  Tab1: 行情概览 + 实时信号
# ============================================================
def render_realtime_only(quote, stock_name, raw_code, history_failures):
    """历史 K 线暂时不可用时，仍展示已成功获取的实时行情。"""
    price = float(quote["price"])
    change = float(quote.get("change") or 0)
    change_pct = float(quote.get("change_pct") or 0)

    st.warning(
        "已获取实时价，但历史 K 线暂时不可用。下方只显示行情快照，不生成交易建议。",
        icon=":material/warning:",
    )
    with st.container(horizontal=True):
        st.metric("最新价", f"HKD {price:.3f}", f"{change:+.3f} ({change_pct:+.2f}%)", border=True)
        st.metric("今开", f"{float(quote.get('open') or price):.3f}", border=True)
        st.metric("最高", f"{float(quote.get('high') or price):.3f}", border=True)
        st.metric("最低", f"{float(quote.get('low') or price):.3f}", border=True)

    st.caption(f"{stock_name} ({raw_code}) | {quote.get('source', '实时行情')}")
    if history_failures:
        with st.expander("历史数据诊断", icon=":material/troubleshoot:"):
            st.code("\n".join(history_failures), language=None)


def render_trade_plan(
    plan, *, current_price, volume_ratio, session_open, data_is_cached, median_turnover,
):
    """先显示实时决策门槛，再保留原有分批计划明细。"""
    decision = assess_trade_decision(
        plan,
        current_price,
        volume_ratio=volume_ratio,
        session_open=session_open,
        data_is_cached=data_is_cached,
        median_turnover=median_turnover,
    )
    color_map = {
        "orderable": "green", "ready_later": "blue", "waiting": "orange",
        "watch": "gray", "blocked": "red",
    }
    icon_map = {
        "orderable": ":material/check_circle:", "ready_later": ":material/schedule:",
        "waiting": ":material/hourglass_top:", "watch": ":material/visibility:",
        "blocked": ":material/block:",
    }
    st.subheader(
        "交易决策卡",
        help="综合分只负责发现候选；当前价格、盈亏比、止损距离和数据状态决定是否可执行。",
    )
    st.badge(
        decision["headline"], color=color_map[decision["status_key"]],
        icon=icon_map[decision["status_key"]],
    )
    st.write(f"**下一步：** {decision['next_action']}")
    if decision["hard_blockers"]:
        st.error("阻断原因：" + "；".join(decision["hard_blockers"]), icon=":material/block:")

    with st.container(horizontal=True):
        st.metric("现价位置", decision["price_location"], border=True)
        st.metric("现价至目标一", f"{decision['reward_risk_now']:.2f}R", border=True)
        st.metric("现价至止损", f"{decision['stop_distance_pct']:.1%}", border=True)
        st.metric(
            "条件满足后本次上限",
            f"{decision['planned_tranche_shares']:,} 股",
            f"风险 HKD {decision['planned_risk_now']:,.0f}",
            border=True,
        )

    st.markdown("**计划参数**")
    st.caption(
        f"市场状态：{plan['regime']} | 数据日：{plan['data_date']} | "
        f"每手 {plan['lot_size']} 股 | 日线计划：{plan['action']}"
    )
    if decision["median_turnover"] is not None:
        st.caption(f"20 日中位成交额：HKD {decision['median_turnover']:,.0f}")

    with st.container(horizontal=True):
        st.metric(
            "回调入场区",
            f"{plan['entry_low']:.3f}~{plan['entry_high']:.3f}",
            border=True,
        )
        st.metric(
            "确认价 / 追价上限",
            f"{plan['breakout_trigger']:.3f} / {decision['max_chase']:.3f}",
            border=True,
        )
        st.metric("保护止损", f"{plan['stop_loss']:.3f}", border=True)
        st.metric(
            "计划仓位",
            f"{plan['shares']:,} 股",
            f"HKD {plan['planned_value']:,.0f}",
            border=True,
        )

    with st.container(horizontal=True):
        st.metric("止盈一", f"{plan['target_1']:.3f}", f"{plan['reward_risk_1']:.2f}R", border=True)
        st.metric("止盈二", f"{plan['target_2']:.3f}", f"{plan['reward_risk_2']:.2f}R", border=True)
        st.metric("计划最大亏损", f"HKD {plan['planned_risk']:,.0f}", border=True)
        st.metric("风险预算", f"HKD {plan['risk_budget']:,.0f}", border=True)

    check_labels = {True: "通过", False: "未通过"}
    checks = " ".join(
        f":{'green' if passed else 'red'}-badge[{check_labels[passed]} · {label}]"
        for label, passed in decision["checks"].items()
    )
    st.markdown(checks)
    st.dataframe(
        pd.DataFrame(plan_order_rows(plan)),
        hide_index=True,
        width="stretch",
        column_config={
            "数量": st.column_config.NumberColumn("数量（股）", format="%d"),
            "价格": st.column_config.NumberColumn("参考价（HKD）", format="%.3f"),
        },
        key="trade_plan_steps",
    )
    with st.expander("实盘前人工复核（平台无法自动确认）", icon=":material/fact_check:"):
        st.markdown(
            "- 已检查公告、业绩、停复牌、公司行动和监管风险\n"
            "- 已向券商核对该股票当前买卖单位，不假设一律为 100 股\n"
            "- 盘口价差和可成交量能承受计划股数\n"
            "- 已核对同主题持仓、组合总风险和汇率风险"
        )
    st.caption(f"失效条件：{plan['invalidation']}。决策卡用于研究与模拟验证，不会自动提交实盘订单。")
    return decision


def render_limited_history_guidance(df):
    """新股/短历史的降级面板，不生成可执行订单。"""
    guidance = build_limited_guidance(df)
    st.subheader("🔎 短历史观察")
    st.caption(
        "这不是完整七策略结论。程序只使用现有日 K 描述趋势与波动，"
        "不计算仓位，不触发策略候选单。"
    )

    if guidance.status == "insufficient":
        st.warning(
            f"当前只有 {guidance.available_bars} 根日 K；"
            f"短周期观察至少需要 20 根。"
        )
        if guidance.reasons:
            st.caption("；".join(guidance.reasons))
        st.line_chart(df[["close"]].rename(columns={"close": "收盘价"}))
        return

    with st.container(horizontal=True):
        st.metric("观察方向", guidance.label, border=True)
        st.metric("短周期得分", f"{guidance.score} / 100", border=True)
        st.metric("有限信心度", f"{guidance.confidence:.0%}", border=True)
        st.metric(
            "样本完整度",
            f"{guidance.available_bars} / {guidance.required_bars}",
            border=True,
        )

    chart_columns = [column for column in ("close", "MA5", "MA20") if column in df]
    chart = df[chart_columns].rename(
        columns={"close": "收盘价", "MA5": "MA5", "MA20": "MA20"}
    )
    st.line_chart(chart, height=300)

    with st.container(horizontal=True):
        st.metric("现价", f"{guidance.price:.3f}", border=True)
        st.metric("波动参考下沿", f"{guidance.stop_reference:.3f}", border=True)
        st.metric("波动参考上沿", f"{guidance.target_reference:.3f}", border=True)
        st.metric(
            "RSI(14)",
            "—" if guidance.rsi is None else f"{guidance.rsi:.1f}",
            border=True,
        )

    if guidance.reasons:
        st.markdown("**当前证据：** " + " · ".join(guidance.reasons))
    st.info(
        "上下沿只是 2×ATR 波动参考，不是止损/止盈指令。"
        "待累积至 60 根日 K 后，平台会自动恢复完整策略。"
    )


def render_overview(df, quote, strategy, risk_mgr, capital, max_risk, max_pos, lot_size, stock_name, raw_code):
    if df is None or df.empty:
        st.warning("暂无行情数据。请检查网络后刷新。")
        return
    # 报价指标行
    price = (quote or {}).get("price") or df["close"].iloc[-1]
    change = (quote or {}).get("change", 0) or 0
    change_pct = (quote or {}).get("change_pct", 0) or 0

    cols = st.columns(6)
    cols[0].metric("💰 最新价", f"{price:.3f}", f"{change:+.3f} ({change_pct:+.2f}%)")
    cols[1].metric("📈 最高", f"{df['high'].iloc[-1]:.3f}")
    cols[2].metric("📉 最低", f"{df['low'].iloc[-1]:.3f}")
    previous_close = (quote or {}).get("prev_close")
    if previous_close is None:
        previous_close = df["close"].iloc[-2] if len(df) > 1 else df["close"].iloc[-1]
    cols[3].metric("📊 昨收", f"{previous_close:.3f}")
    _vr = df['VOL_RATIO'].iloc[-1] if 'VOL_RATIO' in df.columns else 1.0
    cols[4].metric("📦 量比", f"{_vr:.1f}x")
    _atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0.0
    cols[5].metric("📏 ATR", f"{_atr:.3f}")
    turnover_series = (
        pd.to_numeric(df["close"], errors="coerce")
        * pd.to_numeric(df["volume"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan).dropna()
    median_turnover = float(turnover_series.tail(20).median()) if not turnover_series.empty else 0.0

    st.divider()

    if len(df) < 60:
        render_limited_history_guidance(df)
        return

    # ============ 综合交易指导 ============
    st.subheader("🎯 日线交易指导（多策略投票）")
    decision = None
    try:
        cs = evaluate_composite(df)

        # 综合信号展示
        cs_col1, cs_col2, cs_col3, cs_col4 = st.columns([2, 1, 1, 1])

        with cs_col1:
            dir_colors = {
                "strong_buy": "🟢🟢", "buy": "🟢", "neutral": "⚪",
                "sell": "🔴", "strong_sell": "🔴🔴",
            }
            emoji = dir_colors.get(cs.direction, "⚪")
            score_color = "#00E676" if cs.score > 0 else ("#FF5252" if cs.score < 0 else "#BDBDBD")
            st.markdown(f"""
            <div style='padding:15px; border-radius:10px; background:#1a1a2e; text-align:center;'>
                <h2 style='color:{score_color}; margin:0;'>{emoji} 评分 {cs.score:+.1f}/10</h2>
                <p style='color:#ccc; margin:5px 0;'>{cs.summary}</p>
            </div>
            """, unsafe_allow_html=True)

        with cs_col2:
            st.metric("🟢 买入投票", f"{cs.buy_votes} / {len(cs.strategy_details)}")
            st.metric("信心度", f"{cs.confidence:.0%}")

        with cs_col3:
            st.metric("🔴 卖出投票", f"{cs.sell_votes} / {len(cs.strategy_details)}")
            st.metric("建议仓位", f"{cs.recommended_position*100:.0f}%")

        with cs_col4:
            st.metric("止损价", f"{cs.stop_loss:.3f}")
            st.metric("止盈价", f"{cs.take_profit:.3f}")

        plan = build_trade_plan(
            df,
            cs,
            capital=capital,
            risk_pct=max_risk,
            max_position_pct=max_pos,
            lot_size=lot_size,
        )
        decision = render_trade_plan(
            plan,
            current_price=float(price),
            volume_ratio=float(_vr),
            session_open=is_hk_continuous_session(),
            data_is_cached=(
                bool(df.attrs.get("market_data_cached"))
                or quote_requires_execution_block(quote)
            ),
            median_turnover=median_turnover,
        )

        # 各策略投票明细
        with st.expander("📋 各策略投票明细", expanded=True):
            vote_data = pd.DataFrame(cs.strategy_details)
            # 添加方向颜色
            st.dataframe(vote_data, width="stretch", hide_index=True)

            # 投票可视化
            dir_counts = vote_data["方向"].value_counts()
            vote_cols = st.columns(len(dir_counts))
            for i, (dir_name, count) in enumerate(dir_counts.items()):
                emoji_map = {"buy": "🟢", "sell": "🔴", "neutral": "⚪",
                            "lean_buy": "🟡", "lean_sell": "🟠", "error": "❌"}
                vote_cols[i].metric(
                    f"{emoji_map.get(dir_name, '⚪')} {dir_name}",
                    f"{count} 票"
                )
    except Exception as _cs_err:
        st.warning(f"综合信号计算中: {_cs_err}")

    st.divider()

    # 生成信号
    strategy.set_data(df.tail(365))
    strategy.generate_signals()
    signals_list = strategy.get_signals_list()

    # K线图
    col_chart, col_signals = st.columns([3, 2])

    with col_chart:
        st.subheader("📈 K线 + 信号标记")
        fig = plot_kline_with_signals(df.tail(120), signals_list, stock_name, raw_code)
        if fig:
            st.pyplot(fig)

    with col_signals:
        st.subheader("🔔 当前信号")
        recent_signals = []
        if signals_list and len(df) > 0:
            cutoff = df.index[-20]
            recent_signals = [s for s in signals_list if s.date >= cutoff]
        if recent_signals:
            for s in recent_signals[-5:]:
                emoji = "🟢" if s.type == SignalType.BUY else "🔴"
                st.markdown(
                    f"{emoji} **{s.type.value}** @ {s.price:.3f}<br>"
                    f"<small>{s.date.strftime('%m-%d')} | {s.reason}</small><br>"
                    f"<small>置信度: {s.confidence:.0%} | 仓位: {s.position_pct*100:.0f}%</small>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("近期无交易信号")

        st.divider()
        st.subheader("📐 技术指标快照")

        rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 50.0
        macd_hist = df["MACD_HIST"].iloc[-1] if "MACD_HIST" in df.columns else 0.0
        kdj_j = df["KDJ_J"].iloc[-1] if "KDJ_J" in df.columns else 50.0
        boll_pct = df["BOLL_PCT"].iloc[-1] if "BOLL_PCT" in df.columns else 0.5

        st.metric("RSI(14)", f"{rsi:.1f}", delta=f"{'超买' if rsi > cfg.RSI_OVERBOUGHT else '超卖' if rsi < cfg.RSI_OVERSOLD else '正常'}")
        st.metric("MACD 柱", f"{macd_hist:.4f}", delta="多头" if macd_hist > 0 else "空头")
        st.metric("KDJ-J", f"{kdj_j:.1f}", delta="超买钝化" if kdj_j > 100 else "超卖钝化" if kdj_j < 0 else "正常")
        st.metric("布林 %b", f"{boll_pct:.2f}", delta="触上轨" if boll_pct > 0.95 else "触下轨" if boll_pct < 0.05 else "中轨附近")

    # ATR 风险预算只是理论上限，不应与交易决策卡的可执行数量混淆。
    st.divider()
    st.subheader("🛡️ ATR 理论风险上限（非订单）")
    if decision is not None:
        st.caption(
            f"当前决策卡：{decision['headline']}；当前可执行数量 "
            f"{decision['order_shares_now']:,} 股。下方只回答“若已有有效信号，风险预算上限是多少”。"
        )
    else:
        st.caption("下方是 ATR 风险预算上限，不构成当前订单数量。")

    atr_val = df["ATR"].iloc[-1]
    pos = risk_mgr.position_size(
        price,
        atr_val,
        risk_pct=max_risk,
        max_position_pct=max_pos,
        lot_size=lot_size,
    )

    pos_cols = st.columns(4)
    pos_cols[0].metric("ATR 理论股数", f"{pos['shares']:,}股")
    pos_cols[1].metric("理论仓位金额", f"HKD {pos['position_value']:,.0f}", f"{pos['position_pct']:.1f}%")
    pos_cols[2].metric("ATR 止损参考", f"{pos['stop_loss_price']:.3f}", f"风险 HKD{pos['max_risk']:,.0f}")
    pos_cols[3].metric("ATR 止盈参考", f"{pos['take_profit_price']:.3f}")

    # 情景分析
    st.caption("📉 情景分析（基于 ATR）")
    scenario = risk_mgr.scenario_analysis(
        price,
        atr_val,
        risk_pct=max_risk,
        max_position_pct=max_pos,
        lot_size=lot_size,
    )
    sc_cols = st.columns(3)
    sc_cols[0].markdown(f"🟢 **最佳**: {scenario['best_case']['return_pct']:+.1f}% (HKD {scenario['best_case']['pnl']:+,.0f})")
    sc_cols[1].markdown("⚪ **基准**: 0.0%")
    sc_cols[2].markdown(f"🔴 **最差**: {scenario['worst_case']['return_pct']:+.1f}% (HKD {scenario['worst_case']['pnl']:+,.0f})")


# ============================================================
#  Tab2: 策略回测
# ============================================================
def render_backtest(
    df, strategy, capital, backtest_days, max_risk, max_pos, lot_size, context,
):
    st.subheader("🔬 策略回测")

    if df is None or len(df) < 60:
        available = 0 if df is None else len(df)
        st.warning(
            f"当前只有 {available} 根日 K，回测至少需要 60 根。"
            "程序不会用过短样本生成误导性绩效。"
        )
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"**策略**: {strategy.description}")
        st.write(f"**参数**: {strategy.params}")
    with col2:
        if st.button("▶️ 运行回测", type="primary", width="stretch"):
            with st.spinner("回测运行中..."):
                bt_df = df.tail(backtest_days).copy()
                engine = BacktestEngine(
                    strategy,
                    initial_capital=capital,
                    max_position_pct=max_pos,
                    risk_pct=max_risk,
                    lot_size=lot_size,
                )
                result = engine.run(bt_df)
                st.session_state["backtest_run"] = {
                    "context": context,
                    "result": result,
                }

    saved_run = st.session_state.get("backtest_run")
    if saved_run and saved_run.get("context") == context:
        result = saved_run["result"]
        metrics = result["metrics"]
        trades = result["trades"]
        eq = result["equity_curve"]
        assumptions = result.get("execution_assumptions", {})
        st.caption(
            "执行假设：收盘生成信号、下一交易日开盘成交 · "
            f"滑点 {assumptions.get('slippage_bps', 0):.1f}bp · "
            f"券商佣金/边 {assumptions.get('commission_rate', 0):.3%} · "
            f"印花税/边 {assumptions.get('stamp_duty_rate', cfg.STAMP_DUTY_RATE):.2%} · "
            f"港交所及监管征费/边 "
            f"{assumptions.get('sfc_levy_rate', cfg.SFC_LEVY_RATE) + assumptions.get('afrc_levy_rate', cfg.AFRC_LEVY_RATE) + assumptions.get('trading_fee_rate', cfg.TRADING_FEE_RATE):.4%} · "
            f"单笔风险 {assumptions.get('risk_pct', max_risk):.1%} · "
            f"每手 {assumptions.get('lot_size', lot_size)} 股"
        )

        # 绩效指标
        st.divider()
        m = metrics
        if not m or m.get("error"):
            st.warning("回测结果无效，可能数据不足")
            return
        readiness = assess_readiness(m)
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.metric("单次回测基础检查", f"{readiness['score']} / 100")
                st.badge(readiness["label"], color=readiness["color"])
            st.progress(readiness["score"] / 100)
            readiness_checks = " ".join(
                f":{'green' if passed else 'red'}-badge[{'通过' if passed else '未通过'} · {label}]"
                for label, passed in readiness["checks"].items()
            )
            st.markdown(readiness_checks)
            st.caption("这里只检查单次回测；还必须通过下方多折样本外验证，才可进入模拟盘观察。")
        cols = st.columns(6)
        cols[0].metric("总收益", f"{m.get('total_return', 0):+.1f}%",
                       delta=f"年化 {m.get('annual_return', 0):+.1f}%")
        cols[1].metric("最大回撤", f"{m.get('max_drawdown', 0):.1f}%")
        cols[2].metric("夏普比率", f"{m.get('sharpe_ratio', 0):.2f}")
        cols[3].metric("胜率", f"{m.get('win_rate', 0):.1f}%")
        cols[4].metric("盈亏比", f"{m.get('profit_factor', 0):.2f}")
        cols[5].metric("交易次数", f"{m.get('total_trades', 0)}")

        # 基准对比
        bench_ret = m.get('benchmark_return', 0)
        alpha = m.get('alpha', 0)
        bench_dd = m.get('benchmark_drawdown', 0)
        avg_hold = m.get('avg_holding_days', 0)
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("📊 买入持有收益", f"{bench_ret:+.1f}%",
                   delta=f"最大回撤 {bench_dd:.1f}%")
        bc2.metric("🎯 超额收益 Alpha", f"{alpha:+.1f}%",
                   delta="跑赢基准 ✅" if alpha > 0 else "落后基准 ❌")
        bc3.metric("⏱️ 平均持仓", f"{avg_hold:.0f} 天")
        bc4.metric("📐 信息比率", f"{m.get('information_ratio', 0):.2f}",
                   delta=f"Beta {m.get('beta', 0):.2f}")

        # 权益曲线
        st.divider()
        st.subheader("📈 权益曲线")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        ax1.plot(eq.index, eq.values, color="#1976D2", linewidth=1.5, label="策略权益")
        ax1.fill_between(eq.index, eq.values, eq.values[0], alpha=0.1, color="#1976D2")
        # 买入持有基准线
        bt_df = result["df"]
        if len(bt_df) > 0:
            first_price = bt_df["close"].iloc[0]
            if first_price > 0:
                bench_eq = capital * bt_df["close"] / first_price
                bench_eq = bench_eq.reindex(eq.index).ffill()
                ax1.plot(bench_eq.index, bench_eq.values, color="#FF9800",
                         linewidth=1.2, linestyle='--', alpha=0.7, label="买入持有基准")
        ax1.set_ylabel("Equity (HKD)")
        ax1.set_title("策略权益曲线 vs 买入持有基准", fontweight="bold")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 回撤曲线
        peak = eq.cummax()
        drawdown = (eq - peak) / peak * 100
        ax2.fill_between(eq.index, 0, drawdown.values, color="#EF5350", alpha=0.5)
        ax2.set_ylabel("Drawdown %")
        ax2.set_ylim(drawdown.min() * 1.2, 0)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # 交易明细
        st.divider()
        st.subheader("📋 交易明细")
        if trades:
            trade_df = pd.DataFrame(trades, columns=["入场日", "出场日", "入场价", "出场价", "盈亏", "盈亏%", "原因"])
            trade_df["入场日"] = trade_df["入场日"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x)[:10])
            trade_df["出场日"] = trade_df["出场日"].apply(lambda x: x.strftime("%Y-%m-%d") if hasattr(x, 'strftime') else str(x)[:10])
            trade_df["盈亏"] = trade_df["盈亏"].round(2)
            trade_df["盈亏%"] = trade_df["盈亏%"].round(2)

            def color_pnl(val):
                return 'color: #00E676' if val > 0 else 'color: #FF5252'
            st.dataframe(
                trade_df.style.map(color_pnl, subset=["盈亏", "盈亏%"]),
                width="stretch", height=300,
            )
        else:
            st.info("回测期间无交易信号")

        # 追加绩效详情
        with st.expander("📊 完整绩效报告"):
            st.code(metrics_summary(m), language=None)


# ============================================================
#  Tab3: 模拟交易与风控
# ============================================================
def render_paper_trading(
    df, quote, strategy, risk_mgr, capital, max_risk, max_pos, lot_size,
    futu_code, stock_name,
):
    """模拟订单台：把计划、订单确认、风控反馈和成交记录放到同一流程。"""
    from execution.auto_trader import AutoTrader
    from execution.base import OrderSide

    st.subheader("模拟交易控制台")
    st.caption("纯模拟环境：不会连接券商，也不会提交真实订单。成交计入 5bp 滑点和 0.1% 费用。")

    price = float((quote or {}).get("price") or df["close"].iloc[-1])
    account_key = f"{futu_code}|{float(capital):.2f}"
    paper_traders = st.session_state.setdefault("paper_traders", {})
    paper_logs = st.session_state.setdefault("paper_logs", {})
    paper_flashes = st.session_state.setdefault("paper_flashes", {})
    if account_key not in paper_traders:
        paper_traders[account_key] = AutoTrader.paper(capital, symbol=futu_code)
    trader = paper_traders[account_key]
    log = paper_logs.setdefault(account_key, [])

    trader.gate.max_position_pct = float(max_pos)
    trader.gate.max_order_value = float(capital) * float(max_pos)
    trader.gate.daily_loss_limit = float(capital) * 0.05
    trader.broker.update_price(futu_code, price)

    with st.container(horizontal=True):
        st.badge("模拟盘", color="blue", icon=":material/science:")
        st.badge(f"{stock_name} · {futu_code}", color="gray")
        st.badge(
            "已暂停" if trader.is_paused else "可下单",
            color="orange" if trader.is_paused else "green",
            icon=":material/pause:" if trader.is_paused else ":material/check_circle:",
        )

    flash = paper_flashes.pop(account_key, None)
    if flash:
        (st.success if flash[0] else st.error)(flash[1])

    account = trader.broker.get_account()
    positions = trader.broker.get_positions()
    total = float(account["total_assets"])
    initial = float(account.get("initial_cash", capital))
    total_pnl = total - initial
    with st.container(horizontal=True):
        st.metric("账户权益", f"HKD {total:,.0f}", f"{total_pnl:+,.0f}", border=True)
        st.metric("可用现金", f"HKD {account['cash']:,.0f}", border=True)
        st.metric("持仓市值", f"HKD {account['market_value']:,.0f}", border=True)
        st.metric("已付费用", f"HKD {account.get('fees_paid', 0):,.2f}", border=True)
        st.metric("已实现盈亏", f"HKD {account.get('realized_pnl', 0):+,.0f}", border=True)

    if positions:
        position_frame = pd.DataFrame([
            {
                "代码": p["symbol"], "数量": p["qty"], "成本": p["cost"],
                "现价": p["price"], "市值": p["qty"] * p["price"], "浮动盈亏": p["pnl"],
            }
            for p in positions
        ])
        st.dataframe(
            position_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "数量": st.column_config.NumberColumn(format="%d 股"),
                "成本": st.column_config.NumberColumn(format="%.3f"),
                "现价": st.column_config.NumberColumn(format="%.3f"),
                "市值": st.column_config.NumberColumn(format="HKD %.0f"),
                "浮动盈亏": st.column_config.NumberColumn(format="HKD %+.0f"),
            },
        )
    else:
        st.info("当前为空仓。可以先在右侧核对计划，再用左侧订单票模拟一笔交易。", icon=":material/inventory_2:")

    composite = evaluate_composite(df)
    plan = build_trade_plan(
        df,
        composite,
        capital=capital,
        risk_pct=max_risk,
        max_position_pct=max_pos,
        lot_size=lot_size,
    )
    plan_entry_ready = plan["entry_low"] <= price <= plan["entry_high"]

    def record_order(action, side, requested_price, quantity, result):
        fill_price = result.order.price if result.success and result.order else None
        log.append({
            "时间": datetime.now().strftime("%H:%M:%S"),
            "来源": action,
            "方向": "买入" if side == OrderSide.BUY else "卖出",
            "数量": int(quantity),
            "委托价": float(requested_price),
            "成交价": fill_price,
            "状态": "成交" if result.success else "拦截",
            "说明": result.message,
        })
        paper_flashes[account_key] = (
            result.success,
            f"{'模拟成交' if result.success else '订单未执行'}：{result.message}",
        )

    ticket_col, plan_col = st.columns([1, 1], gap="large")
    with ticket_col:
        st.markdown("#### 模拟订单票")
        position_qty = sum(p["qty"] for p in positions if p["symbol"] == futu_code)
        with st.form(f"paper_ticket_{account_key}", border=True):
            side_label = st.segmented_control(
                "方向", ["买入", "卖出"], default="买入",
                selection_mode="single",
            )
            order_price = st.number_input(
                "模拟成交参考价（HKD）", min_value=0.001,
                value=round(price, 3), step=0.01, format="%.3f",
            )
            lots = st.number_input(
                f"手数（1 手 = {lot_size} 股）", min_value=1,
                value=1, step=1,
            )
            quantity = int(lots) * int(lot_size)
            st.caption(
                f"订单数量 {quantity:,} 股 · 名义金额 HKD {order_price * quantity:,.0f} · "
                f"当前可卖 {position_qty:,} 股"
            )
            confirmed = st.checkbox("我确认这是模拟订单，并已核对方向、价格和数量")
            submitted = st.form_submit_button(
                "提交至模拟盘", type="primary", width="stretch",
            )

        if submitted:
            if not confirmed:
                paper_flashes[account_key] = (False, "请先勾选模拟订单确认。")
            else:
                side = OrderSide.BUY if side_label == "买入" else OrderSide.SELL
                result = trader.place_manual_order(order_price, quantity, side)
                record_order("手动订单", side, order_price, quantity, result)
            st.rerun()

    with plan_col:
        st.markdown("#### 策略候选单")
        status_color = "green" if plan["actionable"] else "orange"
        st.badge(plan["action"], color=status_color)
        with st.container(border=True):
            st.write(f"**市场状态：** {plan['regime']}")
            st.write(f"**首笔条件：** {plan['entry_low']:.3f}–{plan['entry_high']:.3f}")
            st.write(f"**首笔数量：** {plan['first_shares']:,} 股")
            st.write(f"**止损 / 止盈：** {plan['stop_loss']:.3f} / {plan['target_1']:.3f}")
            if plan["actionable"] and not plan_entry_ready:
                st.warning(f"现价 {price:.3f} 尚未进入回调区，候选单保持等待。")

            strategy_confirmed = st.checkbox(
                "确认按当前策略首笔试仓（模拟）",
                key=f"strategy_confirm_{account_key}",
            )
            execute_plan = st.button(
                "执行首笔候选单",
                width="stretch",
                disabled=(
                    trader.is_paused or not plan["actionable"] or
                    not plan_entry_ready or plan["first_shares"] <= 0 or
                    not strategy_confirmed
                ),
                key=f"execute_plan_{account_key}",
            )
            if execute_plan:
                quantity = int(plan["first_shares"])
                result = trader.place_manual_order(price, quantity, OrderSide.BUY, action="PLAN")
                record_order("策略计划", OrderSide.BUY, price, quantity, result)
                st.rerun()

    st.divider()
    gate_col, control_col = st.columns([2, 1])
    with gate_col:
        st.markdown("#### 风控状态")
        gate_status = trader.gate.status()
        with st.container(horizontal=True):
            st.metric("日内已实现", f"HKD {gate_status['daily_pnl']:+,.0f}", border=True)
            st.metric("连续亏损", f"{gate_status['consecutive_losses']} 笔", border=True)
            st.metric("安全检查单", f"{gate_status['today_orders']} 笔", border=True)
            st.metric("单笔上限", f"HKD {trader.gate.max_order_value:,.0f}", border=True)
        st.caption(
            f"买入最大仓位 {max_pos:.0%} · 单笔风险预算 {max_risk:.1%} · "
            "卖出减仓不受买入金额上限阻挡"
        )
        if trader.is_paused:
            st.error(f"交易引擎已暂停：{trader.status()['pause_reason']}")

    with control_col:
        st.markdown("#### 账户控制")
        if trader.is_paused:
            if st.button("恢复模拟下单", width="stretch", key=f"resume_{account_key}"):
                trader.resume()
                st.rerun()
        else:
            if st.button("暂停模拟下单", width="stretch", key=f"pause_{account_key}"):
                trader.pause("用户手动暂停")
                st.rerun()
        allow_reset = st.checkbox("允许重置此标的模拟账户", key=f"allow_reset_{account_key}")
        if st.button(
            "重置账户", width="stretch", disabled=not allow_reset,
            key=f"reset_{account_key}",
        ):
            paper_traders[account_key] = AutoTrader.paper(capital, symbol=futu_code)
            paper_logs[account_key] = []
            paper_flashes[account_key] = (True, "模拟账户已重置。")
            st.rerun()

    st.divider()
    orders = trader.broker.get_orders()
    order_col, console_col = st.columns([3, 2], gap="large")
    with order_col:
        st.markdown("#### 成交与拒单记录")
        if orders:
            order_frame = pd.DataFrame([
                {
                    "时间": order.update_time.strftime("%H:%M:%S"),
                    "订单号": order.order_id,
                    "方向": "买入" if order.side == OrderSide.BUY else "卖出",
                    "数量": order.quantity,
                    "成交价": order.price,
                    "状态": order.status.value,
                    "成本说明": order.reason or "—",
                }
                for order in reversed(orders[-30:])
            ])
            st.dataframe(
                order_frame, hide_index=True, width="stretch", height=260,
                column_config={
                    "数量": st.column_config.NumberColumn(format="%d"),
                    "成交价": st.column_config.NumberColumn(format="%.3f"),
                },
            )
        else:
            st.caption("暂无券商模拟记录。")

    with console_col:
        console_state = "error" if log and log[-1]["状态"] == "拦截" else "complete"
        with st.status("执行控制台", state=console_state, expanded=True):
            if not log:
                st.write("等待订单。每次提交后的成交价、费用或拦截原因会显示在这里。")
            for entry in reversed(log[-8:]):
                icon = "✅" if entry["状态"] == "成交" else "⛔"
                st.write(
                    f"{icon} `{entry['时间']}` {entry['来源']} · {entry['方向']} "
                    f"{entry['数量']:,} 股 — {entry['说明']}"
                )

    if log:
        export_frame = pd.DataFrame(log)
        st.download_button(
            "导出模拟订单日志",
            data=export_frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"paper-trades-{futu_code.replace('.', '-')}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="content",
            key=f"download_paper_log_{account_key}",
        )


# ============================================================
#  绘图工具：K线 + 信号标记
# ============================================================
def render_tournament(df, capital, max_risk, max_pos, lot_size, ticker):
    """策略锦标赛：六策略并行回测 → 对比排名 → 参数优化"""
    st.subheader("🏆 策略锦标赛")
    st.caption("七套策略先做样本内横向初筛；最终策略选择由下方研究级走样本外验证决定。")

    if df is None or len(df) < 60:
        available = 0 if df is None else len(df)
        st.warning(f"当前只有 {available} 根日 K；策略锦标赛至少需要 60 根。")
        return

    # ---- 参数：选择要对比的策略 + 回测范围 ----
    col_cfg1, col_cfg2 = st.columns([2, 1])
    with col_cfg1:
        selected_strats = st.multiselect(
            "选择参赛策略",
            list(STRATEGIES.keys()),
            default=list(STRATEGIES.keys()),
            format_func=lambda x: STRATEGIES[x][0],
        )
    with col_cfg2:
        if len(df) > 60:
            bt_days = st.slider("回测天数", 60, len(df), min(len(df), 365 * 3), 30)
        else:
            bt_days = len(df)
            st.caption(f"使用全部 {bt_days} 条数据")

        run_clicked = st.button(
            "🚀 开始锦标赛",
            type="primary",
            width="stretch",
            disabled=not selected_strats,
        )

    tournament_context = (
        ticker,
        float(capital),
        float(max_risk),
        float(max_pos),
        int(lot_size),
        tuple(selected_strats),
        int(bt_days),
    )

    if run_clicked:
        bt_df = df.tail(bt_days).copy()
        results = {}
        progress = st.progress(0, "回测进行中...")
        status_text = st.empty()

        for idx, key in enumerate(selected_strats):
            name, strategy_class = STRATEGIES[key]
            status_text.text(f"正在回测: {name}...")

            try:
                engine = BacktestEngine(
                    strategy_class(),
                    initial_capital=capital,
                    max_position_pct=max_pos,
                    risk_pct=max_risk,
                    lot_size=lot_size,
                )
                result = engine.run(bt_df)
                results[key] = {
                    "name": name,
                    "metrics": result["metrics"],
                    "equity": result["equity_curve"],
                    "trades": result["trades"],
                }
            except Exception as exc:
                results[key] = {"name": name, "error": str(exc)}

            progress.progress((idx + 1) / len(selected_strats))

        status_text.empty()
        progress.empty()
        st.session_state["tournament_run"] = {
            "context": tournament_context,
            "results": results,
        }

    saved_run = st.session_state.get("tournament_run")
    if not saved_run or saved_run.get("context") != tournament_context:
        st.info("选择策略 → 点击「开始锦标赛」运行对比")
        if len(df) > 0:
            st.caption(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}，共{len(df)}条")
        return

    bt_df = df.tail(bt_days).copy()
    results = saved_run["results"]

    # ---- 排名表 ----
    st.divider()
    st.subheader("📊 样本内初筛（按夏普比率）")

    # 构建对比表
    table_data = []
    for key, r in results.items():
        if "error" in r:
            continue
        m = r["metrics"]
        table_data.append({
            "排名": 0,
            "策略": r["name"],
            "总收益%": round(m.get("total_return", 0), 1),
            "基准%": round(m.get("benchmark_return", 0), 1),
            "超额%": round(m.get("alpha", 0), 1),
            "年化%": round(m.get("annual_return", 0), 1),
            "最大回撤%": round(m.get("max_drawdown", 0), 1),
            "夏普": round(m.get("sharpe_ratio", 0), 2),
            "胜率%": round(m.get("win_rate", 0), 1),
            "交易次数": m.get("total_trades", 0),
            "准备度": assess_readiness(m)["score"],
            "key": key,
        })

    # 按夏普排序
    table_data.sort(key=lambda x: x["夏普"], reverse=True)
    for i, row in enumerate(table_data):
        row["排名"] = i + 1

    if not table_data:
        errors = [r.get("error", "未知错误") for r in results.values()]
        st.error("所有策略回测都失败。")
        st.code("\n".join(errors), language=None)
        return

    # 高亮最优
    compare_df = pd.DataFrame(table_data)
    compare_df_display = compare_df.drop(columns=["key"])

    def highlight_best(s):
        if s.name == "排名":
            return ['background-color: #FFD700; font-weight: bold' if v == 1 else '' for v in s]
        return ['' for _ in s]

    styled = compare_df_display.style.apply(highlight_best, axis=0)
    st.dataframe(styled, width="stretch", hide_index=True)

    # 最佳策略标记
    best = table_data[0]
    st.info(
        f"样本内领先: **{best['策略']}** | 夏普 {best['夏普']:.2f} | "
        f"年化 {best['年化%']:.1f}% | 最大回撤 {best['最大回撤%']:.1f}%"
    )
    st.caption("样本内领先不等于可选策略，也不会直接生成买入建议。")

    # ---- 权益曲线叠加 ----
    st.divider()
    st.subheader("📈 权益曲线对比")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#FFD700", "#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#00BCD4"]
    color_map = {row["key"]: colors[i % len(colors)] for i, row in enumerate(table_data)}

    for row in table_data:
        key = row["key"]
        equity = results[key]["equity"]
        normalized = equity / equity.iloc[0]
        ax.plot(normalized.index, normalized.values,
                color=color_map[key], linewidth=1.2, alpha=0.8,
                label=f"{row['策略']} (Sharpe {row['夏普']:.2f})")

    ax.axhline(y=1.0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title("策略权益曲线对比（归一化，初始=1.0）", fontweight='bold')
    ax.set_ylabel("归一化权益")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # ---- 回撤对比 ----
    st.subheader("📉 回撤对比")
    fig2, axes2 = plt.subplots(len(table_data), 1, figsize=(14, 2.5 * len(table_data)), sharex=True)

    if len(table_data) == 1:
        axes2 = [axes2]

    for i, row in enumerate(table_data):
        key = row["key"]
        equity = results[key]["equity"]
        peak = equity.cummax()
        dd = (equity - peak) / peak * 100
        axes2[i].fill_between(dd.index, 0, dd.values, color=color_map[key], alpha=0.4)
        axes2[i].plot(dd.index, dd.values, color=color_map[key], linewidth=0.8)
        axes2[i].set_ylabel(f"{row['策略']}", fontsize=9)
        axes2[i].set_ylim(min(dd.min() * 1.3, -1), 0.5)
        axes2[i].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ---- Alpha 因子 IC 分析（如果 alpha_factors 参赛） ----
    if "alpha_factors" in results and "error" not in results.get("alpha_factors", {}):
        st.divider()
        st.subheader("🔍 Alpha 因子 IC 分析")
        st.caption("IC (Information Coefficient): 因子值与未来收益的秩相关系数。绝对值越大，预测力越强。")

        try:
            from strategies.alpha_factors import factor_ic_analysis
            ics = factor_ic_analysis(bt_df)
            if ics:
                ic_df = pd.DataFrame(
                    {"因子": list(ics.keys()), "IC": list(ics.values())}
                ).sort_values("IC", key=abs, ascending=False)

                fig3, ax3 = plt.subplots(figsize=(12, 5))
                colors_bar = ["#4CAF50" if v > 0 else "#FF5252" for v in ic_df["IC"]]
                ax3.barh(range(len(ic_df)), ic_df["IC"].values, color=colors_bar)
                ax3.set_yticks(range(len(ic_df)))
                ax3.set_yticklabels(ic_df["因子"].values)
                ax3.axvline(x=0, color='gray', linewidth=0.5)
                ax3.set_xlabel("Rank IC (与未来5日收益)")
                ax3.set_title("Alpha 因子 IC 分析", fontweight='bold')
                ax3.invert_yaxis()
                st.pyplot(fig3)
                plt.close(fig3)

                st.dataframe(ic_df, width="stretch", hide_index=True)
        except Exception as e:
            st.warning(f"IC分析失败: {e}")

    # ---- 参数扫描：训练集选参 + 留出样本验证 ----
    render_research_validation(
        bt_df, STRATEGIES, selected_strats, capital, max_risk, max_pos,
        lot_size, ticker, state_prefix="hk",
    )
    return
    st.divider()
    st.subheader("参数稳健性扫描")
    st.caption("前 70% 数据用于观察参数，后 30% 留出验证；最终按样本外表现排序。")

    scan_strat = st.selectbox(
        "选择要优化的策略",
        selected_strats,
        format_func=lambda x: STRATEGIES[x][0],
        key="scan_strat",
    )

    param_name, param_label, params_to_test = scan_grid(scan_strat)
    st.write(f"**扫描参数：** {param_label} (`{param_name}`)")
    st.caption("候选值：" + " · ".join(str(value) for value in params_to_test))
    scan_context = (
        ticker, scan_strat, param_name, tuple(params_to_test), int(bt_days),
        float(capital), float(max_risk), float(max_pos), int(lot_size),
    )
    scan_disabled = len(bt_df) < 180
    if scan_disabled:
        st.warning("至少需要 180 根日 K 才能进行样本外参数扫描。")

    if st.button(
        "运行样本外扫描", icon=":material/experiment:", width="stretch",
        disabled=scan_disabled, key="run_robustness_scan",
    ):
        split_idx = int(len(bt_df) * 0.7)
        validation_start = bt_df.index[split_idx]
        scan_results = []
        with st.status("正在运行训练集与留出样本…", expanded=True) as scan_status:
            for p_val in params_to_test:
                st.write(f"测试 {param_label} = {p_val}")
                _, StratClass = STRATEGIES[scan_strat]
                try:
                    train_engine = BacktestEngine(
                        StratClass(params={param_name: p_val}),
                        initial_capital=capital,
                        max_position_pct=max_pos,
                        risk_pct=max_risk,
                        lot_size=lot_size,
                    )
                    train_result = train_engine.run(bt_df.iloc[:split_idx].copy())
                    validation_engine = BacktestEngine(
                        StratClass(params={param_name: p_val}),
                        initial_capital=capital,
                        max_position_pct=max_pos,
                        risk_pct=max_risk,
                        lot_size=lot_size,
                    )
                    validation_result = validation_engine.run(
                        bt_df.copy(), trade_start_date=validation_start
                    )
                    train_m = train_result["metrics"]
                    validation_m = validation_result["metrics"]
                    scan_results.append({
                        "参数值": p_val,
                        "训练夏普": round(train_m.get("sharpe_ratio", 0), 2),
                        "验证夏普": round(validation_m.get("sharpe_ratio", 0), 2),
                        "验证收益%": round(validation_m.get("total_return", 0), 1),
                        "验证回撤%": round(validation_m.get("max_drawdown", 0), 1),
                        "验证交易数": int(validation_m.get("total_trades", 0)),
                        "稳健分": robustness_score(train_m, validation_m),
                    })
                except Exception as exc:
                    st.write(f"跳过 {p_val}：{exc}")
            scan_status.update(label="样本外扫描完成", state="complete")

        st.session_state["scan_results"] = {
            "context": scan_context,
            "validation_start": str(validation_start.date()),
            "rows": scan_results,
        }

    saved_scan = st.session_state.get("scan_results")
    if saved_scan and saved_scan.get("context") == scan_context and saved_scan.get("rows"):
        scan_df = pd.DataFrame(saved_scan["rows"]).sort_values("稳健分", ascending=False)
        best_param = scan_df.iloc[0]
        if int(best_param["验证交易数"]) < 3:
            st.warning(
                "留出样本交易不足 3 笔，本轮不选择所谓“最优参数”；请扩大回测周期或更换策略。"
            )
        else:
            st.success(
                f"样本外首选：**{param_label} = {best_param['参数值']}** · "
                f"验证夏普 {best_param['验证夏普']:.2f} · "
                f"验证收益 {best_param['验证收益%']:+.1f}%"
            )
        st.caption(
            f"留出样本开始于 {saved_scan['validation_start']}。若验证交易数少于 3，"
            "该结果仍不足以支持实盘决策。"
        )
        st.dataframe(
            scan_df,
            hide_index=True,
            width="stretch",
            column_config={
                "验证收益%": st.column_config.NumberColumn(format="%+.1f%%"),
                "验证回撤%": st.column_config.NumberColumn(format="%.1f%%"),
                "稳健分": st.column_config.NumberColumn(format="%.3f"),
            },
        )


def plot_kline_with_signals(df: pd.DataFrame, signals: list, stock_name="", raw_code=""):
    if df.empty:
        return None

    plot_df = df.tail(120).rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    mc = mpf.make_marketcolors(up='red', down='green', edge='inherit', wick='inherit',
                                volume={'up': 'red', 'down': 'green'})
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':',
        facecolor='#FAFAFA',
        rc={'font.family': _chart_font.get_name() if _chart_font else 'DejaVu Sans'},
    )

    # 准备买入/卖出标记
    buy_markers = []
    sell_markers = []
    for s in signals:
        if s.date in plot_df.index:
            if s.type == SignalType.BUY:
                buy_markers.append(s.date)
            else:
                sell_markers.append(s.date)

    addplot = []
    if buy_markers:
        buy_series = pd.Series(np.nan, index=plot_df.index)
        for d in buy_markers:
            if d in buy_series.index:
                buy_series[d] = plot_df.loc[d, "Low"] * 0.98
        addplot.append(mpf.make_addplot(buy_series, type='scatter', markersize=100,
                                         marker='^', color='#00E676'))
    if sell_markers:
        sell_series = pd.Series(np.nan, index=plot_df.index)
        for d in sell_markers:
            if d in sell_series.index:
                sell_series[d] = plot_df.loc[d, "High"] * 1.02
        addplot.append(mpf.make_addplot(sell_series, type='scatter', markersize=100,
                                         marker='v', color='#FF5252'))

    fig, axes = mpf.plot(
        plot_df, type='candle', mav=(5, 20, 60), volume=True,
        style=style, figsize=(14, 8), returnfig=True,
        addplot=addplot if addplot else [],
        title=f'\n{stock_name} ({raw_code})',
        ylabel='Price (HKD)', ylabel_lower='Volume',
    )
    return fig


if __name__ == "__main__":
    main()
