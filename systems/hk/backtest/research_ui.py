"""研究级回测的 Streamlit 展示层。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .research import evaluate_parameter_grid
from .validation import scan_grid


def render_research_validation(df, strategies, selected_strategies, capital, max_risk, max_position, lot_size, ticker, state_prefix=""):
    st.divider(); st.subheader("研究级走样本外验证")
    st.caption("参数不再靠单次 70/30 切分决定：每个候选值接受 4 折扩展窗口、1 日隔离、下一交易日开盘成交、15bp 滑点压力、块自助区间和多重试验校正。")
    if not selected_strategies:
        st.info("请先选择至少一个参赛策略。"); return
    key_prefix = f"{state_prefix}_research"
    strategy_key = st.selectbox("选择要研究的策略", selected_strategies, format_func=lambda value: strategies[value][0], key=f"{key_prefix}_strategy")
    param_name, param_label, values = scan_grid(strategy_key)
    st.write(f"**研究参数：** {param_label} (`{param_name}`)"); st.caption("完整候选集（全部计入选择偏差）：" + " · ".join(str(value) for value in values))
    context = (ticker, strategy_key, param_name, tuple(values), len(df), float(capital), float(max_risk), float(max_position), int(lot_size)); disabled = len(df) < 241
    if disabled: st.warning("研究级验证至少需要 241 根日 K；请扩大回测周期。")
    if st.button("运行研究级验证", icon=":material/experiment:", type="primary", width="stretch", disabled=disabled, key=f"{key_prefix}_run"):
        _, strategy_class = strategies[strategy_key]
        try:
            with st.status("正在运行 4 折样本外与成本压力检验…", expanded=True) as status:
                st.write("每个参数运行基础成本与 15bp/边滑点压力场景。")
                report = evaluate_parameter_grid(df, strategy_class, param_name, values, {"initial_capital": capital, "max_position_pct": max_position, "risk_pct": max_risk, "lot_size": lot_size})
                status.update(label="研究级验证完成", state="complete", expanded=False)
            st.session_state[f"{key_prefix}_result"] = {"context": context, "report": report}
        except Exception as exc: st.error(f"研究级验证未完成：{type(exc).__name__}: {exc}")
    saved = st.session_state.get(f"{key_prefix}_result")
    if not saved or saved.get("context") != context:
        st.info("运行后才会给出可用于策略选择的样本外结论；上方样本内领先不视为推荐。"); return
    report = saved["report"]; best = report.get("best")
    if not best: st.warning("没有形成有效研究结果。"); return
    if best["gate_pass"]: st.success(f"研究验证通过：**{param_label} = {best['参数值']}** · 研究分 {best['research_score']:.1f}/100。")
    else: st.warning(f"没有参数通过全部研究门槛。当前最高为 **{param_label} = {best['参数值']}**，研究分 {best['research_score']:.1f}/100，仅列为继续观察。")
    with st.container(horizontal=True):
        st.metric("正超额折叠", f"{best['positive_fold_ratio']:.0%}", border=True); st.metric("中位超额", f"{best['median_alpha']:+.2f}%", border=True); st.metric("校正后置信度", f"{best['selection_confidence']:.1%}", border=True); st.metric("压力后收益", f"{best['stressed_return']:+.2f}%", border=True); st.metric("样本外交易", f"{best['total_trades']} 笔", border=True)
    ci_low, ci_high = best.get("annual_return_ci_low"), best.get("annual_return_ci_high")
    if best["total_trades"] > 0 and ci_low is not None and ci_high is not None: st.caption(f"块自助法年化平均收益 95% 区间：{ci_low:+.1f}% ～ {ci_high:+.1f}%")
    st.markdown(" ".join(f":{'green' if passed else 'red'}-badge[{'通过' if passed else '未通过'} · {label}]" for label, passed in best["checks"].items()))
    summary_rows = [{"参数值": row["参数值"], "结论": row["status"], "研究分": row["research_score"], "正超额折叠": row["positive_fold_ratio"], "中位夏普": row["median_sharpe"], "中位超额%": row["median_alpha"], "最差回撤%": row["worst_drawdown"], "样本外收益%": row["oos_return"], "压力后收益%": row["stressed_return"], "校正置信度": row["selection_confidence"], "交易数": row["total_trades"]} for row in report["rows"]]
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, column_config={"研究分": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"), "正超额折叠": st.column_config.NumberColumn(format="percent"), "中位超额%": st.column_config.NumberColumn(format="%+.2f%%"), "最差回撤%": st.column_config.NumberColumn(format="%.2f%%"), "样本外收益%": st.column_config.NumberColumn(format="%+.2f%%"), "压力后收益%": st.column_config.NumberColumn(format="%+.2f%%"), "校正置信度": st.column_config.NumberColumn(format="percent")})
    with st.expander("查看最高分参数的逐折结果"):
        folds = pd.DataFrame(report["fold_rows"]); st.dataframe(folds[folds["参数值"] == best["参数值"]], hide_index=True)
    st.caption("校正置信度会惩罚本次扫描的全部参数试验；研究通过只允许进入模拟盘连续验证，不等同于强烈买入或未来收益保证。")
