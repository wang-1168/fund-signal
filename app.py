# -*- coding: utf-8 -*-
"""
fund-signal · Streamlit 交互界面。

启动::

    streamlit run app.py

设计说明
--------
界面分两层，对应两种「交互成本」：

- **重型分析**（取数 → 特征 → 训练 → 滚动验证）结果会被缓存，
  同一组参数不会重复计算；
- **轻型调节**（阈值 / 交易成本 / 执行延迟）不重新训练，
  只用已有的样本外概率重算回测，因此可以做到拖动即刻刷新。

配色遵循中国市场的红涨绿跌惯例。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

from fund_signal.backtest import backtest_threshold, sweep_thresholds
from fund_signal.config import DEFAULT_BENCHMARKS, Config
from fund_signal.metrics import information_coefficient
from fund_signal.pipeline import run_analysis

# ------------------------------------------------------------------ 配色
UP = "#D94F4F"  # 涨 —— 红（中国市场惯例）
DOWN = "#2E9E5B"  # 跌 —— 绿
STRATEGY = "#D94F4F"
BENCHMARK = "#5B7FA6"
ACCENT = "#E8A33D"
NEUTRAL = "#8A8F98"

st.set_page_config(
    page_title="fund-signal · 基金量化信号",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ 工具
def plotly_theme() -> dict:
    """让图表背景透明并跟随 Streamlit 主题，深浅色都不糊。"""
    try:
        base = (st.get_option("theme.base") or "light").lower()
    except Exception:
        base = "light"
    dark = base == "dark"
    return dict(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6E6E6" if dark else "#2B2B2B", size=13),
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def apply_layout(fig, title: str = "", height: int = 380):
    fig.update_layout(title=title, height=height, **plotly_theme())
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


def pct(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    return f"{v * 100:.{digits}f}%"


def num(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


# ------------------------------------------------------------------ 缓存分析
@st.cache_data(show_spinner=False, ttl=900)
def cached_analysis(
    code: str,
    horizon: int,
    model_type: str,
    benchmark: str,
    n_splits: int,
    cost_bps: float,
    seed: int,
    data_source: str,
    init_ratio: float,
    use_cache: bool = True,
    realtime: bool = True,
):
    """带缓存的完整分析。参数全部为可哈希的基本类型。

    缓存 TTL 设 15 分钟：既避免每次交互都重新训练模型，
    又保证数据不会陈旧。「🔄 刷新数据」按钮会直接清空缓存并强制重新取数。
    """
    cfg = Config(
        fund_code=code,
        horizon=horizon,
        benchmark=benchmark,
        model_type=model_type,
        n_splits=n_splits,
        cost_bps=cost_bps,
        random_state=seed,
        train_ratio=init_ratio,
        use_cache=use_cache,
        realtime=realtime,
    )
    return run_analysis(cfg, data_source=data_source)


# ------------------------------------------------------------------ 侧边栏
def render_sidebar() -> dict:
    sb = st.sidebar
    sb.markdown("## ⚙️ 参数设置")

    sb.markdown("**① 标的**")
    code = sb.text_input(
        "基金代码", value="000001", help="6 位公募基金代码，例如 000001（华夏成长混合）"
    )

    bench_label = sb.selectbox(
        "基准指数", list(DEFAULT_BENCHMARKS.keys()), help="用于构造市场环境与相对强弱特征"
    )

    data_source = sb.radio(
        "数据来源",
        ["akshare", "local"],
        horizontal=True,
        help="akshare = 联网实时取数；local = 使用仓库自带离线样例（无需网络）",
    )

    sb.divider()
    sb.markdown("**② 预测任务**")
    horizon = sb.slider("预测周期（交易日）", 1, 20, 1, help="预测未来多少个交易日的涨跌方向")
    init_ratio = sb.slider(
        "初始训练集占比", 0.3, 0.8, 0.5, 0.05, help="滚动前向验证中，预留多少历史数据作为初始训练集"
    )

    sb.divider()
    sb.markdown("**③ 模型**")
    model_type = sb.selectbox(
        "模型类型",
        ["lightgbm", "logistic"],
        help="LightGBM 为梯度提升树；logistic 为带标准化的逻辑回归基线",
    )
    n_splits = sb.slider("滚动验证折数", 2, 10, 5)
    seed = sb.number_input("随机种子", value=42, step=1)

    sb.divider()
    sb.markdown("**④ 交易成本**")
    cost_bps = sb.slider(
        "单边成本（基点）",
        0.0,
        100.0,
        15.0,
        1.0,
        help="1 基点 = 0.01%。C 类基金无申购费但有销售服务费",
    )

    sb.divider()
    realtime = sb.toggle(
        "启用实时数据",
        value=True,
        help="开启后：把净值序列补齐到最新交易日，并尝试抓取盘中估值。"
        "盘中估值仅覆盖部分基金，且非交易时段无数据。",
    )

    c1, c2 = sb.columns(2)
    run = c1.button("🚀 开始分析", type="primary", width="stretch")
    refresh = c2.button("🔄 刷新数据", width="stretch", help="清空缓存，强制重新拉取最新数据")

    sb.caption(
        "⚠️ 本工具用于方法论演示与学习，输出为统计信号，"
        "**不构成任何投资建议**。基金净值短期走势接近随机。"
    )
    return dict(
        code=code,
        benchmark=DEFAULT_BENCHMARKS[bench_label],
        data_source=data_source,
        horizon=horizon,
        init_ratio=init_ratio,
        model_type=model_type,
        n_splits=n_splits,
        seed=int(seed),
        cost_bps=cost_bps,
        realtime=realtime,
        run=run,
        refresh=refresh,
    )


# ------------------------------------------------------------------ 欢迎页
def render_welcome():
    st.title("📈 fund-signal")
    st.markdown(
        "#### 公募基金量化信号分析框架\n从数据获取、特征工程到滚动前向验证与样本外回测的完整链路。"
    )

    c1, c2, c3 = st.columns(3)
    c1.info(
        "**① 严格时序验证**\n\n滚动前向 walk-forward，训练数据永远早于被预测样本，杜绝信息穿越。"
    )
    c2.info("**② 可交易收益**\n\n标签以 T+1 日净值为买入价，包含真实执行延迟，不凭空多赚一天。")
    c3.info(
        "**③ 诚实评估**\n\n同时输出 AUC、多数类基线与持仓日胜率，直接告诉你模型到底有没有优势。"
    )

    st.divider()
    st.warning(
        "**先读这段再动手。** 公募基金净值的日频方向预测，在学术与业界都缺少"
        "可靠证据支持（弱式有效市场）。本项目的价值在于**演示一套严谨的"
        "量化研究流程**，以及**展示'预测准确率略高于 50% 却依然亏钱'这一真实规律**，"
        "而不是提供一个能赚钱的策略。\n\n"
        "在左侧设置参数后点击 **🚀 开始分析**。"
    )
    st.caption("首次运行需要联网获取数据；也可将「数据来源」切为 local 使用离线样例。")


# ------------------------------------------------------------------ 数据新鲜度
def render_freshness(res):
    """数据新鲜度 + 实时估值面板。

    公募基金净值在每个交易日收盘后（约 20:00）才公布，因此这里的「最新」
    指的是最近一个**已公布净值**的交易日，而不是自然日。
    """
    last = res.last_nav_date
    if last is None:
        return

    age = res.data_age_days
    est = res.realtime_estimate
    meta = res.realtime.get("meta", {})

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        label = f"**数据截至**　{last.date()}"
        if age <= 1:
            st.success(f"{label}\n\n已是最新交易日")
        elif age <= 3:
            st.info(f"{label}\n\n（{int(age)} 天前）")
        else:
            st.warning(f"{label}\n\n已滞后 {int(age)} 天，建议点「🔄 刷新数据」")

    with c2:
        if est is not None:
            growth = est.get("est_growth")
            st.metric(
                "盘中实时估值",
                f"{est['est_nav']:.4f}",
                (f"{growth:+.2%}" if growth is not None else None),
                delta_color="normal" if (growth or 0) >= 0 else "inverse",
                help="第三方按基金持仓拟合，与最终公布净值存在偏差，仅供参考",
            )
        else:
            st.metric("盘中实时估值", "—")
            st.caption("数据源未覆盖该基金，或当前非交易时段")

    with c3:
        st.metric("申购状态", meta.get("purchase_status", "—"))

    with c4:
        st.metric("申购费率", meta.get("fee", "—"))

    st.caption(
        f"本地取数时间 {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}　|　"
        "净值公布时间：每个交易日约 20:00 之后"
    )


# ------------------------------------------------------------------ 概览
def render_overview(res):
    m = res.summary_metrics()
    nav = res.nav

    st.markdown(f"### {res.fund_name}　`{res.config.fund_code}`")
    st.caption(
        f"净值区间 {nav['date'].iloc[0].date()} ~ {nav['date'].iloc[-1].date()}"
        f"（{len(nav)} 个交易日）　|　"
        f"样本 {m['n_samples']} 行 × {m['n_features']} 特征　|　"
        f"预测周期 {res.config.horizon} 个交易日"
    )

    render_freshness(res)

    # ---- 最新信号 ----
    st.markdown("#### 🎯 最新信号")
    c1, c2, c3, c4 = st.columns(4)

    prob = res.latest_prob
    if res.latest_date is not None and not np.isnan(prob):
        verdict = "偏多" if prob >= res.config.threshold else "偏空 / 观望"
        c1.metric("上涨概率", f"{prob:.1%}", delta=verdict, delta_color="off")
        c2.metric("特征日期", str(pd.Timestamp(res.latest_date).date()))
    else:
        c1.metric("上涨概率", "—")
        c2.metric("特征日期", "—")

    c3.metric(
        "样本外 AUC",
        num(m["auc"]),
        delta=f"超额 {num(m['acc_edge'], 4)}",
        delta_color="normal" if res.has_edge else "inverse",
    )
    c4.metric("持仓时间占比", pct(m["exposure"], 1))

    # ---- 结论提示 ----
    if res.has_edge:
        st.success(
            f"模型准确率 {pct(m['accuracy'])} 高于多数类基线 {pct(m['majority_acc'])}，"
            f"超额 **{num(m['acc_edge'], 4)}**——存在微弱但可测的统计信号。"
            "注意：微弱信号通常不足以覆盖交易成本，请看回测页的净结果。"
        )
    else:
        st.warning(
            f"模型准确率 {pct(m['accuracy'])} **未明显超过**多数类基线 "
            f"{pct(m['majority_acc'])}（超额 {num(m['acc_edge'], 4)}）。"
            "这说明该基金在所选周期上的方向变化接近随机——**这是最常见、也最诚实的结果**。"
        )

    for note in res.notes:
        st.caption(f"· {note}")

    # ---- 净值曲线 ----
    st.markdown("#### 📉 净值走势")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nav["date"],
            y=nav["nav"],
            mode="lines",
            name="单位净值",
            line=dict(color=BENCHMARK, width=2),
            fill="tozeroy",
            fillcolor="rgba(91,127,166,0.10)",
        )
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.4f}<extra></extra>")
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.06)
    st.plotly_chart(apply_layout(fig, "", 420), width="stretch")


# ------------------------------------------------------------------ 预测
def render_prediction(res):
    wf = res.wf
    prob = wf.predictions
    frame = res.dataset.frame

    st.markdown("#### 样本外预测概率序列")
    st.caption(
        "以下概率全部来自滚动前向验证——每个点都是「用过去预测未来」的结果，"
        "不含任何训练集内的自预测。"
    )

    y_true = frame.loc[prob.index, "label"].astype(int)
    fwd = frame.loc[prob.index, "fwd_ret"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prob.index,
            y=prob.values,
            mode="lines",
            name="上涨概率",
            line=dict(color=ACCENT, width=1.6),
        )
    )
    fig.add_hline(
        y=res.config.threshold, line_dash="dash", line_color=NEUTRAL, annotation_text="阈值"
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(128,128,128,0.4)")
    fig.update_traces(hovertemplate="%{x|%Y-%m-%d}<br>概率 %{y:.3f}<extra></extra>")
    st.plotly_chart(apply_layout(fig, "", 360), width="stretch")

    c1, c2 = st.columns(2)

    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=prob.values, nbinsx=40, marker_color=ACCENT, opacity=0.85))
        st.plotly_chart(apply_layout(fig2, "预测概率分布", 320), width="stretch")
        st.caption("分布越集中在 0.5 附近，说明模型越「不敢表态」——这与弱信号是一致的。")

    with c2:
        st.markdown("**概率分组的实际涨跌表现**")
        st.caption("把预测概率分成 5 档，看每档真实的上涨比例。若模型有效，应呈单调上升。")
        bins = pd.qcut(prob, 5, labels=False, duplicates="drop")
        grp = (
            pd.DataFrame({"bin": bins, "y": y_true.values})
            .groupby("bin")["y"]
            .agg(["mean", "count"])
        )
        grp.index = [f"Q{i + 1}" for i in range(len(grp))]
        colors = [UP if v >= y_true.mean() else DOWN for v in grp["mean"]]
        fig3 = go.Figure(
            go.Bar(
                x=grp.index,
                y=grp["mean"],
                marker_color=colors,
                text=[
                    f"{v:.1%}<br>n={int(n)}" for v, n in zip(grp["mean"], grp["count"], strict=True)
                ],
                textposition="outside",
            )
        )
        fig3.add_hline(
            y=float(y_true.mean()),
            line_dash="dash",
            line_color=NEUTRAL,
            annotation_text="整体上涨率",
        )
        fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(apply_layout(fig3, "", 320), width="stretch")

    ic = information_coefficient(prob, fwd)
    st.metric(
        "信息系数 IC（概率 vs 未来收益的秩相关）",
        num(ic, 4),
        help="不依赖阈值的诚实指标。|IC| < 0.03 基本等同于噪音",
    )
    st.caption(
        f"当前 IC = {num(ic, 4)}。"
        + (
            "这个量级意味着预测与真实收益几乎没有单调关系。"
            if abs(ic) < 0.03
            else "存在一定的单调关系，但仍需扣除交易成本后再判断。"
        )
    )


# ------------------------------------------------------------------ 模型评估
def render_model(res):
    wf = res.wf
    frame = res.dataset.frame
    prob = wf.predictions
    y_true = frame.loc[prob.index, "label"].astype(int)

    st.markdown("#### 滚动前向验证 · 各折表现")
    folds = pd.DataFrame(wf.fold_metrics)
    if not folds.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=folds["fold"].astype(str),
                y=folds["auc"],
                marker_color=[UP if a >= 0.5 else DOWN for a in folds["auc"]],
                text=[f"{a:.3f}" for a in folds["auc"]],
                textposition="outside",
                name="AUC",
            )
        )
        fig.add_hline(
            y=0.5, line_dash="dash", line_color=NEUTRAL, annotation_text="0.5 = 无预测能力"
        )
        fig.update_yaxes(range=[0.4, max(0.65, folds["auc"].max() + 0.05)])
        st.plotly_chart(apply_layout(fig, "", 340), width="stretch")

        show = folds[
            [
                "fold",
                "train_end",
                "test_start",
                "test_end",
                "n_train",
                "n_test",
                "auc",
                "accuracy",
                "majority_acc",
                "acc_edge",
            ]
        ].copy()
        st.dataframe(
            show.style.format(
                {
                    "auc": "{:.4f}",
                    "accuracy": "{:.4f}",
                    "majority_acc": "{:.4f}",
                    "acc_edge": "{:+.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**ROC 曲线（样本外合并）**")
        try:
            fpr, tpr, _ = roc_curve(y_true, prob)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"AUC = {res.auc:.4f}",
                    line=dict(color=ACCENT, width=2.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="随机猜测",
                    line=dict(color=NEUTRAL, dash="dash"),
                )
            )
            fig.update_xaxes(title="假正率")
            fig.update_yaxes(title="真正率")
            st.plotly_chart(apply_layout(fig, "", 360), width="stretch")
        except ValueError as err:
            st.info(f"无法绘制 ROC：{err}")

    with c2:
        st.markdown("**特征重要性**")
        if len(res.importance):
            top = res.importance.head(15)[::-1]
            fig = go.Figure(
                go.Bar(
                    x=top.values,
                    y=top.index,
                    orientation="h",
                    marker_color=BENCHMARK,
                    hovertemplate="%{y}<br>重要性 %{x:.3f}<extra></extra>",
                )
            )
            st.plotly_chart(apply_layout(fig, "", 360), width="stretch")
        else:
            st.info("当前模型不支持特征重要性输出。")

    # 混淆矩阵
    st.markdown(f"**混淆矩阵（阈值 = {res.config.threshold:.2f}）**")
    pred = (prob >= res.config.threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fig = go.Figure(
        go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["预测跌", "预测涨"],
            y=["实际跌", "实际涨"],
            text=[[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]],
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    st.plotly_chart(apply_layout(fig, "", 300), width="stretch")


# ------------------------------------------------------------------ 回测
def render_backtest(res):
    wf = res.wf
    frame = res.dataset.frame
    prob = wf.predictions
    fwd = frame.loc[prob.index, "fwd_ret"]

    st.markdown("#### 交互回测")
    st.caption("以下三个滑块**不会重新训练模型**，只重算回测，因此可以实时观察敏感度。")

    c1, c2, c3 = st.columns(3)
    thr = c1.slider("看多阈值", 0.30, 0.70, float(res.config.threshold), 0.01, key="bt_thr")
    cost = c2.slider("单边成本（基点）", 0.0, 100.0, float(res.config.cost_bps), 1.0, key="bt_cost")
    lag = c3.slider(
        "额外执行延迟（交易日）",
        0,
        5,
        0,
        1,
        key="bt_lag",
        help="0 = 信号次日按净值成交（已内含在收益定义中）；调大可做「执行更慢」的压力测试",
    )

    bt = backtest_threshold(prob, fwd, threshold=thr, cost_bps=cost, execution_lag=lag)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("策略累计收益", pct(bt.strategy_metrics["total_return"], 1))
    m2.metric("买入持有", pct(bt.benchmark_metrics["total_return"], 1))
    m3.metric(
        "超额收益",
        pct(bt.excess_return, 1),
        delta_color="normal" if bt.excess_return > 0 else "inverse",
    )
    m4.metric("最大回撤", pct(bt.strategy_metrics["max_drawdown"], 1))
    m5.metric(
        "持仓日胜率",
        pct(bt.win_rate_active, 1),
        help="只统计有仓位日子的胜率，比全样本胜率更能反映择时能力",
    )

    # ---- 净值曲线 + 回撤 ----
    eq = pd.DataFrame(
        {
            "策略": (1 + bt.daily["strategy_ret"]).cumprod(),
            "买入持有": (1 + bt.daily["benchmark_ret"]).cumprod(),
        }
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=("净值曲线（起点归一）", "策略回撤"),
    )
    fig.add_trace(
        go.Scatter(x=eq.index, y=eq["策略"], name="策略", line=dict(color=STRATEGY, width=2.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index, y=eq["买入持有"], name="买入持有", line=dict(color=BENCHMARK, width=2.2)
        ),
        row=1,
        col=1,
    )
    dd = eq["策略"] / eq["策略"].cummax() - 1
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="回撤",
            line=dict(color=DOWN, width=1.4),
            fill="tozeroy",
            fillcolor="rgba(46,158,91,0.20)",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(apply_layout(fig, "", 520), width="stretch")

    # ---- 指标对照表 ----
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**绩效指标对照**")
        st.dataframe(bt.summary(), width="stretch")

    with c2:
        st.markdown("**阈值敏感度**")
        st.caption("若只有某一个阈值表现突出、相邻阈值立刻崩掉，多半是过拟合。")
        sw = sweep_thresholds(prob, fwd, cost_bps=cost, execution_lag=lag)
        if not sw.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=sw["threshold"],
                    y=sw["total_return"],
                    mode="lines+markers",
                    name="策略累计收益",
                    line=dict(color=STRATEGY, width=2.2),
                )
            )
            fig.add_hline(
                y=float(bt.benchmark_metrics["total_return"]),
                line_dash="dash",
                line_color=BENCHMARK,
                annotation_text="买入持有",
            )
            fig.add_hline(y=0, line_color="rgba(128,128,128,0.4)")
            fig.update_xaxes(title="看多阈值")
            fig.update_yaxes(title="累计收益", tickformat=".0%")
            st.plotly_chart(apply_layout(fig, "", 330), width="stretch")

    # ---- 月度收益 ----
    st.markdown("**策略月度收益**")
    monthly = (1 + bt.daily["strategy_ret"]).resample("ME").prod() - 1
    if len(monthly) > 3:
        md_ = pd.DataFrame({"y": monthly.index.year, "m": monthly.index.month, "r": monthly.values})
        piv = md_.pivot(index="y", columns="m", values="r")
        fig = go.Figure(
            go.Heatmap(
                z=piv.values,
                x=[f"{c}月" for c in piv.columns],
                y=piv.index.astype(str),
                colorscale=[[0, DOWN], [0.5, "#F2F2F2"], [1, UP]],
                zmid=0,
                colorbar=dict(title="收益"),
                hovertemplate="%{y}年%{x}<br>%{z:.2%}<extra></extra>",
            )
        )
        st.plotly_chart(apply_layout(fig, "", 340), width="stretch")


# ------------------------------------------------------------------ 数据探索
def render_data(res):
    nav = res.nav
    frame = res.dataset.frame

    st.markdown("#### 净值与收益特征")

    df = nav.copy()
    df["日收益率"] = df["nav"].pct_change()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(
            go.Histogram(x=df["日收益率"].dropna(), nbinsx=80, marker_color=BENCHMARK, opacity=0.85)
        )
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(apply_layout(fig, "日收益率分布", 330), width="stretch")
    with c2:
        roll = df.set_index("date")["日收益率"].rolling(60).std() * np.sqrt(244)
        fig = go.Figure(
            go.Scatter(
                x=roll.index,
                y=roll,
                mode="lines",
                line=dict(color=ACCENT, width=1.6),
                fill="tozeroy",
                fillcolor="rgba(232,163,61,0.15)",
            )
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(apply_layout(fig, "滚动 60 日年化波动率", 330), width="stretch")

    st.markdown("#### 特征相关性")
    st.caption("高度相关的特征会让树模型的重要性被稀释，这是特征工程的常见观察点。")
    X = frame[res.feature_names]
    corr = X.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="相关系数"),
        )
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(apply_layout(fig, "", 620), width="stretch")


# ------------------------------------------------------------------ 原始数据
def render_raw(res):
    st.markdown("#### 特征与标签数据集")
    st.caption(
        "可直接下载用于自己的建模实验。注意 `fwd_ret` 已包含执行延迟（以 T+1 日净值为买入价）。"
    )
    frame = res.dataset.frame
    st.dataframe(frame.tail(300), width="stretch")

    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ 下载完整数据集 (CSV)",
        data=frame.to_csv().encode("utf-8-sig"),
        file_name=f"fund_signal_{res.config.fund_code}_dataset.csv",
        mime="text/csv",
        width="stretch",
    )
    wf_tbl = res.wf.predictions.to_frame("prob").join(frame[["label", "fwd_ret", "nav"]])
    c2.download_button(
        "⬇️ 下载样本外预测 (CSV)",
        data=wf_tbl.to_csv().encode("utf-8-sig"),
        file_name=f"fund_signal_{res.config.fund_code}_oos.csv",
        mime="text/csv",
        width="stretch",
    )

    st.divider()
    st.markdown("#### 本次运行的参数")
    st.json(res.config.to_dict())


# ------------------------------------------------------------------ 主流程
def main():
    st.title("📈 fund-signal")
    st.caption("公募基金量化信号分析框架　·　输出为统计信号，不构成投资建议")

    params = render_sidebar()
    run_clicked = params.pop("run")
    refresh_clicked = params.pop("refresh")

    if refresh_clicked:
        # 同时绕过两级缓存：Streamlit 内存缓存 + 本地 CSV 文件缓存
        st.cache_data.clear()
        params["use_cache"] = False
        st.toast("已清空缓存，正在拉取最新数据…", icon="🔄")

    if run_clicked or refresh_clicked:
        try:
            with st.spinner("正在取数、构建特征、训练模型并做滚动前向验证…"):
                result = cached_analysis(**params)
            st.session_state["result"] = result
            st.session_state["params"] = params
        except ImportError as err:
            st.error(f"依赖缺失：\n\n```\n{err}\n```")
            st.stop()
        except Exception as err:
            st.error(f"分析失败：{err}")
            st.info(
                "常见原因：\n"
                "- 基金代码有误（应为 6 位数字）\n"
                "- 该基金历史过短，样本不足以做时序验证\n"
                "- 网络不通导致取数失败（可把数据来源切为 `local` 试试）"
            )
            st.stop()

    result = st.session_state.get("result")
    if result is None:
        render_welcome()
        return

    tabs = st.tabs(
        ["📊 概览", "🎯 预测详情", "🧪 模型评估", "💰 回测", "🔍 数据探索", "📋 原始数据"]
    )
    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_prediction(result)
    with tabs[2]:
        render_model(result)
    with tabs[3]:
        render_backtest(result)
    with tabs[4]:
        render_data(result)
    with tabs[5]:
        render_raw(result)

    st.divider()
    st.caption(
        "**免责声明**　本项目为量化研究方法论演示，所有输出均为基于历史数据的统计结果，"
        "不构成投资建议。基金净值短期走势接近随机，回测表现不代表未来收益。"
        "投资有风险，决策需谨慎。"
    )


if __name__ == "__main__":
    main()
else:
    main()
