# -*- coding: utf-8 -*-
"""
回测。

执行时序假设（很重要，决定了回测是否可信）
------------------------------------------
公募基金的实际交易流程：

1. 交易日 T 晚上，基金公司公布 T 日的单位净值；
2. 投资者在 T+1 日 15:00 前提交申购/赎回，**按 T+1 日净值成交**；
3. 因此 T 日的信号，只能赚到「nav[T+1] → nav[T+1+horizon]」这一段行情。

关键点：**执行延迟已经内含在 ``fwd_ret`` 里**
（见 :func:`fund_signal.features.build_dataset`，它以 ``nav[T+1]`` 为买入价）。
因此本模块默认 ``execution_lag=0``——信号所在的行，直接对应它预测的那段收益，
二者严格对齐，不会重复滞后，也不会凭空多赚一天。

``execution_lag`` 参数保留下来是做**压力测试**用的：把它设为 1 或 2，
可以量化「信号执行越迟缓，策略退化多少」。

同时默认计入 ``cost_bps`` 单边交易成本。基金 C 类免申购费但有销售服务费，
A 类申购费打折后约 0.15%，赎回费与持有期挂钩。忽略成本的回测没有意义。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import ANNUAL_FACTOR, equity_curve, performance_metrics
from .utils import get_logger

log = get_logger()

BPS = 1e-4  # 1 个基点 = 0.01% = 1e-4


@dataclass
class BacktestResult:
    """回测结果。

    Attributes
    ----------
    daily:
        日度明细表，含 ``prob``（预测概率）、``position``（实际持仓）、
        ``strategy_ret``（策略日收益，已扣费）、``benchmark_ret``（买入持有日收益）。
    strategy_metrics / benchmark_metrics:
        两套绩效指标，用于横向对比。
    """

    daily: pd.DataFrame
    strategy_metrics: dict = field(default_factory=dict)
    benchmark_metrics: dict = field(default_factory=dict)
    threshold: float = 0.5
    cost_bps: float = 15.0
    execution_lag: int = 0

    @property
    def excess_return(self) -> float:
        """相对买入持有的累计超额收益。"""
        s = self.strategy_metrics.get("total_return", float("nan"))
        b = self.benchmark_metrics.get("total_return", float("nan"))
        if np.isnan(s) or np.isnan(b):
            return float("nan")
        return float(s - b)

    @property
    def exposure(self) -> float:
        """持仓时间占比。"""
        if self.daily.empty:
            return float("nan")
        return float((self.daily["position"] > 0).mean())

    @property
    def win_rate_active(self) -> float:
        """**持仓期间**的日胜率。

        不要用 ``strategy_metrics['win_rate']`` 评价择时能力——那个指标把
        空仓日（收益恰好为 0）也算进分母，会系统性拉低胜率读数。
        """
        if self.daily.empty:
            return float("nan")
        active = self.daily["position"] > 0
        if not bool(active.any()):
            return float("nan")
        return float((self.daily.loc[active, "strategy_ret"] > 0).mean())

    def summary(self) -> pd.DataFrame:
        """生成可读的指标对照表。"""
        rows = [
            ("累计收益", "total_return", "{:.2%}"),
            ("年化收益", "annual_return", "{:.2%}"),
            ("年化波动", "annual_vol", "{:.2%}"),
            ("夏普比率", "sharpe", "{:.2f}"),
            ("最大回撤", "max_drawdown", "{:.2%}"),
            ("卡玛比率", "calmar", "{:.2f}"),
            ("日胜率(全样本)", "win_rate", "{:.2%}"),
        ]
        data = {}
        for label, key, fmt in rows:
            data[label] = {
                "策略": _fmt(self.strategy_metrics.get(key), fmt),
                "买入持有": _fmt(self.benchmark_metrics.get(key), fmt),
            }
        return pd.DataFrame(data).T


def _fmt(v, fmt: str) -> str:
    try:
        if v is None or np.isnan(v):
            return "—"
        return fmt.format(v)
    except (TypeError, ValueError):
        return "—"


def backtest_threshold(
    prob: pd.Series,
    forward_return: pd.Series,
    threshold: float = 0.5,
    cost_bps: float = 15.0,
    execution_lag: int = 0,
    long_only: bool = True,
) -> BacktestResult:
    """按概率阈值执行的择时策略回测。

    规则
    ----
    - ``prob >= threshold`` → 持有（仓位 1.0）；否则空仓（仓位 0.0）。
    - ``long_only=False`` 时，空头一侧做反向持仓（仓位 -1.0），
      但请注意公募基金**无法直接做空**，该模式仅用于研究分析。
    - 仓位在 ``execution_lag`` 个交易日后生效（默认 0，见模块 docstring）。
    - 每次仓位变化按 ``cost_bps`` 单边收取交易成本。

    Parameters
    ----------
    prob:
        预测的上涨概率，索引为日期。
    forward_return:
        与之对应的**可交易收益**（与 ``prob`` 同索引，取自
        :func:`fund_signal.features.build_dataset` 的 ``fwd_ret`` 列）。
    threshold:
        看多阈值。
    cost_bps:
        单边交易成本（基点）。
    execution_lag:
        额外执行延迟（交易日）。默认 0，因为 ``forward_return`` 本身
        已经是从「可买入日」起算的收益。设为 1 或 2 可做压力测试。
    long_only:
        是否只做多。公募基金现实中应保持 True。

    Returns
    -------
    BacktestResult
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold 必须在 (0, 1) 之间，收到 {threshold}")

    df = pd.DataFrame(
        {
            "prob": pd.Series(prob).astype(float),
            "fwd_ret": pd.Series(forward_return).astype(float),
        }
    ).dropna()
    df = df.sort_index()

    if df.empty:
        raise ValueError("回测输入为空，请检查 prob 与 forward_return 的对齐情况。")

    # --- 目标仓位（依据 T 日信号） ---
    signal = (df["prob"] >= threshold).astype(float)
    if not long_only:
        signal = signal.replace(0.0, -1.0)

    # --- 延迟生效：T 日信号 → T+lag 日持仓 ---
    position = signal.shift(execution_lag).fillna(0.0)

    # --- 交易成本：仅在仓位变动时产生 ---
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    cost = turnover * cost_bps * BPS

    strategy_ret = position * df["fwd_ret"] - cost
    benchmark_ret = df["fwd_ret"]  # 买入持有的日收益

    daily = pd.DataFrame(
        {
            "prob": df["prob"],
            "position": position,
            "turnover": turnover,
            "cost": cost,
            "strategy_ret": strategy_ret,
            "benchmark_ret": benchmark_ret,
        }
    )

    result = BacktestResult(
        daily=daily,
        strategy_metrics=performance_metrics(strategy_ret),
        benchmark_metrics=performance_metrics(benchmark_ret),
        threshold=threshold,
        cost_bps=cost_bps,
        execution_lag=execution_lag,
    )

    log.info(
        "回测完成 | 阈值=%.2f 成本=%.0fbps 延迟=%d日 | 策略累计 %.2f%% vs 持有 %.2f%% | 持仓占比 %.1f%%",
        threshold,
        cost_bps,
        execution_lag,
        result.strategy_metrics["total_return"] * 100,
        result.benchmark_metrics["total_return"] * 100,
        result.exposure * 100,
    )
    return result


def sweep_thresholds(
    prob: pd.Series,
    forward_return: pd.Series,
    thresholds: np.ndarray | None = None,
    cost_bps: float = 15.0,
    execution_lag: int = 0,
) -> pd.DataFrame:
    """遍历多个阈值，返回指标对照表。

    用于观察策略对阈值的敏感度——若某个阈值上表现特别好、换个阈值就崩，
    那多半是过拟合，而非真实规律。
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.35, 0.66, 0.05), 2)

    rows = []
    for t in thresholds:
        try:
            r = backtest_threshold(
                prob,
                forward_return,
                threshold=float(t),
                cost_bps=cost_bps,
                execution_lag=execution_lag,
            )
        except ValueError:
            continue
        sm, bm = r.strategy_metrics, r.benchmark_metrics
        rows.append(
            {
                "threshold": float(t),
                "exposure": r.exposure,
                "total_return": sm["total_return"],
                "annual_return": sm["annual_return"],
                "sharpe": sm["sharpe"],
                "max_drawdown": sm["max_drawdown"],
                "n_trades": sm["n_trades"],
                "excess_vs_hold": sm["total_return"] - bm["total_return"],
            }
        )
    return pd.DataFrame(rows)


def build_equity_frame(result: BacktestResult) -> pd.DataFrame:
    """构造净值曲线对照表（策略 vs 买入持有），用于绘图。"""
    return pd.DataFrame(
        {
            "策略": equity_curve(result.daily["strategy_ret"]),
            "买入持有": equity_curve(result.daily["benchmark_ret"]),
        }
    )


__all__ = [
    "BacktestResult",
    "backtest_threshold",
    "sweep_thresholds",
    "build_equity_frame",
    "ANNUAL_FACTOR",
]
