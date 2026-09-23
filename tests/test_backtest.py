# -*- coding: utf-8 -*-
"""回测测试。

重点验证两件事：
1. 收益对齐正确（信号不会赚到"已经发生的行情"）；
2. 交易成本真的被扣掉，且成本越高收益越低。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund_signal import backtest as bt
from fund_signal.metrics import equity_curve, performance_metrics


@pytest.fixture(scope="module")
def toy():
    """构造一段确定性行情，便于手算校验。"""
    idx = pd.bdate_range("2022-01-03", periods=12)
    ret = pd.Series(
        [0.01, -0.01, 0.02, -0.02, 0.01, 0.03, -0.01, 0.02, -0.02, 0.01, 0.02, -0.01], index=idx
    )
    prob = pd.Series([0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.9], index=idx)
    return prob, ret


# ------------------------------------------------------------------ 仓位与延迟
def test_position_follows_signal(toy):
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, threshold=0.5, cost_bps=0.0, execution_lag=0)
    expect = (prob >= 0.5).astype(float)
    pd.testing.assert_series_equal(r.daily["position"], expect, check_names=False)


def test_execution_lag_shifts_position(toy):
    prob, ret = toy
    r1 = bt.backtest_threshold(prob, ret, threshold=0.5, cost_bps=0.0, execution_lag=1)
    assert r1.daily["position"].iloc[0] == 0.0, "延迟 1 日时首日必然空仓"
    assert r1.daily["position"].iloc[1] == pytest.approx((prob >= 0.5).astype(float).iloc[0])


def test_strategy_return_equals_position_times_forward(toy):
    """策略收益 = 仓位 × 可交易收益（零成本时）。"""
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, threshold=0.5, cost_bps=0.0, execution_lag=0)
    expect = r.daily["position"] * r.daily["benchmark_ret"]
    pd.testing.assert_series_equal(r.daily["strategy_ret"], expect, check_names=False)


def test_missing_first_signal_does_not_leak(toy):
    """延迟生效时，未被任何信号覆盖的第一天不能凭空产生收益。"""
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, threshold=0.5, cost_bps=0.0, execution_lag=2)
    assert r.daily["position"].iloc[0] == 0.0
    assert r.daily["strategy_ret"].iloc[0] == 0.0


# ------------------------------------------------------------------ 交易成本
def test_cost_reduces_return(toy):
    prob, ret = toy
    no_cost = bt.backtest_threshold(prob, ret, cost_bps=0.0)
    high_cost = bt.backtest_threshold(prob, ret, cost_bps=100.0)
    assert high_cost.strategy_metrics["total_return"] < no_cost.strategy_metrics["total_return"], (
        "更高的交易成本必须拉低收益"
    )


def test_cost_charged_only_on_position_change(toy):
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, cost_bps=50.0)
    changed = r.daily["position"].diff().abs().fillna(r.daily["position"].abs())
    # 未发生仓位变动的日子不应计费
    assert (r.daily.loc[changed == 0, "cost"] == 0).all()
    assert (r.daily.loc[changed > 0, "cost"] > 0).all()


def test_zero_cost_keeps_full_return(toy):
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, cost_bps=0.0)
    assert r.daily["cost"].sum() == pytest.approx(0.0)


# ------------------------------------------------------------------ 基准与指标
def test_benchmark_equals_buy_and_hold(toy):
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, cost_bps=15.0)
    pd.testing.assert_series_equal(r.daily["benchmark_ret"], ret, check_names=False)


def test_all_in_position_matches_benchmark(toy):
    """若信号恒为看多，策略（零成本）应与买入持有完全一致。"""
    _, ret = toy
    all_long = pd.Series(1.0, index=ret.index)
    r = bt.backtest_threshold(all_long, ret, threshold=0.5, cost_bps=0.0)
    assert r.strategy_metrics["total_return"] == pytest.approx(
        r.benchmark_metrics["total_return"], rel=1e-9
    )


def test_exposure_and_win_rate_range(toy):
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, cost_bps=15.0)
    assert 0.0 <= r.exposure <= 1.0
    wr = r.win_rate_active
    assert np.isnan(wr) or 0.0 <= wr <= 1.0


def test_summary_table_shape(toy):
    prob, ret = toy
    r = bt.backtest_threshold(prob, ret, cost_bps=15.0)
    s = r.summary()
    assert list(s.columns) == ["策略", "买入持有"]
    assert "累计收益" in s.index


# ------------------------------------------------------------------ 阈值扫描
def test_sweep_thresholds(toy):
    prob, ret = toy
    sw = bt.sweep_thresholds(prob, ret, thresholds=np.array([0.4, 0.5, 0.6]))
    assert len(sw) == 3
    assert {"threshold", "total_return", "exposure"}.issubset(sw.columns)
    # 阈值越高，持仓时间应越短（单调不增）
    assert sw["exposure"].is_monotonic_decreasing


# ------------------------------------------------------------------ 绩效指标
def test_performance_metrics_on_flat_series():
    r = performance_metrics(pd.Series([0.0] * 30))
    assert r["total_return"] == pytest.approx(0.0)
    assert r["max_drawdown"] == pytest.approx(0.0)


def test_max_drawdown_is_negative():
    r = performance_metrics(pd.Series([0.05, -0.10, 0.02, -0.05, 0.01]))
    assert r["max_drawdown"] <= 0


def test_equity_curve_starts_at_one():
    eq = equity_curve(pd.Series([0.01, 0.02, -0.01]))
    assert eq.iloc[0] == pytest.approx(1.01)
    assert len(eq) == 3


def test_bad_threshold_raises(toy):
    prob, ret = toy
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            bt.backtest_threshold(prob, ret, threshold=bad)
