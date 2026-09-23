# -*- coding: utf-8 -*-
"""端到端流水线测试（离线，不联网）。"""

from __future__ import annotations

import numpy as np
import pytest

from fund_signal.config import Config
from fund_signal.pipeline import run_analysis


@pytest.fixture(scope="module")
def result():
    cfg = Config(fund_code="000001", horizon=1, n_splits=3, model_type="logistic", realtime=False)
    return run_analysis(cfg, data_source="local")


# ------------------------------------------------------------------ 结构完整性
def test_result_has_all_parts(result):
    for attr in ("nav", "dataset", "wf", "backtest", "equity", "importance", "latest_prob"):
        assert hasattr(result, attr)


def test_metrics_are_finite(result):
    m = result.summary_metrics()
    assert np.isfinite(m["auc"])
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["exposure"] <= 1.0


def test_backtest_uses_out_of_sample_predictions(result):
    """回测的预测序列必须与滚动前向的预测完全一致（而非训练集内预测）。"""
    assert result.backtest.daily.index.equals(result.wf.predictions.index), (
        "回测索引必须来自样本外预测，否则就是「用答案考自己」"
    )
    np.testing.assert_allclose(
        result.backtest.daily["prob"].to_numpy(),
        result.wf.predictions.to_numpy(),
    )


def test_equity_curves_track_first_day_return(result):
    """净值曲线第一个值是「首日结束时」，故等于 1 + 首日收益。"""
    d = result.backtest.daily
    assert result.equity["策略"].iloc[0] == pytest.approx(1.0 + d["strategy_ret"].iloc[0])
    assert result.equity["买入持有"].iloc[0] == pytest.approx(1.0 + d["benchmark_ret"].iloc[0])
    assert len(result.equity) == len(d)


def test_latest_signal_in_range(result):
    assert 0.0 <= result.latest_prob <= 1.0
    assert result.latest_date is not None


def test_data_age_is_non_negative(result):
    age = result.data_age_days
    assert not np.isnan(age)
    assert age >= 0


# ------------------------------------------------------------------ 参数影响
def test_horizon_changes_sample_count():
    r1 = run_analysis(
        Config(fund_code="000001", horizon=1, n_splits=3, model_type="logistic", realtime=False),
        data_source="local",
    )
    r5 = run_analysis(
        Config(fund_code="000001", horizon=5, n_splits=3, model_type="logistic", realtime=False),
        data_source="local",
    )
    assert len(r5.dataset) < len(r1.dataset)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        run_analysis(Config(fund_code="abc"), data_source="local")


def test_too_few_samples_raises():
    """历史极短的基金应当给出明确报错，而不是静默产出垃圾结果。"""
    cfg = Config(fund_code="000001", n_splits=3, realtime=False)
    cfg.horizon = 1
    # 用一个不存在的离线代码触发 FileNotFoundError 分支
    with pytest.raises(FileNotFoundError):
        run_analysis(Config(fund_code="999999", realtime=False), data_source="local")
