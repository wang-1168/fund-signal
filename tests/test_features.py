# -*- coding: utf-8 -*-
"""特征工程测试。

其中 ``test_no_lookahead_bias`` 是整个项目**最重要**的测试：
它验证「用前 N 天数据算出的特征」与「用前 N+k 天数据算出的特征」
在重叠区间上完全一致。只要有任何一处用到了未来信息，这个测试就会失败。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund_signal import features as ft


# ------------------------------------------------------------------ 基本结构
def test_price_features_columns(synthetic_nav):
    nav = synthetic_nav.set_index("date")["nav"]
    f = ft.compute_price_features(nav)
    for col in (
        "ret_1",
        "ret_20",
        "vol_20",
        "ma_gap_5",
        "rsi_14",
        "macd_hist",
        "bias_20",
        "max_dd_20",
        "up_days_10",
        "dow",
        "month",
    ):
        assert col in f.columns, f"缺少特征 {col}"
    assert len(f) == len(nav)


def test_rsi_within_range(synthetic_nav):
    nav = synthetic_nav.set_index("date")["nav"]
    rsi = ft.compute_price_features(nav)["rsi_14"].dropna()
    assert rsi.between(0.0, 1.0).all(), "归一化后的 RSI 必须落在 [0, 1]"


def test_calendar_features(synthetic_nav):
    nav = synthetic_nav.set_index("date")["nav"]
    f = ft.compute_price_features(nav)
    assert f["dow"].between(0, 4).all(), "工作日序列的 dow 应为 0~4"
    assert f["month"].between(1, 12).all()


# ------------------------------------------------------------------ 防未来函数
def test_no_lookahead_bias(synthetic_nav, synthetic_index):
    """核心测试：截断数据不能改变重叠区间的特征值。

    做法：分别用前 400 行和前 600 行数据算特征，前 400 行必须逐值相同。
    若某个滚动统计量误用了全样本信息（例如全样本标准化、居中窗口、
    或 ``shift`` 方向写反），这里立刻会暴露。
    """
    cut = 400
    full = ft.build_dataset(synthetic_nav, synthetic_index, horizon=1)
    part = ft.build_dataset(synthetic_nav.iloc[:cut], synthetic_index, horizon=1)

    common = part.frame.index.intersection(full.frame.index)
    assert len(common) > 100, "重叠样本太少，测试无意义"

    cols = [c for c in part.feature_names if c in full.frame.columns]
    a = part.frame.loc[common, cols]
    b = full.frame.loc[common, cols]

    pd.testing.assert_frame_equal(
        a,
        b,
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
        obj="截断后的特征与全量特征在重叠区间必须完全一致（否则存在未来函数）",
    )


def test_label_matches_definition(offline_nav, offline_index):
    """标签必须等于「T+1 日买入、持有 horizon 天」的收益方向。"""
    horizon = 1
    ds = ft.build_dataset(offline_nav, offline_index, horizon=horizon)

    nav = offline_nav.set_index("date")["nav"].sort_index()
    buy = nav.shift(-1)
    sell = nav.shift(-(1 + horizon))
    expect_ret = (sell / buy - 1.0).reindex(ds.frame.index)

    assert np.allclose(ds.frame["fwd_ret"].to_numpy(), expect_ret.to_numpy(), rtol=1e-9, atol=1e-12)

    expect_label = (expect_ret > 0).astype(int)
    assert (ds.frame["label"].astype(int) == expect_label).all()


def test_feature_set_alignment(dataset):
    """数据集里的特征名应与 config 中登记的清单一致（除市场特征缺失时）。"""
    names = set(dataset.feature_names)
    assert set(ft.PRICE_FEATURES).issubset(names), "价格类特征缺失"
    assert set(ft.CALENDAR_FEATURES).issubset(names), "日历类特征缺失"


# ------------------------------------------------------------------ 最新行
def test_latest_row_is_separate_from_frame(offline_nav, offline_index):
    """latest 必须比 frame 的最后一行更新，且标签尚未产生。"""
    ds = ft.build_dataset(offline_nav, offline_index, horizon=1)

    assert len(ds.latest) == 1, "应当恰好有一行最新特征"
    assert ds.latest.index[-1] > ds.frame.index[-1], "最新特征行必须严格晚于可用样本的最后一行"
    assert ds.latest[ds.feature_names].notna().all().all(), "最新特征行不应含缺失值"


def test_horizon_effect_on_sample_count(offline_nav, offline_index):
    """horizon 越大，尾部被剔除的行越多，样本数应单调不增。"""
    n1 = len(ft.build_dataset(offline_nav, offline_index, horizon=1).frame)
    n5 = len(ft.build_dataset(offline_nav, offline_index, horizon=5).frame)
    assert n5 <= n1
    assert n1 - n5 == 4, "horizon 每增加 1，样本应恰好减少 1 行"


# ------------------------------------------------------------------ 异常处理
def test_build_dataset_rejects_bad_horizon(synthetic_nav):
    with pytest.raises(ValueError):
        ft.build_dataset(synthetic_nav, None, horizon=0)


def test_feature_columns_excludes_meta(dataset):
    cols = ft.feature_columns(dataset.frame)
    for bad in ("label", "nav", "fwd_ret"):
        assert bad not in cols
