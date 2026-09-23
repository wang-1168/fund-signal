# -*- coding: utf-8 -*-
"""模型与时序验证测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund_signal import model as md


# ------------------------------------------------------------------ 时序切分
def test_train_test_split_is_chronological(dataset):
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)

    X_tr, X_te, y_tr, y_te = md.train_test_split_ts(X, y, train_ratio=0.7)

    assert X_tr.index.max() < X_te.index.min(), (
        "训练集的最后一天必须严格早于测试集的第一天——这是时序安全的底线"
    )
    assert len(X_tr) + len(X_te) == len(X)
    assert list(X_tr.index) + list(X_te.index) == list(X.index), "顺序不能被打乱"


def test_split_rejects_insufficient_samples(synthetic_nav):
    X = pd.DataFrame({"a": range(30)}, index=pd.bdate_range("2020-01-01", periods=30))
    y = pd.Series([0, 1] * 15, index=X.index)
    with pytest.raises(ValueError, match="样本量不足"):
        md.train_test_split_ts(X, y, train_ratio=0.7)


# ------------------------------------------------------------------ 模型工厂
@pytest.mark.parametrize("model_type", ["lightgbm", "logistic"])
def test_build_and_fit(model_type, dataset):
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)

    model = md.build_model(model_type, random_state=0)
    model.fit(X.iloc[:400], y.iloc[:400])

    prob = model.predict_proba(X.iloc[400:420])[:, 1]
    assert len(prob) == 20
    assert np.all((prob >= 0) & (prob <= 1)), "概率必须落在 [0, 1]"


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="未知 model_type"):
        md.build_model("random_forest")


# ------------------------------------------------------------------ 滚动前向
def test_walk_forward_predictions_cover_test_period(dataset):
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)

    res = md.walk_forward_validate(
        X,
        y,
        model_type="logistic",
        n_splits=3,
        init_train_ratio=0.5,
        verbose=False,
    )

    assert len(res.fold_metrics) == 3
    assert len(res.predictions) == len(X) - int(len(X) * 0.5)

    # 预测必须严格覆盖「初始训练集之后」的时间段，且不重叠
    first_test_start = int(len(X) * 0.5)
    assert res.predictions.index.min() >= X.index[first_test_start]
    assert res.predictions.index.is_monotonic_increasing
    assert not res.predictions.index.has_duplicates

    # 每一折的训练截止日都必须早于该折测试起始日
    for f in res.fold_metrics:
        assert f["train_end"] < f["test_start"], (
            f"第 {f['fold']} 折出现时间穿越：训练至 {f['train_end']}，却预测 {f['test_start']}"
        )


def test_walk_forward_auc_is_sane(dataset):
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)
    res = md.walk_forward_validate(X, y, model_type="logistic", n_splits=3, verbose=False)
    for f in res.fold_metrics:
        auc = f["auc"]
        assert np.isnan(auc) or 0.0 <= auc <= 1.0, f"AUC 越界：{auc}"


def test_walk_forward_rejects_too_many_splits(dataset):
    X = dataset.frame[dataset.feature_names].iloc[:150]
    y = dataset.frame["label"].astype(int).iloc[:150]
    with pytest.raises(ValueError):
        md.walk_forward_validate(X, y, n_splits=200, verbose=False)


# ------------------------------------------------------------------ 预测与重要性
def test_predict_latest_returns_probability(synthetic_nav, synthetic_index):
    from fund_signal import features as ft

    ds = ft.build_dataset(synthetic_nav, synthetic_index, horizon=1)
    X = ds.frame[ds.feature_names]
    y = ds.frame["label"].astype(int)

    model = md.train_final_model(X, y, model_type="logistic")
    p = md.predict_latest(model, ds.latest)
    assert 0.0 <= p <= 1.0


def test_feature_importance_normalised(dataset):
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)
    model = md.train_final_model(X, y, model_type="lightgbm", random_state=0)

    imp = md.feature_importance(model, dataset.feature_names)
    assert len(imp) == len(dataset.feature_names)
    assert pytest.approx(imp.sum(), abs=1e-6) == 1.0, "归一化后重要性和应为 1"
    assert imp.is_monotonic_decreasing
