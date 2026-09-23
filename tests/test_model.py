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
@pytest.mark.parametrize("model_type", ["lightgbm", "logistic", "ensemble"])
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


# ------------------------------------------------------------------ 净化间隔
def test_embargo_shortens_training_window(dataset):
    """开启净化间隔后，每折的训练截止日必须严格提前 horizon 天。"""
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)

    plain = md.walk_forward_validate(X, y, model_type="logistic", n_splits=3, verbose=False)
    purged = md.walk_forward_validate(
        X, y, model_type="logistic", n_splits=3, embargo=1, verbose=False
    )

    assert len(purged.predictions) == len(plain.predictions), "净化不应减少预测样本"
    for a, b in zip(plain.fold_metrics, purged.fold_metrics, strict=True):
        assert b["train_end"] < a["train_end"], "净化后的训练截止日应更早"
        assert b["test_start"] == a["test_start"]


# ------------------------------------------------------------------ 概率校准
def test_calibrator_reduces_ece(dataset):
    """样本外校准应当把 ECE 压下来（这是校准唯一被允许存在的原因）。"""
    from fund_signal.audit import calibration_diagnostics

    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)

    res = md.walk_forward_validate(
        X, y, model_type="lightgbm", n_splits=3, calibrate=True, verbose=False
    )
    assert res.calibrate_method == "isotonic"
    assert res.raw_predictions is not None

    y_true = y.loc[res.predictions.index]
    ece_raw = calibration_diagnostics(res.raw_predictions, y_true, 10)["ece"]
    ece_cal = calibration_diagnostics(res.predictions, y_true, 10)["ece"]

    assert ece_cal <= ece_raw + 0.01, f"校准后 ECE({ece_cal:.4f}) 不应明显高于校准前({ece_raw:.4f})"
    # 排序能力不应被校准破坏（isotonic 是单调变换，AUC 理论上不变）
    auc_raw = md.walk_forward_validate(
        X, y, model_type="lightgbm", n_splits=3, calibrate=False, verbose=False
    ).overall_metrics["auc"]
    auc_cal = res.overall_metrics["auc"]
    assert abs(auc_raw - auc_cal) < 0.05


def test_calibrator_falls_back_on_tiny_samples(synthetic_nav, synthetic_index):
    """样本太少时校准必须优雅回退，而不是抛异常。"""
    from fund_signal import features as ft

    small = synthetic_nav.iloc[:180]
    ds = ft.build_dataset(small, synthetic_index.iloc[:180], horizon=1)
    X = ds.frame[ds.feature_names]
    y = ds.frame["label"].astype(int)

    res = md.walk_forward_validate(
        X,
        y,
        model_type="logistic",
        n_splits=2,
        init_train_ratio=0.5,
        calibrate=True,
        verbose=False,
    )
    assert res.predictions.notna().all()
    assert res.calibrate_method in ("isotonic", "sigmoid", "none")


def test_calibrator_identity_when_none():
    """calibrator=None 时 apply_calibrator 必须原样返回。"""
    p = np.array([0.1, 0.4, 0.9])
    assert np.allclose(md.apply_calibrator(None, p), p)


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


def test_feature_importance_for_ensemble(dataset):
    """集成模型的重要性必须是「逐成员归一化后取平均」，同样应归一到 1。"""
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)
    model = md.train_final_model(X, y, model_type="ensemble", random_state=0)

    imp = md.feature_importance(model, dataset.feature_names)
    assert len(imp) == len(dataset.feature_names)
    assert pytest.approx(imp.sum(), abs=1e-6) == 1.0
    assert imp.is_monotonic_decreasing
    assert (imp > 0).sum() >= 3, "集成的重要性不应全部集中在单一特征上"
