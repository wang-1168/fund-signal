# -*- coding: utf-8 -*-
"""能力边界审计模块的单元测试。

这里的断言大多是「结构性」的（形状、范围、单调性、幂等），
而不是具体数值——金融指标的具体数值会随数据版本漂移，
但它们必须满足的**约束**是稳定的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fund_signal import audit
from fund_signal.model import walk_forward_validate


# ------------------------------------------------------------------ 理论换算
@pytest.mark.parametrize(
    ("auc", "acc"),
    [(0.50, 0.50), (0.965, 0.90), (1.0, 1.0)],
)
def test_auc_to_accuracy_known_points(auc, acc):
    """AUC-准确率换算在三个锚点上必须精确成立。"""
    assert audit.auc_to_accuracy(auc) == pytest.approx(acc, abs=2e-3)


def test_required_auc_for_90pct():
    """准确率 90% 反推所需 AUC 应在 0.96~0.97 之间。"""
    need = audit.accuracy_to_auc(0.90)
    assert 0.96 < need < 0.97


def test_auc_accuracy_curve_monotonic():
    """AUC 越高，理论准确率上界必须单调不减。"""
    df = audit.auc_accuracy_curve(0.50, 0.99, 40)
    assert len(df) == 40
    assert np.all(np.diff(df["accuracy"].to_numpy()) >= -1e-12)
    assert df["accuracy"].iloc[0] == pytest.approx(0.5, abs=1e-6)


# ------------------------------------------------------------------ 复杂度地形
def test_complexity_path_overfit_gap_increases(dataset):
    """复杂度拉满时，训练准确率必须显著高于样本外（过拟合差距为正且变大）。

    这是本项目对「过拟合」最核心的一条断言：如果它失败了，
    说明要么数据可预测性极高（不太可能），要么实现出了问题。
    """
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)
    grid = audit.DEFAULT_COMPLEXITY_GRID[:3] + [audit.DEFAULT_COMPLEXITY_GRID[-1]]
    df = audit.complexity_path(X, y, grid=grid, n_splits=3, embargo=1)

    assert list(df.columns) == [
        "配置",
        "训练准确率",
        "样本外准确率",
        "过拟合差距",
        "样本外AUC",
        "多数类基线",
        "节点数",
    ]
    # 训练准确率随复杂度单调上升
    train = df["训练准确率"].to_numpy()
    assert np.all(np.diff(train) >= -1e-9)
    # 最复杂一档的过拟合差距必须明显大于最简单一档
    assert df["过拟合差距"].iloc[-1] > df["过拟合差距"].iloc[0] + 0.15
    # 样本外始终在合理区间内摆动，不可能被复杂度拉到 0.9
    assert df["样本外准确率"].between(0.3, 0.8).all()


# ------------------------------------------------------------------ 泄漏审计
def test_leakage_audit_flags_the_leak(dataset):
    """答案入特征时准确率必须接近满分——这是泄漏审计的存在理由。"""
    df = audit.leakage_audit(dataset.frame, dataset.feature_names, horizon=1)
    assert len(df) == 3

    normal = df.loc[df["变体"].str.contains("正常"), "准确率"].iloc[0]
    leaked = df.loc[df["变体"].str.contains("答案入特征"), "准确率"].iloc[0]
    lookahead = df.loc[df["变体"].str.contains("未来函数"), "准确率"].iloc[0]

    assert leaked > 0.99, "故意泄漏理应拿到接近满分的准确率"
    # 未来函数几乎不改变准确率 —— 这正是准确率不能作为安全指标的证据
    assert abs(lookahead - normal) < 0.10
    # 正常版本的准确率必须落在「不是满分」的诚实区间
    assert 0.35 < normal < 0.75


# ------------------------------------------------------------------ 置信度分层
def test_confidence_subset_table_shape(dataset):
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)
    wf = walk_forward_validate(X, y, n_splits=3, verbose=False)
    tbl = audit.confidence_subset_table(wf.predictions, y.loc[wf.predictions.index], 4)

    assert len(tbl) >= 2
    assert tbl["覆盖率"].sum() == pytest.approx(1.0, abs=1e-6)
    assert tbl["方向准确率"].between(0, 1).all()
    # 等频分桶下各桶样本数应大致相等（允许 10% 偏差）
    sizes = tbl["样本数"].to_numpy()
    assert sizes.max() <= sizes.min() * 1.2 + 2


def test_selective_accuracy_table_is_monotone_in_coverage(dataset):
    """覆盖率越低（只挑最自信的样本），平均置信度必须越高。"""
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)
    wf = walk_forward_validate(X, y, n_splits=3, verbose=False)
    tbl = audit.selective_accuracy_table(
        wf.predictions, y.loc[wf.predictions.index], coverages=(1.0, 0.5, 0.2, 0.1)
    )
    assert len(tbl) == 4
    conf = tbl["平均置信度"].to_numpy()
    assert np.all(np.diff(conf) > 0), "选择性预测的平均置信度应随覆盖率下降而上升"
    assert tbl["覆盖天数"].is_monotonic_decreasing


# ------------------------------------------------------------------ 校准诊断
def test_calibration_diagnostics_perfectly_calibrated():
    """构造一个完美校准的预测器，ECE 必须接近 0。"""
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    out = audit.calibration_diagnostics(p, y, n_bins=10)
    assert out["ece"] < 0.03
    assert out["brier"] < 0.25
    assert len(out["bins"]) == 10


def test_calibration_diagnostics_miscalibrated():
    """把概率整体抬高 0.3，ECE 必须显著大于 0。"""
    rng = np.random.default_rng(1)
    n = 3000
    p = np.clip(rng.uniform(0.1, 0.5, n) + 0.3, 0, 1)
    y = (rng.uniform(size=n) < p - 0.3).astype(int)
    out = audit.calibration_diagnostics(p, y, n_bins=10)
    assert out["ece"] > 0.10


# ------------------------------------------------------------------ 综合体检
def test_capability_report_fields(dataset):
    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)
    wf = walk_forward_validate(X, y, n_splits=3, verbose=False)
    rep = audit.capability_report(wf.predictions, y.loc[wf.predictions.index])

    for key in (
        "auc",
        "accuracy",
        "majority_acc",
        "acc_edge",
        "理论准确率上界",
        "所需AUC",
        "AUC缺口",
        "ece",
        "brier",
    ):
        assert key in rep

    assert rep["所需AUC"] == pytest.approx(0.965, abs=2e-3)
    # 理论上界必须不低于实际准确率（在均衡两类的前提下）
    assert rep["理论准确率上界"] >= rep["accuracy"] - 0.05
    assert not pd.isna(rep["AUC缺口"]) and rep["AUC缺口"] > 0.3
