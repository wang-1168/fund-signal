# -*- coding: utf-8 -*-
"""
模型训练与「滚动前向」时序验证。

为什么不用 scikit-learn 默认的交叉验证？
----------------------------------------
金融时间序列有强自相关，随机划分训练/测试集会让「未来信息」渗漏到训练集里
（例如把 2024 年的样本拿去训练、再用 2023 年的样本测试），结果 AUC 能轻松
做到 0.7+，但实盘一文不值。

本模块提供两种时序安全的评估方式：

1. :func:`train_test_split_ts` —— 一次性按时间切分，简单直观。
2. :func:`walk_forward_validate` —— 滚动前向验证。用「过去」预测「未来」，
   并在时间轴上不断向前推进，最接近真实交易场景。

两者都保证：**任何一次预测，其训练数据在时间上都严格早于被预测的样本。**
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .utils import get_logger

log = get_logger()

MIN_TRAIN_SAMPLES = 60  # 低于此样本量训练出的模型毫无意义
MIN_TEST_SAMPLES = 20


# ------------------------------------------------------------------ 模型工厂
def build_model(model_type: str = "lightgbm", random_state: int = 42, **kwargs):
    """构造分类器。

    LightGBM 的默认参数针对「小样本 + 低信噪比」做了保守设置：
    浅树、小学习率、强正则，避免把噪声学成规律。
    """
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "缺少 lightgbm，请执行：pip install lightgbm\n（或改用 model_type='logistic'）"
            ) from err

        params = dict(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
        params.update(kwargs)
        return LGBMClassifier(**params)

    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        params = dict(C=0.1, max_iter=2000, random_state=random_state)
        params.update(kwargs)
        # scaler 与分类器打包在一起，保证标准化统计量只来自训练集
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**params)),
            ]
        )

    raise ValueError(f"未知 model_type: {model_type!r}，可选 'lightgbm' / 'logistic'")


# ------------------------------------------------------------------ 时序切分
def train_test_split_ts(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.7):
    """按时间顺序切分数据集（**不做任何打乱**）。

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 (0, 1) 之间")

    n = len(X)
    split = int(n * train_ratio)
    if split < MIN_TRAIN_SAMPLES or (n - split) < MIN_TEST_SAMPLES:
        raise ValueError(
            f"样本量不足：共 {n} 行，按 {train_ratio:.0%} 切分后训练集 {split} 行、"
            f"测试集 {n - split} 行。请拉长基金的历史区间或减小 horizon。"
        )

    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ------------------------------------------------------------------ 滚动前向
@dataclass
class WalkForwardResult:
    """滚动前向验证的结果集合。"""

    predictions: pd.Series  # 索引=日期，值=上涨概率
    fold_metrics: list[dict] = field(default_factory=list)  # 每折的评估指标
    overall_metrics: dict = field(default_factory=dict)  # 汇总指标

    @property
    def mean_auc(self) -> float:
        aucs = [f["auc"] for f in self.fold_metrics if not np.isnan(f["auc"])]
        return float(np.mean(aucs)) if aucs else float("nan")

    @property
    def auc_summary(self) -> str:
        vals = [f"AUC={f['auc']:.3f}" for f in self.fold_metrics if not np.isnan(f["auc"])]
        return f"均值 {self.mean_auc:.3f}（{' | '.join(vals)}）"


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lightgbm",
    n_splits: int = 5,
    init_train_ratio: float = 0.5,
    threshold: float = 0.5,
    random_state: int = 42,
    model_params: dict | None = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """滚动前向验证（walk-forward validation）。

    做法：预留前 ``init_train_ratio`` 的样本作为初始训练集，把剩余样本
    等分成 ``n_splits`` 段；对每一段，只用它**之前**的全部数据训练模型，
    对该段做预测。因此预测序列是「逐段实盘模拟」的产物。

    Parameters
    ----------
    X, y:
        特征与标签，索引须为按时间升序的日期。
    model_type:
        ``"lightgbm"`` 或 ``"logistic"``。
    n_splits:
        测试段数。越多越稳健，但计算量越大。
    init_train_ratio:
        初始训练集占比。
    threshold:
        计算准确率等阈值型指标时使用的分界值。
    model_params:
        透传给分类器的额外参数。

    Returns
    -------
    WalkForwardResult
    """
    n = len(X)
    init = int(n * init_train_ratio)
    if init < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"初始训练集仅 {init} 行，不足 {MIN_TRAIN_SAMPLES} 行。"
            "请拉长基金历史区间，或降低 init_train_ratio。"
        )
    if n - init < n_splits:
        raise ValueError("测试样本少于折数，请降低 n_splits。")

    model_params = model_params or {}
    segments = np.array_split(np.arange(init, n), n_splits)

    preds: list[pd.Series] = []
    fold_metrics: list[dict] = []
    all_true: list[np.ndarray] = []

    for k, idx in enumerate(segments, start=1):
        if len(idx) < 2:
            continue
        train_idx = np.arange(0, int(idx[0]))
        model = build_model(model_type, random_state=random_state, **model_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        prob = model.predict_proba(X.iloc[idx])[:, 1]
        preds.append(pd.Series(prob, index=X.index[idx]))
        all_true.append(y.iloc[idx].to_numpy())

        m = classification_metrics(y.iloc[idx], prob, threshold)
        m.update(
            {
                "fold": k,
                "train_end": str(pd.Timestamp(X.index[train_idx[-1]]).date()),
                "test_start": str(pd.Timestamp(X.index[idx[0]]).date()),
                "test_end": str(pd.Timestamp(X.index[idx[-1]]).date()),
                "n_train": int(len(train_idx)),
                "n_test": int(len(idx)),
            }
        )
        fold_metrics.append(m)
        if verbose:
            log.info(
                "折 %d/%d | 训练至 %s | 测试 %s ~ %s | AUC=%.3f 准确率=%.3f",
                k,
                n_splits,
                m["train_end"],
                m["test_start"],
                m["test_end"],
                m["auc"],
                m["accuracy"],
            )

    if not preds:
        raise ValueError("没有任何有效折，请检查 n_splits 与样本量。")

    predictions = pd.concat(preds).sort_index()
    y_true_all = np.concatenate(all_true)
    overall = classification_metrics(y_true_all, predictions.to_numpy(), threshold)

    result = WalkForwardResult(
        predictions=predictions, fold_metrics=fold_metrics, overall_metrics=overall
    )
    if verbose:
        log.info("滚动前向汇总：%s", result.auc_summary)
    return result


# ------------------------------------------------------------------ 训练与预测
def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lightgbm",
    random_state: int = 42,
    model_params: dict | None = None,
):
    """用全部历史数据训练最终模型，用于产出「最新一期」信号。"""
    if len(X) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"样本量不足（{len(X)} 行），无法训练。")
    model = build_model(model_type, random_state=random_state, **(model_params or {}))
    model.fit(X, y)
    return model


def predict_latest(model, X_latest: pd.DataFrame) -> float:
    """对最新一行特征输出上涨概率。"""
    prob = model.predict_proba(X_latest.iloc[[-1]])[0, 1]
    return float(prob)


def predict_proba_frame(model, X: pd.DataFrame) -> pd.Series:
    """对整段特征输出概率序列（带原索引）。"""
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)


def feature_importance(model, feature_names: Sequence[str]) -> pd.Series:
    """统一提取特征重要性（树模型用 split gain，线性模型用系数绝对值）。"""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_, dtype=float)).ravel()
    else:
        return pd.Series(dtype=float)

    if len(imp) != len(feature_names):
        return pd.Series(dtype=float)

    total = imp.sum()
    if total > 0:
        imp = imp / total  # 归一化为占比，便于跨模型比较

    return pd.Series(imp, index=list(feature_names)).sort_values(ascending=False)


__all__ = [
    "build_model",
    "train_test_split_ts",
    "walk_forward_validate",
    "WalkForwardResult",
    "train_final_model",
    "predict_latest",
    "predict_proba_frame",
    "feature_importance",
]
