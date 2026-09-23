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

    支持 ``"ensemble"``——四种**偏差方向不同**的弱模型做软投票：

    ==================  ==========================================
    成员                作用
    ==================  ==========================================
    LightGBM            梯度提升树，擅长非线性交互
    LogisticRegression  线性基准，强正则，抗噪
    ExtraTrees          极度随机化森林，方差低
    GradientBoosting    小学习率提升树，与 LightGBM 互补
    ==================  ==========================================

    集成能小幅提升 AUC，更重要的是**降低单模型的方差**——在信噪比极低的
    金融数据上，方差比偏差更致命。但请注意：集成无法突破数据本身的信息上限，
    它只会把「没有预测力」变成「更稳定的没有预测力」，而不会凭空造出 alpha。
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

    if model_type == "ensemble":
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            VotingClassifier,
        )

        members = [
            ("lgbm", build_model("lightgbm", random_state=random_state, **kwargs)),
            ("logit", build_model("logistic", random_state=random_state)),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=20,
                    max_features=0.6,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
            (
                "gbdt",
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.03,
                    max_depth=2,
                    subsample=0.8,
                    random_state=random_state,
                ),
            ),
        ]
        # voting="soft" 需要每个成员都能输出概率；标准化只在 logistic 内部做，
        # 树模型对量纲不敏感，因此无需把 scaler 提到集成外面。
        return VotingClassifier(estimators=members, voting="soft", n_jobs=1)

    raise ValueError(f"未知 model_type: {model_type!r}，可选 'lightgbm' / 'logistic' / 'ensemble'")


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


# ------------------------------------------------------------------ 概率校准
def fit_calibrator(prob, y, method: str = "isotonic"):
    """在**样本外**概率上拟合一个概率校准器。

    为什么需要校准
    --------------
    树模型输出的「概率」往往是失真的：它会把 0.55 和 0.95 都压在高分区，
    于是「概率 0.8」既不意味着 80% 会涨，也不能直接拿去当仓位权重。
    校准把预测概率映射回真实频率，是让概率**可用于决策**的前提。

    为什么必须在样本外拟合
    ----------------------
    在训练集上拟合校准器，等于让校准器去背训练集的答案（模型在训练集上
    几乎必然过拟合，isotonic 会把它们全拉成 0/1），校准后看起来完美，
    换到新数据立刻失效。所以本函数只接受**未参与训练**的概率。

    Parameters
    ----------
    prob:
        样本外预测概率。
    y:
        对应真实标签。
    method:
        ``"isotonic"`` 非参数单调回归（样本足够时优先）；
        ``"sigmoid"`` Platt 缩放（样本少时更稳）。
    """
    p = np.asarray(prob, dtype=float)
    yv = np.asarray(y, dtype=int)
    if p.size < 30 or np.unique(yv).size < 2:
        return None  # 样本不足或只有单一类别，放弃校准（退回原始概率）

    try:
        if method == "sigmoid":
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(p.reshape(-1, 1), yv)
            return ("sigmoid", lr)

        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, yv)
        return ("isotonic", iso)
    except Exception as err:  # pragma: no cover - 数值退化时不应中断主流程
        log.warning("概率校准器拟合失败，回退为原始概率：%s", err)
        return None


def apply_calibrator(calibrator, prob):
    """应用校准器；``calibrator`` 为 None 时原样返回。"""
    if calibrator is None:
        return np.asarray(prob, dtype=float)
    kind, obj = calibrator
    p = np.asarray(prob, dtype=float)
    if kind == "sigmoid":
        return obj.predict_proba(p.reshape(-1, 1))[:, 1]
    return np.clip(obj.predict(p), 0.0, 1.0)


# ------------------------------------------------------------------ 滚动前向
@dataclass
class WalkForwardResult:
    """滚动前向验证的结果集合。"""

    predictions: pd.Series  # 索引=日期，值=上涨概率
    fold_metrics: list[dict] = field(default_factory=list)  # 每折的评估指标
    overall_metrics: dict = field(default_factory=dict)  # 汇总指标
    raw_predictions: pd.Series | None = None  # 校准前的原始概率（用于对比）
    calibrate_method: str = "none"  # 实际生效的校准方式

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
    embargo: int = 0,
    calibrate: bool = False,
    calibrate_method: str = "isotonic",
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
        ``"lightgbm"`` / ``"logistic"`` / ``"ensemble"``。
    n_splits:
        测试段数。越多越稳健，但计算量越大。
    init_train_ratio:
        初始训练集占比。
    threshold:
        计算准确率等阈值型指标时使用的分界值。
    model_params:
        透传给分类器的额外参数。
    embargo:
        **净化间隔**。设成 ``horizon`` 可消除「标签窗口重叠」带来的隐性泄漏：
        训练集最后一天的标签用的是 ``T+1..T+1+horizon`` 的净值，而测试段第一天
        紧邻其后，两者的标签窗口会重叠若干天，等于模型提前偷看了测试期的一小段。
        丢掉紧邻的 ``embargo`` 个训练样本即可切断这条通道。
        影响通常很小，但它属于「不做也能跑、做了才严谨」的那一类设置。
    calibrate:
        是否在每折内部做**样本外概率校准**。开启后，训练集会被切成
        「核心训练段 + 尾部校准段」，模型只在核心段上拟合，校准器只用
        尾部（模型没见过的）样本拟合。这样产出的概率才具备频率含义，
        可以放心拿去做仓位映射或置信度筛选。

    Returns
    -------
    WalkForwardResult
        其中 ``predictions`` 为（可能已校准的）样本外概率，
        ``raw_predictions`` 保留校准前的原始概率，便于对比两种概率的质量。
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
    embargo = max(0, int(embargo))
    segments = np.array_split(np.arange(init, n), n_splits)

    preds: list[pd.Series] = []
    raws: list[pd.Series] = []
    fold_metrics: list[dict] = []
    all_true: list[np.ndarray] = []
    n_calibrated = 0

    for k, idx in enumerate(segments, start=1):
        if len(idx) < 2:
            continue
        full_train = np.arange(0, int(idx[0]))
        # --- 净化：丢掉紧邻测试段的样本，切断重叠标签窗口 ---
        train_idx = full_train[: max(0, len(full_train) - embargo)]
        if len(train_idx) < MIN_TRAIN_SAMPLES:
            train_idx = full_train  # 样本太短时不硬砍，宁可保留可训练性

        model = build_model(model_type, random_state=random_state, **model_params)

        calibrator = None
        if calibrate and len(train_idx) >= MIN_TRAIN_SAMPLES + 40:
            calib_n = max(40, int(round(0.2 * len(train_idx))))
            core_idx, cal_idx = train_idx[:-calib_n], train_idx[-calib_n:]
            model.fit(X.iloc[core_idx], y.iloc[core_idx])
            calibrator = fit_calibrator(
                model.predict_proba(X.iloc[cal_idx])[:, 1],
                y.iloc[cal_idx].to_numpy(),
                method=calibrate_method,
            )
            if calibrator is not None:
                n_calibrated += 1
        else:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

        raw = model.predict_proba(X.iloc[idx])[:, 1]
        prob = apply_calibrator(calibrator, raw)

        raws.append(pd.Series(raw, index=X.index[idx]))
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
                "calibrated": bool(calibrator is not None),
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
    raw_predictions = pd.concat(raws).sort_index()
    y_true_all = np.concatenate(all_true)
    overall = classification_metrics(y_true_all, predictions.to_numpy(), threshold)

    result = WalkForwardResult(
        predictions=predictions,
        fold_metrics=fold_metrics,
        overall_metrics=overall,
        raw_predictions=raw_predictions,
        calibrate_method=(calibrate_method if n_calibrated else "none"),
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


def _single_importance(estimator, n_features: int) -> np.ndarray | None:
    """从一个「单个」分类器里取原始重要性向量（未归一化）。"""
    clf = (
        getattr(estimator, "named_steps", {}).get("clf", estimator)
        if hasattr(estimator, "named_steps")
        else estimator
    )

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_, dtype=float)).ravel()
    else:
        return None

    if imp.shape[0] != n_features:
        return None
    return imp


def feature_importance(model, feature_names: Sequence[str]) -> pd.Series:
    """统一提取特征重要性（树模型用 split gain，线性模型用系数绝对值）。

    对 ``ensemble``（VotingClassifier）：先把每个成员的重要性归一化，
    再按成员平均——否则一棵树多的模型会靠量纲压过其它成员。
    """
    names = list(feature_names)
    n_features = len(names)

    # --- 集成：逐成员归一化后平均 ---
    if hasattr(model, "estimators_"):
        acc = np.zeros(n_features, dtype=float)
        used = 0
        for est in model.estimators_:
            imp = _single_importance(est, n_features)
            if imp is None:
                continue
            total = imp.sum()
            if total > 0:
                acc += imp / total
                used += 1
        if used == 0:
            return pd.Series(dtype=float)
        acc /= used
        return pd.Series(acc, index=names).sort_values(ascending=False)

    imp = _single_importance(model, n_features)
    if imp is None:
        return pd.Series(dtype=float)

    total = imp.sum()
    if total > 0:
        imp = imp / total  # 归一化为占比，便于跨模型比较

    return pd.Series(imp, index=names).sort_values(ascending=False)


__all__ = [
    "build_model",
    "train_test_split_ts",
    "walk_forward_validate",
    "WalkForwardResult",
    "train_final_model",
    "predict_latest",
    "predict_proba_frame",
    "feature_importance",
    "fit_calibrator",
    "apply_calibrator",
]
