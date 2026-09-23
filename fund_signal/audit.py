# -*- coding: utf-8 -*-
"""能力边界审计。

这个模块回答一个所有量化项目最终都必须正面回答的问题：

    **这套模型到底能做到多准？以及，那些「看起来很准」是怎么来的？**

它提供四类工具：

1. :func:`auc_to_accuracy` / :func:`required_auc` —— 准确率与 AUC 的理论换算。
   这条换算关系戳破了很多幻觉：想要 90% 的方向准确率，需要 AUC ≈ 0.965，
   而在公募基金日频净值上实测 AUC 只有 0.50~0.58。
2. :func:`complexity_path` —— 模型复杂度 → (训练准确率, 样本外准确率) 曲线。
   这条曲线就是**过拟合与欠拟合的完整地形图**：左端两端都低（欠拟合），
   右端训练集飞高、样本外掉回 50%（过拟合），中间存在一个小而平的最优区。
3. :func:`leakage_audit` —— 泄漏审计。用「故意犯错」的方式证明：
   一个把答案塞进特征集的模型，可以在**严格的滚动前向验证下**拿到 100%
   准确率；而真正的未来函数（把买入价写成 T 日）几乎不改变准确率。
   换句话说，**高准确率不能证明模型有效，只有独立的泄漏审计才能。**
4. :func:`confidence_subset_table` —— 置信度分层。只在模型概率足够极端时才下注，
   此时「已下注样本」的准确率会显著高于总体。这是在不作弊的前提下，
   唯一能诚实地把准确率做上去的正常手段——代价是覆盖面大幅缩小。

本模块的所有函数都只使用样本外预测，不做任何参数搜索式的择优。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import accuracy_score

from . import metrics as mt
from .utils import get_logger

log = get_logger()


# ================================================================= 理论换算
def auc_to_accuracy(auc: float) -> float:
    """AUC → 该 AUC 下**最优判定**所能达到的准确率上界。

    推导（假设正负类得分分别服从 ``N(m, 1)`` 与 ``N(0, 1)``）：

    - ``AUC = Φ(m / √2)``，故 ``m = √2 · Φ⁻¹(AUC)``
    - 最优判定阈值为 ``m / 2``，此时准确率 ``= Φ(m / 2) = Φ(Φ⁻¹(AUC) / √2)``

    这个换算说明一件反直觉的事：**AUC 只是「排序能力」，而准确率对 AUC 极其敏感**。
    AUC 从 0.52 涨到 0.60，准确率只从 51.4% 涨到 57.1%；想摸到 90%，
    AUC 得先到 0.965——那意味着市场几乎已经确定性可预测。
    """
    if auc is None or (isinstance(auc, float) and np.isnan(auc)):
        return float("nan")
    a = float(np.clip(auc, 1e-6, 1 - 1e-6))
    return float(norm.cdf(norm.ppf(a) / np.sqrt(2.0)))


def accuracy_to_auc(acc: float) -> float:
    """:func:`auc_to_accuracy` 的反函数：达到指定准确率所需的 AUC。"""
    if acc is None or (isinstance(acc, float) and np.isnan(acc)):
        return float("nan")
    a = float(np.clip(acc, 1e-6, 1 - 1e-6))
    return float(norm.cdf(np.sqrt(2.0) * norm.ppf(a)))


def auc_accuracy_curve(lo: float = 0.50, hi: float = 0.99, n: int = 60) -> pd.DataFrame:
    """生成 AUC-准确率换算法曲线，供界面绘图。"""
    aucs = np.linspace(lo, hi, int(n))
    return pd.DataFrame({"auc": aucs, "accuracy": [auc_to_accuracy(a) for a in aucs]})


# ================================================================= 复杂度地形
DEFAULT_COMPLEXITY_GRID: list[tuple[str, dict]] = [
    (
        "① 极简(深度1)",
        dict(n_estimators=100, learning_rate=0.05, num_leaves=2, max_depth=1, min_child_samples=60),
    ),
    (
        "② 浅树(默认)",
        dict(n_estimators=300, learning_rate=0.03, num_leaves=8, max_depth=2, min_child_samples=30),
    ),
    (
        "③ 中等",
        dict(
            n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4, min_child_samples=20
        ),
    ),
    (
        "④ 较深",
        dict(
            n_estimators=300, learning_rate=0.06, num_leaves=63, max_depth=8, min_child_samples=10
        ),
    ),
    (
        "⑤ 不剪枝",
        dict(
            n_estimators=600, learning_rate=0.08, num_leaves=255, max_depth=-1, min_child_samples=1
        ),
    ),
]
"""从欠拟合到过拟合的复杂度阶梯（LightGBM 参数）。

刻意只保留 5 档、并压低树数量：这条曲线的目的是**展示趋势**，
不是精确排名，跑得动比跑得全重要。
"""


def complexity_path(
    X: pd.DataFrame,
    y: pd.Series,
    grid: Sequence[tuple[str, dict]] | None = None,
    n_splits: int = 5,
    embargo: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """扫描模型复杂度，同时记录**训练集准确率**与**样本外准确率**。

    ``gap = 训练准确率 - 样本外准确率`` 就是过拟合的直接度量：

    - 训练 0.55 / 样本外 0.51 → 欠拟合（模型太弱，两头都上不去）
    - 训练 0.78 / 样本外 0.50 → 有一定拟合能力但信息量不足，正常
    - 训练 1.00 / 样本外 0.49 → **彻底过拟合**，模型把噪声全背下来了

    在真实的金融数据上，通常从第二行就会跳到最后一行——这正是「把准确率做到
    90% 以上」最省事的办法，也是它在实盘必然失效的原因。
    """
    from .model import train_final_model, walk_forward_validate

    grid = list(grid or DEFAULT_COMPLEXITY_GRID)
    rows: list[dict] = []
    for label, params in grid:
        wf = walk_forward_validate(
            X,
            y,
            model_type="lightgbm",
            n_splits=n_splits,
            embargo=embargo,
            random_state=random_state,
            model_params=params,
            verbose=False,
        )
        model = train_final_model(
            X, y, model_type="lightgbm", random_state=random_state, model_params=params
        )
        train_acc = float(accuracy_score(y, model.predict(X)))
        oos_acc = float(wf.overall_metrics["accuracy"])
        rows.append(
            {
                "配置": label,
                "训练准确率": train_acc,
                "样本外准确率": oos_acc,
                "过拟合差距": train_acc - oos_acc,
                "样本外AUC": float(wf.overall_metrics["auc"]),
                "多数类基线": float(wf.overall_metrics["majority_acc"]),
                "节点数": int(params.get("num_leaves", 31)) * int(params.get("max_depth", 3) or 1),
            }
        )
    return pd.DataFrame(rows)


# ================================================================= 泄漏审计
def leakage_audit(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    horizon: int = 1,
    model_type: str = "lightgbm",
    n_splits: int = 5,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """泄漏审计：用三组对照证明「准确率不是安全指标」。

    三个变体共用同一份数据、同一套超参数、同一种验证方式，只改一处：

    ====================  ====================================================
    变体                  改动
    ====================  ====================================================
    ① 正常                只用 T 日及之前的信息构造特征
    ② 答案入特征          额外把 ``fwd_ret``（未来收益本身）塞进特征集
    ③ 未来函数标签        标签改成「T 日买入」（假设看完 T 日净值还能按 T 日成交）
    ====================  ====================================================

    预期结果与它的含义：

    - ② 准确率 ≈ 1.00 —— **泄漏能在严格的滚动前向验证下拿到满分**。
      所以「我做了滚动前向验证，准确率 95%」绝不是有效性的证据。
    - ③ 准确率与 ① 相差无几 —— **准确率根本发现不了未来函数**，
      只有回测收益（和净值曲线）才会虚高。必须独立做标签时点审计。

    Parameters
    ----------
    frame:
        :class:`~fund_signal.features.Dataset` 的 ``frame``（需含 ``label``、
        ``fwd_ret``；可选 ``nav``）。
    feature_names:
        正常特征列名。
    """
    feats = list(feature_names)
    rows: list[dict] = []

    def _run(tag: str, X: pd.DataFrame, y: pd.Series, note: str) -> None:
        from .model import walk_forward_validate

        valid = X.notna().all(axis=1) & y.notna()
        Xv, yv = X.loc[valid], y.loc[valid].astype(int)
        wf = walk_forward_validate(
            Xv,
            yv,
            model_type=model_type,
            n_splits=n_splits,
            threshold=threshold,
            verbose=False,
        )
        m = wf.overall_metrics
        rows.append(
            {
                "变体": tag,
                "样本数": m["n_samples"],
                "准确率": m["accuracy"],
                "AUC": m["auc"],
                "多数类基线": m["majority_acc"],
                "超额": m["acc_edge"],
                "说明": note,
            }
        )

    _run("① 正常（只用历史特征）", frame[feats], frame["label"], "实盘可执行 —— 这才是可信的数字")

    leak_cols = [c for c in ("fwd_ret", "label") if c in frame.columns]
    if leak_cols:
        _run(
            "② 答案入特征（故意泄漏）",
            frame[feats + leak_cols],
            frame["label"],
            "准确率≈100%，但实盘拿不到未来收益，属于自欺",
        )

    if "nav" in frame.columns:
        nav = frame["nav"]
        h = max(1, int(horizon))
        # 未来函数：以 T 日净值为买入价（比正确做法早一天）
        bad_fwd = nav.shift(-h) / nav - 1.0
        bad_label = (bad_fwd > 0).astype(float)
        bad_label[bad_fwd.isna()] = np.nan
        _run(
            "③ 未来函数标签（T 日买入）",
            frame[feats],
            bad_label,
            "准确率几乎不变 —— 说明准确率查不出未来函数",
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info("泄漏审计完成：\n%s", df.to_string(index=False))
    return df


# ================================================================= 置信度分层
def confidence_subset_table(
    prob: pd.Series,
    y_true: pd.Series,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """按「模型自信程度」**等频**分桶，统计每桶的方向准确率。

    为什么用等频而不是等距？
    ------------------------
    校准之后模型的概率会极度收缩——实测里 99.6% 的预测落在 0.5±0.05 之间。
    这时用 ``|p-0.5| = 0.05 / 0.10 / 0.15`` 之类等距切分，除了第一个桶全是空的，
    会得到一张毫无信息量的表。等频分桶保证每桶都有足够样本，
    再窄的概率分布也能显示出「越自信越准」这个单调性。

    读法：如果 Q1（最不自信）到 Q5（最自信）的准确率没有单调上升，
    说明模型输出的大小关系**不携带信息**，此时给概率排序毫无意义。
    """
    p = pd.Series(prob, dtype=float)
    yv = pd.Series(np.asarray(y_true, dtype=int), index=p.index)
    conf = (p - 0.5).abs()
    if len(p) < 40:
        return pd.DataFrame()

    try:
        bucket = pd.qcut(conf, q=int(n_buckets), labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    rows: list[dict] = []
    for b in sorted(pd.Series(bucket).dropna().unique()):
        mask = bucket == b
        yy, pp = yv[mask], p[mask]
        pred = (pp >= 0.5).astype(int)
        acc = float(accuracy_score(yy, pred))
        base = max(float(yy.mean()), 1.0 - float(yy.mean()))
        rows.append(
            {
                "信心分组": f"Q{int(b) + 1}",
                "置信度范围": f"{conf[mask].min():.3f}~{conf[mask].max():.3f}",
                "样本数": int(mask.sum()),
                "覆盖率": float(mask.mean()),
                "方向准确率": acc,
                "层内基线": base,
                "超额": acc - base,
            }
        )
    return pd.DataFrame(rows)


def selective_accuracy_table(
    prob: pd.Series,
    y_true: pd.Series,
    coverages: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05),
) -> pd.DataFrame:
    """选择性预测：只对模型最自信的前 ``k%`` 样本下注，准确率能到多少。

    这是**唯一诚实有效的提高准确率的手段**：不改变模型、不引入未来信息，
    只是「不确定时选择不说话」。

    但必须同时看两列：

    - ``覆盖率``：一年 244 个交易日里，模型愿意开口的有几天；
    - ``超额``：开口时相对该子集内「恒猜多数类」的增量。

    如果覆盖率 5% 时准确率 70%、但超额只有 1%，说明这 70% 主要来自
    「这段时间本来就在涨」，而不是模型的本事。**准确率必须减去基线才有意义。**
    """
    p = pd.Series(prob, dtype=float)
    yv = pd.Series(np.asarray(y_true, dtype=int), index=p.index)
    conf = (p - 0.5).abs()
    n = len(p)
    if n < 40:
        return pd.DataFrame()

    order = conf.sort_values(ascending=False).index
    rows: list[dict] = []
    for cov in coverages:
        k = max(10, int(round(n * float(cov))))
        k = min(k, n)
        sel = order[:k]
        yy, pp = yv.loc[sel], p.loc[sel]
        pred = (pp >= 0.5).astype(int)
        acc = float(accuracy_score(yy, pred))
        base = max(float(yy.mean()), 1.0 - float(yy.mean()))
        rows.append(
            {
                "下注比例": float(cov),
                "覆盖天数": k,
                "方向准确率": acc,
                "该子集基线": base,
                "超额": acc - base,
                "平均置信度": float(conf.loc[sel].mean()),
            }
        )
    return pd.DataFrame(rows)


# ================================================================= 校准诊断
def calibration_diagnostics(prob, y_true, n_bins: int = 10) -> dict:
    """概率校准质量：ECE / MCE / Brier 分解。

    ``ECE``（期望校准误差）= 各分箱「预测概率均值」与「实际频率」之差的加权平均。
    它衡量的是**概率本身可不可信**——一个 ECE 只有 0.01 的模型，
    即使 AUC 平平，它说「70% 概率上涨」时，历史上确实有约 70% 涨了。
    这比准确率有用得多，因为概率可以直接映射成仓位。
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.size < 30:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "brier": float("nan"),
            "bins": pd.DataFrame(),
        }

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, int(n_bins) - 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for b in range(int(n_bins)):
        m = idx == b
        if not m.any():
            continue
        conf = float(p[m].mean())
        frac = float(y[m].mean())
        w = float(m.mean())
        gap = abs(conf - frac)
        ece += w * gap
        mce = max(mce, gap)
        rows.append(
            {
                "分箱": f"{bins[b]:.1f}~{bins[b + 1]:.1f}",
                "样本数": int(m.sum()),
                "平均预测概率": conf,
                "实际上涨频率": frac,
                "偏差": conf - frac,
            }
        )

    brier = float(np.mean((p - y) ** 2))
    return {"ece": float(ece), "mce": float(mce), "brier": brier, "bins": pd.DataFrame(rows)}


# ================================================================= 综合体检
def capability_report(
    prob: pd.Series,
    y_true: pd.Series,
    n_bins: int = 10,
    target_accuracy: float = 0.90,
) -> dict:
    """把上面的诊断汇成一份「能力边界」结论。"""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(y_true, dtype=int)
    base = mt.classification_metrics(y, p, 0.5)
    cal = calibration_diagnostics(p, y, n_bins)
    auc = base["auc"]

    return {
        "n": int(base["n_samples"]),
        "auc": auc,
        "accuracy": base["accuracy"],
        "majority_acc": base["majority_acc"],
        "acc_edge": base["acc_edge"],
        "理论准确率上界": auc_to_accuracy(auc),
        "目标准确率": float(target_accuracy),
        "所需AUC": accuracy_to_auc(target_accuracy),
        "AUC缺口": (accuracy_to_auc(target_accuracy) - auc) if not np.isnan(auc) else float("nan"),
        "ece": cal["ece"],
        "brier": cal["brier"],
        "calibration_bins": cal["bins"],
    }


__all__ = [
    "auc_to_accuracy",
    "accuracy_to_auc",
    "auc_accuracy_curve",
    "DEFAULT_COMPLEXITY_GRID",
    "complexity_path",
    "leakage_audit",
    "confidence_subset_table",
    "selective_accuracy_table",
    "calibration_diagnostics",
    "capability_report",
]
