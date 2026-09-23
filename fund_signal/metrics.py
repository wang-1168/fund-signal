# -*- coding: utf-8 -*-
"""
评估指标。

分两类：
- **分类质量**：衡量「涨跌方向预测」本身准不准（AUC / 准确率 / Brier 等）。
- **策略绩效**：衡量「按信号交易」的收益风险特征（年化 / 夏普 / 最大回撤等）。

两类指标必须同时看。一个常见的陷阱是：模型 AUC 只有 0.52（几乎没预测力），
但只要基准上涨概率够高，回测曲线依然可能是正的——那是市场给的，不是模型的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ANNUAL_FACTOR = 244  # A 股年均交易日


# ------------------------------------------------------------------ 分类质量
def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """计算方向预测的分类指标。

    注意 ``majority_acc``（多数类基线）—— 如果模型准确率没有明显超过它，
    说明模型其实什么都没学到。这是评估中最容易被忽略的对照项。
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    base_rate = float(y_true.mean())
    out = {
        "n_samples": int(len(y_true)),
        "base_rate": base_rate,  # 上涨样本占比
        "majority_acc": float(max(base_rate, 1.0 - base_rate)),  # 多数类基线
    }

    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = float("nan")  # 测试集只有一个类别时无法计算

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["brier"] = float(brier_score_loss(y_true, y_prob))

    # 相对基线的增量：> 0 才说明模型有正贡献
    out["auc_edge"] = out["auc"] - 0.5 if not np.isnan(out["auc"]) else float("nan")
    out["acc_edge"] = out["accuracy"] - out["majority_acc"]
    return out


def information_coefficient(y_prob, forward_return) -> float:
    """信息系数 IC：预测概率与未来收益的 Spearman 秩相关。

    IC 是量化里更「诚实」的指标——它不依赖任何阈值，直接衡量
    预测值与真实收益的单调关系。|IC| < 0.03 基本等同于噪音。
    """
    prob = pd.Series(y_prob).reset_index(drop=True)
    ret = pd.Series(forward_return).reset_index(drop=True)
    valid = prob.notna() & ret.notna()
    if valid.sum() < 10:
        return float("nan")
    return float(prob[valid].corr(ret[valid], method="spearman"))


# ------------------------------------------------------------------ 策略绩效
def performance_metrics(
    daily_returns, annual_factor: int = ANNUAL_FACTOR, risk_free: float = 0.0
) -> dict:
    """由日频收益序列计算绩效指标。

    Parameters
    ----------
    daily_returns:
        日收益率序列（小数形式，例如 0.0012 表示 +0.12%）。
    annual_factor:
        年化因子，A 股取 244。
    risk_free:
        年化无风险利率，默认 0（便于横向对比，不引入额外假设）。
    """
    r = pd.Series(daily_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    empty = {
        "total_return": float("nan"),
        "annual_return": float("nan"),
        "annual_vol": float("nan"),
        "sharpe": float("nan"),
        "max_drawdown": float("nan"),
        "calmar": float("nan"),
        "win_rate": float("nan"),
        "n_days": 0,
        "n_trades": 0,
    }
    if r.empty:
        return empty

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    years = len(r) / annual_factor
    if years > 0 and (1.0 + total_return) > 0:
        annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    else:
        annual_return = float("nan")

    annual_vol = float(r.std(ddof=1) * np.sqrt(annual_factor)) if len(r) > 1 else float("nan")
    sharpe = (
        float((r.mean() * annual_factor - risk_free) / annual_vol)
        if annual_vol and not np.isnan(annual_vol) and annual_vol > 0
        else float("nan")
    )

    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = (
        float(annual_return / abs(max_drawdown))
        if max_drawdown and max_drawdown < 0
        else float("nan")
    )

    # 交易次数：持仓状态发生变化的次数
    active = (r != 0).astype(int)
    n_trades = int((active.diff().fillna(0) == 1).sum())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": float((r > 0).mean()),
        "n_days": int(len(r)),
        "n_trades": n_trades,
    }


def equity_curve(daily_returns, start: float = 1.0) -> pd.Series:
    """由日收益序列还原净值曲线。

    注意语义：返回序列的第 i 个值等于 ``start × ∏(1 + r[0..i])``，
    也就是说**第一个值是「首日结束时」的净值**，而不是 1.0。
    做相对比较与绘图时这不影响结论，但如果你需要严格从 1.0 出发的曲线，
    自行在序列前补一个 ``start`` 即可。
    """
    r = pd.Series(daily_returns, dtype=float).fillna(0.0)
    return start * (1.0 + r).cumprod()


__all__ = [
    "classification_metrics",
    "information_coefficient",
    "performance_metrics",
    "equity_curve",
    "ANNUAL_FACTOR",
]
