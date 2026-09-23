# -*- coding: utf-8 -*-
"""
特征工程。

设计原则（重要）
----------------
1. **严格避免未来函数**：第 t 行的所有特征，只能用第 t 日收盘及之前的信息。
   公募基金净值于每个交易日收盘后公布，因此「用 t 日净值算信号 → t+1 日交易」
   在实盘上是可执行的，不构成信息穿越。
2. **标签独立**：标签 y(t) 由 t+1..t+horizon 的净值决定，与特征在时间上错开，
   训练前必须丢弃尾部 horizon 行。
3. 所有滚动统计量均使用「向后看」窗口（``rolling`` 默认行为），不使用
   居中窗口或全样本统计量。

如果后续有人要扩展特征，请务必保持以上三点，否则回测结果会虚高。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    CALENDAR_FEATURES,
    MARKET_FEATURES,
    PRICE_FEATURES,
)
from .utils import get_logger

log = get_logger()

# 年化因子：中国 A 股市场一年约 244 个交易日
ANNUAL_FACTOR = 244


# ------------------------------------------------------------------ 单指标
def _rsi(nav: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标（Wilder 平滑）。"""
    delta = nav.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    # avg_loss 为 0 时 RS 无穷大，对应 RSI = 100
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(avg_loss != 0.0, 100.0).where(avg_gain != 0.0, 0.0)


def _macd(
    nav: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD：返回 (DIF, DEA, 柱状图)。"""
    ema_fast = nav.ewm(span=fast, adjust=False).mean()
    ema_slow = nav.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2.0
    return dif, dea, hist


# ------------------------------------------------------------------ 净值特征
def compute_price_features(nav: pd.Series) -> pd.DataFrame:
    """基于基金单位净值序列构造价格类特征。

    Parameters
    ----------
    nav:
        以交易日为索引（升序）、值为单位净值的 Series。

    Returns
    -------
    DataFrame
        与 ``nav`` 同索引的特征表。
    """
    nav = nav.astype(float)
    out = pd.DataFrame(index=nav.index)

    # --- 动量 / 收益率 ---
    for w in (1, 5, 10, 20):
        out[f"ret_{w}"] = nav.pct_change(w)

    # --- 波动率（年化） ---
    daily_ret = nav.pct_change()
    out["vol_5"] = daily_ret.rolling(5, min_periods=5).std() * np.sqrt(ANNUAL_FACTOR)
    out["vol_20"] = daily_ret.rolling(20, min_periods=10).std() * np.sqrt(ANNUAL_FACTOR)

    # --- 均线偏离度 ---
    for w in (5, 20, 60):
        ma = nav.rolling(w, min_periods=max(2, w // 2)).mean()
        out[f"ma_gap_{w}"] = nav / ma - 1.0

    # --- 技术指标 ---
    out["rsi_14"] = _rsi(nav, 14) / 100.0  # 归一化到 [0, 1]
    out["macd_hist"] = _macd(nav)[2] / nav  # 除以净值做量纲统一

    # --- 乖离率 ---
    ma20 = nav.rolling(20, min_periods=10).mean()
    out["bias_20"] = (nav - ma20) / ma20

    # --- 区间最大回撤（滚动近似） ---
    roll_max = nav.rolling(20, min_periods=5).max()
    drawdown = nav / roll_max - 1.0
    out["max_dd_20"] = drawdown.rolling(20, min_periods=5).min()

    # --- 上涨天数占比 ---
    out["up_days_10"] = (daily_ret > 0).rolling(10, min_periods=5).mean()

    # --- 日历效应 ---
    idx = pd.DatetimeIndex(out.index)
    out["dow"] = idx.dayofweek
    out["month"] = idx.month

    return out


# ------------------------------------------------------------------ 市场特征
def compute_market_features(nav: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """构造与基准指数相关的特征（市场环境 + 相对强弱）。

    两个序列会先按日期取交集对齐，避免因交易日历差异产生错位。
    """
    df = pd.DataFrame({"nav": nav.astype(float), "bench": benchmark.astype(float)})
    df = df.dropna()
    if df.empty:
        raise ValueError("基金净值与基准指数没有重叠的交易日，请检查代码是否正确。")

    out = pd.DataFrame(index=df.index)
    bench_ret = df["bench"].pct_change()

    for w in (1, 5, 20):
        out[f"idx_ret_{w}"] = df["bench"].pct_change(w)

    fund_ret_20 = df["nav"].pct_change(20)
    out["rel_strength_20"] = fund_ret_20 - out["idx_ret_20"]

    out["corr_20"] = df["nav"].pct_change().rolling(20, min_periods=10).corr(bench_ret)
    return out


# ------------------------------------------------------------------ 组装
@dataclass
class Dataset:
    """建模数据集。

    这里刻意把「训练样本」与「最新特征行」分开，因为二者**不是同一行**：

    - ``frame`` 需要标签（即未来收益已经发生），因此它的最后一行必然
      落在 ``horizon`` 个交易日之前；
    - ``latest`` 是**最新一个交易日**的特征——它的标签还没发生，
      正是我们要预测的对象。

    很多「基金预测」脚本会把这两者混为一谈，导致模型永远在预测已经
    发生过的行情（甚至直接用到了未来数据）。分开建模可以从结构上杜绝这类错误。

    Attributes
    ----------
    frame:
        训练 / 回测用样本，含特征列 + ``label`` + ``nav`` + ``fwd_ret``。
    latest:
        单行 DataFrame，最新交易日的完整特征（无标签）。
    feature_names:
        特征列名列表。
    """

    frame: pd.DataFrame
    latest: pd.DataFrame
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def latest_date(self):
        """最新特征行对应的交易日。"""
        return self.latest.index[-1] if len(self.latest) else None


def build_dataset(
    nav_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None, horizon: int = 1
) -> Dataset:
    """组装完整建模数据集。

    Parameters
    ----------
    nav_df:
        含 ``date`` 与 ``nav`` 两列，按日期升序。
    benchmark_df:
        含 ``date`` 与 ``close`` 两列；为 None 时跳过市场特征。
    horizon:
        预测未来多少个交易日。

    Returns
    -------
    Dataset
        ``frame`` 为训练/回测样本（已剔除特征 NaN 与尾部无法定标签的行），
        ``latest`` 为最新交易日的特征行（待预测对象）。
    """
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")

    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.sort_values("date").drop_duplicates("date")
    nav = nav_df.set_index("date")["nav"].astype(float)

    parts = [compute_price_features(nav)]

    if benchmark_df is not None and not benchmark_df.empty:
        bench_df = benchmark_df.copy()
        bench_df["date"] = pd.to_datetime(bench_df["date"])
        bench_df = bench_df.sort_values("date").drop_duplicates("date")
        bench = bench_df.set_index("date")["close"].astype(float)

        # 指数与基金交易日对齐：指数用前值填充到基金的交易日
        bench = bench.reindex(nav.index.union(bench.index)).ffill().reindex(nav.index)
        try:
            parts.append(compute_market_features(nav, bench))
        except ValueError as err:
            log.warning("市场特征构造失败，已跳过：%s", err)

    features = pd.concat(parts, axis=1)

    # --- 标签与「可交易收益」 ---
    # 时点约定（与公募基金真实交易流程一致）：
    #   T 日收盘后公布净值 → 当晚算出特征 X[T] 与信号；
    #   T+1 日 15:00 前下单，按 T+1 日净值成交（买入价 = nav[T+1]）；
    #   持有 horizon 个交易日后，按 nav[T+1+horizon] 卖出。
    #
    # 因此可交易收益必须以 T+1 日为起点。若用 nav[T] 当买入价，
    # 相当于假设"看完 T 日净值还能按 T 日净值成交"，凭空多赚一天，
    # 这是开源量化项目里最常见的回测虚高来源。
    buy_nav = nav.shift(-1)
    sell_nav = nav.shift(-(1 + horizon))
    fwd_ret = sell_nav / buy_nav - 1.0
    label = (fwd_ret > 0).astype(float)
    label[fwd_ret.isna()] = np.nan  # 尾部无法定标签

    dataset = features.copy()
    dataset["label"] = label
    dataset["nav"] = nav
    dataset["fwd_ret"] = fwd_ret

    feature_only = [c for c in dataset.columns if c not in ("label", "nav", "fwd_ret")]

    dataset = dataset.replace([np.inf, -np.inf], np.nan)

    # --- 最新一行：特征齐全但标签尚未发生，这才是「待预测」的对象 ---
    complete = dataset[feature_only].notna().all(axis=1)
    if complete.any():
        latest = dataset.loc[complete, feature_only].iloc[[-1]].copy()
    else:
        latest = pd.DataFrame(columns=feature_only)

    before = len(dataset)
    frame = dataset.dropna()
    log.info(
        "数据集构建完成：%d 行 → %d 行可用样本（horizon=%d，剔除 %d 行 NaN/尾部）",
        before,
        len(frame),
        horizon,
        before - len(frame),
    )

    if len(frame) < 120:
        log.warning("可用样本仅 %d 行，模型结果可能不稳定，建议拉长历史区间。", len(frame))
    if len(latest):
        log.info("最新特征行：%s（用于预测下一期）", pd.Timestamp(latest.index[-1]).date())

    return Dataset(frame=frame, latest=latest, feature_names=feature_only)


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """从数据集中挑出特征列（排除 label 及辅助列）。"""
    excluded = {"label", "nav", "fwd_ret"}
    return [c for c in dataset.columns if c not in excluded]


__all__ = [
    "Dataset",
    "compute_price_features",
    "compute_market_features",
    "build_dataset",
    "feature_columns",
    "PRICE_FEATURES",
    "MARKET_FEATURES",
    "CALENDAR_FEATURES",
]
