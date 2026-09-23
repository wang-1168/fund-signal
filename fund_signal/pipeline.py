# -*- coding: utf-8 -*-
"""
端到端分析流水线。

把 ``data → features → model → backtest`` 串成一条可复用的链路。
Streamlit 界面与命令行都调用本模块，保证两个入口跑出来的结果完全一致。

最重要的一条纪律
----------------
回测**只使用滚动前向验证产生的样本外预测**（``walk_forward.predictions``），
绝不使用最终模型在训练集上的预测。后者是「用答案考自己」，任何项目这么做
都能画出一条漂亮曲线，但毫无意义。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import audit
from . import backtest as bt
from . import data as dt
from . import features as ft
from . import metrics as mt
from . import model as md
from .audit import accuracy_to_auc
from .config import Config
from .utils import get_logger

log = get_logger()

ProgressFn = Callable[[str, float], None]
DataSourceName = str  # "akshare" | "local"


@dataclass
class AnalysisResult:
    """一次完整分析的产物。"""

    config: Config
    fund_name: str
    nav: pd.DataFrame  # 原始净值
    dataset: ft.Dataset  # 特征 + 标签 + 最新行
    wf: md.WalkForwardResult  # 滚动前向验证结果
    importance: pd.Series  # 特征重要性
    latest_prob: float  # 最新一期上涨概率
    latest_date: pd.Timestamp | None  # 最新特征行日期
    backtest: bt.BacktestResult  # 回测结果
    equity: pd.DataFrame  # 净值曲线对照
    realtime: dict = field(default_factory=dict)  # 元信息 / 最新净值 / 盘中估值
    notes: list[str] = field(default_factory=list)  # 过程中的警告/提示
    capability: dict = field(default_factory=dict)  # 能力边界体检
    raw_metrics: dict = field(default_factory=dict)  # 校准前概率的指标

    # ---------------------------------------------------------------- 实时数据
    @property
    def last_nav_date(self) -> pd.Timestamp | None:
        """净值序列的最后一个交易日。"""
        if self.nav is None or self.nav.empty:
            return None
        return pd.Timestamp(self.nav["date"].iloc[-1])

    @property
    def realtime_estimate(self) -> dict | None:
        """盘中估值。

        仅当数据源覆盖该基金、且处于交易时段时才有值；
        其余情况返回 None（这是数据源覆盖范围决定的，不是错误）。
        """
        return self.realtime.get("estimate")

    @property
    def data_age_days(self) -> float:
        """净值数据距今天数，用于判断数据是否足够新。"""
        last = self.last_nav_date
        if last is None:
            return float("nan")
        delta = pd.Timestamp.now().normalize() - last.normalize()
        return float(delta.days)

    # ---------------------------------------------------------------- 便捷属性
    @property
    def feature_names(self) -> list[str]:
        return self.dataset.feature_names

    @property
    def auc(self) -> float:
        return float(self.wf.overall_metrics.get("auc", float("nan")))

    @property
    def has_edge(self) -> bool:
        """模型是否显著优于「永远猜多数类」的基线。

        这是本项目最该被关注的一个指标：若为 False，
        说明模型的预测能力与抛硬币没有统计上的差别。
        """
        edge = self.wf.overall_metrics.get("acc_edge", float("nan"))
        return bool(not np.isnan(edge) and edge > 0.02)

    def summary_metrics(self) -> dict:
        """汇总关键指标（英文键，便于程序消费）。"""
        c = self.wf.overall_metrics
        s = self.backtest.strategy_metrics
        b = self.backtest.benchmark_metrics
        return {
            "fund_name": self.fund_name,
            "n_samples": len(self.dataset),
            "n_features": len(self.feature_names),
            "auc": c.get("auc"),
            "accuracy": c.get("accuracy"),
            "majority_acc": c.get("majority_acc"),
            "acc_edge": c.get("acc_edge"),
            "base_rate": c.get("base_rate"),
            "latest_prob": self.latest_prob,
            "latest_date": self.latest_date,
            "strategy_return": s.get("total_return"),
            "benchmark_return": b.get("total_return"),
            "excess_return": self.backtest.excess_return,
            "sharpe": s.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"),
            "exposure": self.backtest.exposure,
            "has_edge": self.has_edge,
        }


def run_analysis(
    config: Config | None = None,
    data_source: DataSourceName = "akshare",
    progress: ProgressFn | None = None,
    model_params: dict | None = None,
) -> AnalysisResult:
    """执行一次完整分析。

    Parameters
    ----------
    config:
        :class:`~fund_signal.config.Config` 实例；为 None 时使用默认参数。
    data_source:
        ``"akshare"`` 联网取数；``"local"`` 读取随仓库附带的离线样例数据
        （无需网络，用于快速体验与 CI 测试）。
    progress:
        进度回调 ``fn(消息, 进度0~1)``，供界面显示进度条。
    model_params:
        透传给底层分类器的额外参数。

    Returns
    -------
    AnalysisResult
    """
    cfg = config or Config()
    cfg.validate()

    notes: list[str] = []

    def emit(msg: str, frac: float) -> None:
        log.info(msg)
        if progress is not None:
            try:
                progress(msg, frac)
            except Exception:  # 界面回调异常不应影响主流程
                pass

    # ---------------------------------------------------------- 1. 净值
    emit(f"获取基金 {cfg.fund_code} 历史净值…", 0.05)
    realtime_info: dict = {}
    if data_source == "local":
        nav = dt.load_sample_nav(cfg.fund_code)
        fund_name = f"{cfg.fund_code}（离线样例）"
        notes.append("当前使用仓库自带的离线样例数据，仅用于演示流程，与实时行情可能存在差异。")
    else:
        nav = dt.fetch_fund_nav(cfg.fund_code, use_cache=cfg.use_cache)
        meta = dt.fetch_fund_meta(cfg.fund_code, use_cache=cfg.use_cache)
        fund_name = meta.get("name") or cfg.fund_code
        realtime_info["meta"] = meta

        if cfg.realtime:
            # ---- 把净值序列补齐到最新交易日 ----
            emit("核对最新交易日净值…", 0.10)
            latest = None
            try:
                latest = dt.fetch_latest_nav(cfg.fund_code, use_cache=cfg.use_cache)
            except Exception as err:
                log.warning("最新净值校验失败（沿用历史接口数据）：%s", err)

            if latest is not None and latest["date"] > nav["date"].iloc[-1]:
                nav = (
                    pd.concat([nav, pd.DataFrame([latest])], ignore_index=True)
                    .sort_values("date")
                    .drop_duplicates("date")
                    .reset_index(drop=True)
                )
                notes.append(f"已补齐最新交易日净值：{latest['date'].date()} = {latest['nav']:.4f}")
                realtime_info["appended"] = latest
            elif latest is not None:
                realtime_info["verified"] = latest

            # ---- 盘中估值（仅交易时段有值，且数据源不覆盖全部基金）----
            try:
                est = dt.fetch_realtime_estimate(cfg.fund_code, use_cache=cfg.use_cache)
                if est is not None:
                    realtime_info["estimate"] = est
            except Exception as err:
                log.warning("实时估值获取失败（不影响主流程）：%s", err)

    # ---------------------------------------------------------- 2. 基准指数
    benchmark_df = None
    emit(f"获取基准指数 {cfg.benchmark}…", 0.15)
    try:
        if data_source == "local":
            benchmark_df = dt.load_sample_index(cfg.benchmark)
        else:
            benchmark_df = dt.fetch_index(cfg.benchmark, use_cache=cfg.use_cache)
    except Exception as err:
        notes.append(f"基准指数获取失败，已跳过市场类特征（{err}）。")
        log.warning("基准指数获取失败，跳过市场类特征：%s", err)

    # ---------------------------------------------------------- 3. 数据集
    emit("构建特征与标签…", 0.30)
    dataset = ft.build_dataset(nav, benchmark_df, horizon=cfg.horizon)

    if len(dataset) < 150:
        raise ValueError(
            f"可用样本仅 {len(dataset)} 行，不足以做时序验证（建议 ≥ 350 行）。\n"
            "可尝试：换一只成立时间更长的基金 / 把 horizon 调小 / 降低 n_splits。"
        )

    X = dataset.frame[dataset.feature_names]
    y = dataset.frame["label"].astype(int)

    # ---------------------------------------------------------- 4. 滚动前向验证
    emit(f"滚动前向验证（{cfg.n_splits} 折）…", 0.45)
    wf = md.walk_forward_validate(
        X,
        y,
        model_type=cfg.model_type,
        n_splits=cfg.n_splits,
        threshold=cfg.threshold,
        random_state=cfg.random_state,
        model_params=model_params,
        embargo=cfg.horizon if cfg.embargo else 0,
        calibrate=cfg.calibrate,
        verbose=True,
    )
    if cfg.calibrate and wf.calibrate_method == "none":
        notes.append("样本量不足，本折未启用概率校准（沿用原始概率）。")
    elif wf.calibrate_method != "none":
        notes.append(
            f"已启用样本外概率校准（{wf.calibrate_method}）：训练集尾部 20% "
            "专用于拟合校准器，模型本体未见这部分样本。"
        )

    # ---------------------------------------------------------- 5. 最终模型
    emit("训练最终模型、生成最新信号…", 0.70)
    final_model = md.train_final_model(
        X,
        y,
        model_type=cfg.model_type,
        random_state=cfg.random_state,
        model_params=model_params,
    )
    importance = md.feature_importance(final_model, dataset.feature_names)

    latest_prob = float("nan")
    if len(dataset.latest):
        latest_prob = md.predict_latest(final_model, dataset.latest)
    else:
        notes.append("未能构造最新特征行，可能是历史数据过短。")

    # ---------------------------------------------------------- 6. 回测
    # 注意：使用 wf.predictions（样本外），不是 final_model 的样本内预测
    emit("样本外回测…", 0.85)
    oos_prob = wf.predictions
    oos_forward = dataset.frame.loc[oos_prob.index, "fwd_ret"]
    backtest_result = bt.backtest_threshold(
        oos_prob,
        oos_forward,
        threshold=cfg.threshold,
        cost_bps=cfg.cost_bps,
    )
    equity = bt.build_equity_frame(backtest_result)

    # ---------------------------------------------------------- 7. 能力边界体检
    emit("能力边界体检…", 0.95)
    capability: dict = {}
    raw_metrics: dict = {}
    try:
        capability = audit.capability_report(oos_prob, y.loc[oos_prob.index], target_accuracy=0.90)
        raw_metrics = mt.classification_metrics(
            y.loc[oos_prob.index].to_numpy(),
            (wf.raw_predictions if wf.raw_predictions is not None else oos_prob)
            .loc[oos_prob.index]
            .to_numpy(),
            cfg.threshold,
        )
    except Exception as err:  # 体检失败不应影响主流程
        log.warning("能力边界体检失败（不影响主流程）：%s", err)

    # ---------------------------------------------------------- 8. 结语
    if not (wf.overall_metrics.get("acc_edge") or 0) > 0.02:
        notes.append(
            "模型准确率未明显超过「多数类基线」——这在基金日频预测中是常见结果，"
            "说明该基金在未来一个交易日上的方向变化接近随机。"
        )
    need_auc = accuracy_to_auc(0.90)
    notes.append(
        f"能力边界：准确率 90% 需要 AUC ≈ {need_auc:.3f}，本次实测 AUC "
        f"= {wf.overall_metrics.get('auc', float('nan')):.3f}。"
        "想靠调参补齐这个缺口，只会得到过拟合（见「精度审计」页签的复杂度曲线）。"
    )

    emit("分析完成", 1.0)

    return AnalysisResult(
        config=cfg,
        fund_name=fund_name,
        nav=nav,
        dataset=dataset,
        wf=wf,
        importance=importance,
        latest_prob=latest_prob,
        latest_date=dataset.latest_date,
        backtest=backtest_result,
        equity=equity,
        realtime=realtime_info,
        notes=notes,
        capability=capability,
        raw_metrics=raw_metrics,
    )


def run_from_code(fund_code: str, **kwargs) -> AnalysisResult:
    """便捷入口：只给一个基金代码就能跑完整流程。"""
    cfg = kwargs.pop("config", None) or Config(fund_code=fund_code)
    cfg.fund_code = fund_code
    return run_analysis(config=cfg, **kwargs)


__all__ = ["AnalysisResult", "run_analysis", "run_from_code"]
