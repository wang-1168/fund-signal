# -*- coding: utf-8 -*-
"""
全局配置。

所有可调参数集中在此处，方便调用方通过 ``Config`` 一次性传递，
也方便命令行 / Streamlit 界面复用同一套默认值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------- 路径

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
MODEL_DIR: Path = PROJECT_ROOT / "models"
SAMPLE_DIR: Path = PROJECT_ROOT / "examples" / "sample_data"

for _d in (DATA_DIR, CACHE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 类型别名

ModelType = Literal["lightgbm", "logistic"]
DataSource = Literal["akshare", "local"]

# ---------------------------------------------------------------- 默认基准指数
# 用于构造「市场环境」类特征。键为中文可读名，值为 akshare 指数代码。
DEFAULT_BENCHMARKS: dict[str, str] = {
    "沪深300": "sh000300",
    "中证500": "sh000905",
    "创业板指": "sz399006",
}


@dataclass
class Config:
    """一次完整分析任务的全部参数。

    Parameters
    ----------
    fund_code:
        6 位公募基金代码，例如 ``"000001"``。
    horizon:
        预测未来多少个交易日。1 表示预测下一个交易日涨跌。
    benchmark:
        基准指数 akshare 代码，用于构造相对强弱特征。
    model_type:
        ``"lightgbm"`` 或 ``"logistic"``。
    threshold:
        回测中判定为「看多」的概率阈值，默认 0.5。
    train_ratio:
        单次留出法评估中训练集占比（按时间顺序切分）。
    n_splits:
        walk-forward 滚动前向验证的折数。
    cost_bps:
        单边交易成本，单位基点（1 bps = 0.01%）。
        公募基金 C 类无申购费但有销售服务费，A 类申购费通常 0.15% 左右，
        此处默认 15 bps 以贴近实际。
    random_state:
        随机种子，保证结果可复现。
    use_cache:
        是否使用本地缓存，默认开启以减少对数据源的请求。
    realtime:
        是否启用实时数据能力：分析前把净值序列补齐到最新交易日，
        并尝试抓取盘中估值。关闭后只用历史净值接口的数据（更省流量）。
    """

    fund_code: str = "000001"
    horizon: int = 1
    benchmark: str = DEFAULT_BENCHMARKS["沪深300"]

    model_type: ModelType = "lightgbm"
    threshold: float = 0.5

    train_ratio: float = 0.7
    n_splits: int = 5
    cost_bps: float = 15.0
    random_state: int = 42
    use_cache: bool = True
    realtime: bool = True

    def to_dict(self) -> dict:
        """导出为普通字典（便于日志与界面展示）。"""
        return asdict(self)

    def validate(self) -> None:
        """基础参数校验，尽早暴露调用方的手误。"""
        if not (
            isinstance(self.fund_code, str)
            and len(self.fund_code) == 6
            and self.fund_code.isdigit()
        ):
            raise ValueError(f"fund_code 必须是 6 位数字字符串，收到 {self.fund_code!r}")
        if self.horizon < 1:
            raise ValueError(f"horizon 必须 >= 1，收到 {self.horizon}")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"threshold 必须在 (0, 1) 之间，收到 {self.threshold}")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError(f"train_ratio 必须在 (0, 1) 之间，收到 {self.train_ratio}")
        if self.n_splits < 2:
            raise ValueError(f"n_splits 必须 >= 2，收到 {self.n_splits}")
        if self.model_type not in ("lightgbm", "logistic"):
            raise ValueError(f"未知 model_type: {self.model_type!r}")
        if self.cost_bps < 0:
            raise ValueError(f"cost_bps 不能为负，收到 {self.cost_bps}")


# 特征列清单（供各模块引用，避免字符串散落各处）
PRICE_FEATURES = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_5",
    "vol_20",
    "ma_gap_5",
    "ma_gap_20",
    "ma_gap_60",
    "rsi_14",
    "macd_hist",
    "bias_20",
    "max_dd_20",
    "up_days_10",
]

MARKET_FEATURES = [
    "idx_ret_1",
    "idx_ret_5",
    "idx_ret_20",
    "rel_strength_20",
    "corr_20",
]

CALENDAR_FEATURES = ["dow", "month"]

ALL_FEATURES = PRICE_FEATURES + MARKET_FEATURES + CALENDAR_FEATURES
