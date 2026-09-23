# -*- coding: utf-8 -*-
"""
fund-signal —— 公募基金量化信号分析框架。

本包提供从数据获取、特征工程、时序建模到回测评估的完整链路，
用于研究「基金净值的统计规律」，而非提供投资建议。

主要模块
--------
data       : 基金净值与基准指数数据获取（akshare）+ 本地缓存
features   : 特征工程（严格避免未来函数）
model      : 时序交叉验证与模型训练（LightGBM / 逻辑回归）
backtest   : 基于概率阈值的策略回测
metrics    : 分类与绩效评估指标
pipeline   : 端到端流水线（CLI 与 Streamlit 共用）

顶层快捷用法
------------
::

    from fund_signal import Config, run_analysis

    cfg = Config(fund_code="000001", horizon=1)
    result = run_analysis(cfg, data_source="local")
    print(result.latest_prob)

.. note::
   为避免 `import fund_signal` 时就把 akshare 这类重依赖拉起来，
   除 ``Config`` 外的高层对象采用惰性导入（PEP 562 模块级 ``__getattr__``）。
   首次访问时才会真正加载对应模块。
"""

from __future__ import annotations

import warnings as _warnings

from .config import Config

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Config",
    "AnalysisResult",
    "run_analysis",
    "run_from_code",
    "list_sample_codes",
]

# 抑制 akshare 内部的部分警告噪音，避免污染用户终端输出
_warnings.filterwarnings("ignore", category=FutureWarning, module="akshare")
_warnings.filterwarnings("ignore", message=".*pkg_resources.*")


# ---------------------------------------------------------------- 惰性导出
# 名字 -> (子模块, 属性名)
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AnalysisResult": ("fund_signal.pipeline", "AnalysisResult"),
    "run_analysis": ("fund_signal.pipeline", "run_analysis"),
    "run_from_code": ("fund_signal.pipeline", "run_from_code"),
    "list_sample_codes": ("fund_signal.data", "list_sample_codes"),
}


def __getattr__(name: str):
    """按需加载高层 API，避免 `import fund_signal` 触发 akshare 全量导入。"""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
