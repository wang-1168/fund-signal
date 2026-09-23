# -*- coding: utf-8 -*-
"""pytest 公共夹具。

所有测试都只依赖**离线样例数据**，不联网，保证 CI 稳定可重复。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fund_signal import data as dt  # noqa: E402
from fund_signal import features as ft  # noqa: E402
from fund_signal.config import SAMPLE_DIR  # noqa: E402


# ------------------------------------------------------------------ 合成数据
@pytest.fixture(scope="session")
def synthetic_nav() -> pd.DataFrame:
    """确定性的合成净值序列（几何随机游走）。

    固定随机种子，保证跨机器结果一致。
    """
    rng = np.random.default_rng(20260923)
    n = 800
    dates = pd.bdate_range("2020-01-01", periods=n)
    ret = rng.normal(0.0004, 0.012, n)
    nav = 1.0 * np.cumprod(1.0 + ret)
    return pd.DataFrame({"date": dates, "nav": nav})


@pytest.fixture(scope="session")
def synthetic_index() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 800
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 3000 * np.cumprod(1.0 + rng.normal(0.0003, 0.010, n))
    return pd.DataFrame({"date": dates, "close": close})


# ------------------------------------------------------------------ 离线真实数据
@pytest.fixture(scope="session")
def offline_nav() -> pd.DataFrame:
    """仓库自带的离线样例净值（真实基金 000001）。"""
    if not (SAMPLE_DIR / "000001_nav.csv").exists():
        pytest.skip("缺少离线样例数据，请先运行 scripts/build_sample_data.py")
    return dt.load_sample_nav("000001")


@pytest.fixture(scope="session")
def offline_index() -> pd.DataFrame:
    if not (SAMPLE_DIR / "index_sh000300.csv").exists():
        pytest.skip("缺少离线样例指数，请先运行 scripts/build_sample_data.py")
    return dt.load_sample_index("sh000300")


@pytest.fixture(scope="session")
def dataset(offline_nav, offline_index):
    """基于离线真实数据构建的数据集（session 级，只算一次）。"""
    return ft.build_dataset(offline_nav, offline_index, horizon=1)
