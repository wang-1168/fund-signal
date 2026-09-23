# -*- coding: utf-8 -*-
"""数据层测试。

只测试**不依赖网络**的部分：列名归一化、离线数据加载、参数校验。
实时接口的行为放在集成测试里（需要联网，CI 中默认跳过）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from fund_signal import data as dt


# ------------------------------------------------------------------ 离线数据
def test_load_sample_nav(offline_nav):
    assert {"date", "nav"}.issubset(offline_nav.columns)
    assert pd.api.types.is_datetime64_any_dtype(offline_nav["date"])
    assert offline_nav["date"].is_monotonic_increasing, "必须按日期升序"
    assert offline_nav["date"].is_unique, "交易日期不应重复"
    assert (offline_nav["nav"] > 0).all(), "单位净值必须为正"


def test_load_sample_index(offline_index):
    assert {"date", "close"}.issubset(offline_index.columns)
    assert offline_index["date"].is_monotonic_increasing
    assert (offline_index["close"] > 0).all()


def test_list_sample_codes():
    codes = dt.list_sample_codes()
    assert "000001" in codes, "样例数据至少应包含 000001"


def test_load_missing_sample_raises():
    with pytest.raises(FileNotFoundError):
        dt.load_sample_nav("999999")


# ------------------------------------------------------------------ 列名归一化
def test_normalize_nav_chinese_columns():
    raw = pd.DataFrame(
        {
            "净值日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "单位净值": [1.0, 1.01, 1.02],
            "日增长率": [0.0, 1.0, 0.99],
        }
    )
    out = dt._normalize_nav(raw, "000001")
    assert list(out.columns) == ["date", "nav"]
    assert len(out) == 3
    assert out["nav"].iloc[-1] == pytest.approx(1.02)


def test_normalize_nav_english_columns():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "nav": [1.0, 1.5],
        }
    )
    out = dt._normalize_nav(raw, "000001")
    assert len(out) == 2


def test_normalize_nav_drops_duplicates_and_sorts():
    raw = pd.DataFrame(
        {
            "净值日期": ["2024-01-04", "2024-01-02", "2024-01-02"],
            "单位净值": [1.02, 1.00, 1.00],
        }
    )
    out = dt._normalize_nav(raw, "000001")
    assert len(out) == 2, "重复日期应被去重"
    assert out["date"].is_monotonic_increasing


def test_normalize_nav_rejects_unknown_columns():
    raw = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError, match="无法识别净值表列名"):
        dt._normalize_nav(raw, "000001")


def test_normalize_nav_rejects_empty():
    with pytest.raises(ValueError):
        dt._normalize_nav(pd.DataFrame(), "000001")


def test_normalize_index():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [1, 2],
            "high": [1, 2],
            "low": [1, 2],
            "close": [3000.5, 3010.0],
            "volume": [1, 2],
        }
    )
    out = dt._normalize_index(raw, "sh000300")
    assert list(out.columns) == ["date", "close"]
    assert out["close"].iloc[-1] == pytest.approx(3010.0)


# ------------------------------------------------------------------ 参数校验
@pytest.mark.parametrize("bad", ["12345", "1234567", "abcdef", "", None, 123456])
def test_fetch_fund_nav_rejects_bad_code(bad):
    with pytest.raises(ValueError):
        dt.fetch_fund_nav(bad)


# ------------------------------------------------------------------ 体检告警
def test_sanity_check_warns_on_flat_nav(caplog):
    flat = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=100),
            "nav": [1.0] * 100,
        }
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="fund_signal"):
        dt._sanity_check_nav(flat, "000999")
    assert any("货币基金" in r.message or "恒定" in r.message for r in caplog.records), (
        "净值恒定时应给出提示"
    )


# ------------------------------------------------------------------ 缓存键
def test_cache_key_is_stable_and_sensitive():
    a = dt.cache_key("nav", "v2", "000001")
    b = dt.cache_key("nav", "v2", "000001")
    c = dt.cache_key("nav", "v2", "000002")
    assert a == b, "相同参数必须生成相同缓存键"
    assert a != c, "不同参数必须生成不同缓存键"
