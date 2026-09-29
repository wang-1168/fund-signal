# -*- coding: utf-8 -*-
"""事件模块测试。

三条主线：

1. **日历事件能算对** —— 日期规则（跨年、第 n 个星期几、吸附交易日）正确，
   且历史与未来都能生成；
2. **不能有未来函数** —— 事件日当天的涨跌绝不能被事件窗口"吃到"，
   这是本模块最容易出错、也最致命的地方；
3. **断网要能降级** —— 实时快讯/经济日历取不到时返回空表而不是抛异常，
   否则界面会整块崩掉。

全部离线运行，不联网。

"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from fund_signal import events as ev

# ================================================================= 日期规则


def test_month_shift_crosses_year():
    assert ev._month_shift(date(2025, 11, 1), 3) == date(2026, 2, 1)
    assert ev._month_shift(date(2025, 12, 1), 1) == date(2026, 1, 1)
    assert ev._month_shift(date(2025, 1, 1), 12) == date(2026, 1, 1)
    assert ev._month_shift(date(2025, 6, 1), 18) == date(2026, 12, 1)


def test_nth_weekday():
    # 2025-06 第一个周五 = 06-06，第二个 = 06-13
    assert ev._nth_weekday(2025, 6, 4, 1) == date(2025, 6, 6)
    assert ev._nth_weekday(2025, 6, 4, 2) == date(2025, 6, 13)
    # 2025-12-01 是周一，第 2 个周五 = 12-12
    assert ev._nth_weekday(2025, 12, 4, 2) == date(2025, 12, 12)


def test_snap_skips_weekend_without_calendar():
    """没有交易日历时，周六应吸附到下周一。"""
    assert ev._snap(date(2025, 6, 7), None) == date(2025, 6, 9)  # 周六 → 周一
    assert ev._snap(date(2025, 6, 9), None) == date(2025, 6, 9)  # 周一 → 自身


def test_snap_falls_back_when_calendar_does_not_cover_future():
    """回归测试：交易日历只到"昨天"时，未来事件不能被静默丢弃。

    这是真实踩过的坑——调用方传进来的 ``trade_dates`` 往往只是基金净值
    已有的日期（截止昨天），若直接用它吸附未来事件，会全部匹配失败，
    "接下来有什么大事"整块变空。
    """
    past_only = {date(2025, 6, i) for i in range(1, 10)}  # 只覆盖 6/1~6/9
    future = date(2025, 9, 20)  # 周六，且远在日历覆盖范围之外
    got = ev._snap(future, past_only)
    assert got is not None, "未来事件被静默丢弃了"
    assert got == date(2025, 9, 22)  # 周六 → 周一


# ================================================================= 日程生成


def test_scheduled_events_has_required_columns():
    df = ev.scheduled_events("2025-01-01", "2025-12-31")
    assert not df.empty
    for col in ("date", "name", "category", "importance", "importance_label", "precision", "note"):
        assert col in df.columns, f"缺少列 {col}"
    assert df["date"].is_monotonic_increasing, "结果应按日期升序"
    assert df["importance"].between(1, 3).all()


def test_scheduled_events_lpr_on_20th():
    """LPR 规则日是每月 20 日；20 日落在周末时会顺延到下一交易日。"""
    df = ev.scheduled_events("2025-01-01", "2025-12-31")
    lpr = df[df["name"] == ev.CALENDAR_RULES["lpr"].name]
    assert len(lpr) == 12, "LPR 每月一次，一年应 12 次"
    # 未给交易日历时应为「20 日起的第一个工作日」，即 20/21/22 日且必为工作日
    assert set(lpr["date"].dt.day).issubset({20, 21, 22})
    assert (lpr["date"].dt.weekday < 5).all()
    # 每个月都必须恰好出现一次
    assert lpr["date"].dt.to_period("M").nunique() == 12


def test_scheduled_events_lianghui_only_in_march_and_gdp_quarterly():
    df = ev.scheduled_events("2025-01-01", "2025-12-31")
    lh = df[df["name"] == ev.CALENDAR_RULES["lianghui"].name]
    assert len(lh) == 1 and lh["date"].dt.month.iloc[0] == 3

    gdp = df[df["name"] == ev.CALENDAR_RULES["gdp"].name]
    assert len(gdp) == 4, "GDP 每季度一次"
    assert set(gdp["date"].dt.month) == {1, 4, 7, 10}


def test_scheduled_events_can_generate_future():
    """未来事件必须能算出来——这是"提前知道有什么大事"的前提。"""
    df = ev.scheduled_events("2030-01-01", "2030-12-31")
    assert not df.empty
    assert (df["date"].dt.year == 2030).all()


def test_scheduled_events_rejects_reversed_range():
    with pytest.raises(ValueError):
        ev.scheduled_events("2025-12-31", "2025-01-01")


def test_scheduled_events_snaps_to_real_trade_dates():
    """给了真实交易日历时，事件日必须落在交易日上（跳过周末/假日）。"""
    td = pd.Series(pd.bdate_range("2025-01-01", "2025-12-31"))
    df = ev.scheduled_events("2025-01-01", "2025-12-31", trade_dates=td)
    assert not df.empty
    assert set(df["date"]).issubset(set(td)), "事件日必须在交易日历内"


# ================================================================= 未来事件


def test_upcoming_events_returns_future_only():
    """回归测试：未来事件窗口不能为空（见 _snap 的坑）。"""
    nav = pd.DataFrame({"date": pd.bdate_range("2024-01-01", "2025-08-31"), "nav": 1.0})
    up = ev.upcoming_events(days=30, min_importance=2, trade_dates=nav["date"], as_of="2025-09-01")
    assert not up.empty, "未来 30 天应至少有一个中等以上事件"
    assert (up["date"] >= pd.Timestamp("2025-09-01")).all()
    assert (up["importance"] >= 2).all()


def test_upcoming_events_filters_by_importance():
    lo = ev.upcoming_events(days=60, min_importance=1, as_of="2025-09-01")
    hi = ev.upcoming_events(days=60, min_importance=3, as_of="2025-09-01")
    assert len(hi) <= len(lo)
    assert (hi["importance"] >= 3).all()


# ================================================================= 事件影响


def test_event_impact_empty_inputs():
    nav = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=100), "nav": 1.0})
    assert ev.event_impact(pd.DataFrame(), pd.DataFrame()).empty
    assert ev.event_impact(nav, pd.DataFrame()).empty
    assert ev.event_impact(None, None).empty


@pytest.fixture(scope="module")
def impact_frame():
    rng = np.random.default_rng(11)
    n = 900
    dates = pd.bdate_range("2021-01-01", periods=n)
    nav = pd.DataFrame(
        {
            "date": dates,
            "nav": 1.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.011, n)),
        }
    )
    cal = ev.scheduled_events(dates[0].date(), dates[-1].date(), trade_dates=nav["date"])
    return ev.event_impact(nav, cal, before=0, after=1, horizon=1, min_importance=3)


def test_event_impact_columns_and_summary_first(impact_frame):
    for col in (
        "事件",
        "重要性",
        "精度",
        "样本数",
        "覆盖率",
        "窗口平均收益",
        "非事件期平均收益",
        "收益差",
        "窗口上涨概率",
        "非事件期上涨概率",
        "胜率差",
        "窗口日波动",
        "非事件期日波动",
        "波动比",
    ):
        assert col in impact_frame.columns, f"缺少列 {col}"
    assert impact_frame["事件"].iloc[0].startswith("★"), "合并行应在首行"


def test_event_impact_coverage_bounded(impact_frame):
    cov = impact_frame["覆盖率"]
    assert (cov >= 0).all() and (cov <= 1).all()
    assert float(impact_frame.attrs["coverage"]) <= 1.0


def test_event_impact_narrow_window_is_meaningful(impact_frame):
    """默认窄窗口（事件日 + 次一交易日）的覆盖率不应吞掉大半样本。

    第一版用宽窗口时覆盖率高达 93%，"非事件期"只剩零星样本，对照彻底失效。
    这里把"必须留出足够对照样本"固化成断言。
    """
    assert float(impact_frame.attrs["coverage"]) < 0.5, (
        "事件窗口覆盖率过高，对照失效——请收窄窗口或提高重要性门槛"
    )


def test_event_impact_does_not_capture_same_day_jump():
    """核心不可回归项：事件日当天的涨跌不能被事件窗口赚到。

    构造一条净值：前 10 天恒为 1.0，第 10 天（索引 10）起跳到 1.1 并保持。
    按"T 日净值已知 → T+1 才可成交"的口径，唯一非零的可交易收益出现在
    索引 8（因为真正的跳变发生在 9→10 之间）。把事件放在**跳变当天（索引 10）**，
    窗口只能覆盖索引 10、11，其间可交易收益均为 0——若某处写成
    ``nav.shift(-1)/nav - 1``，这里就会读出 +10%，测试立刻失败。
    """
    n = 20
    dates = pd.bdate_range("2025-01-06", periods=n)
    nav_vals = np.where(np.arange(n) < 10, 1.0, 1.1)
    nav = pd.DataFrame({"date": dates, "nav": nav_vals})

    jump_day = dates[10]
    events = pd.DataFrame(
        [
            {
                "date": jump_day,
                "name": "测试事件",
                "category": "测试",
                "importance": 3,
                "importance_label": "高",
                "precision": "exact",
                "note": "",
            }
        ]
    )
    out = ev.event_impact(nav, events, before=0, after=1, horizon=1)
    assert not out.empty
    row = out.iloc[0]
    assert row["事件"].startswith("★")
    assert row["窗口平均收益"] == pytest.approx(0.0, abs=1e-12), (
        "事件窗口吃到了事件日当天的涨跌 —— 出现未来函数"
    )
    # 而对照期（含索引 8）确实含有那一次跳变，说明数据本身没问题
    assert row["非事件期平均收益"] > 0


def test_event_impact_summary_survives_small_event_samples():
    """回归测试：单个事件类别样本太少时，合并总览行不能跟着消失。

    曾经写成"若没有任何类别通过样本量门槛就直接返回空表"，结果是短区间内
    「事件到底有没有用」根本答不出来——而那正是这一页唯一要回答的问题。
    """
    n = 20
    dates = pd.bdate_range("2025-01-06", periods=n)
    nav = pd.DataFrame({"date": dates, "nav": np.linspace(1.0, 1.1, n)})
    events = pd.DataFrame(
        [
            {
                "date": dates[10],
                "name": "孤立事件",
                "category": "测试",
                "importance": 3,
                "importance_label": "高",
                "precision": "exact",
                "note": "",
            }
        ]
    )
    out = ev.event_impact(nav, events, before=0, after=1, horizon=1)
    assert not out.empty, "样本不足时合并总览行也不该消失"
    assert out["事件"].iloc[0].startswith("★")


def test_event_summary_text_warns_on_high_coverage():
    df = pd.DataFrame(
        [
            {
                "事件": "★ 全部事件窗口合并",
                "重要性": 0,
                "精度": "-",
                "样本数": 100,
                "覆盖率": 0.85,
                "窗口平均收益": 0.0,
                "非事件期平均收益": 0.0,
                "收益差": 0.0,
                "窗口上涨概率": 0.5,
                "非事件期上涨概率": 0.5,
                "胜率差": 0.0,
                "窗口日波动": 0.01,
                "非事件期日波动": 0.01,
                "波动比": 1.0,
            }
        ]
    )
    txt = ev.event_summary_text(df)
    assert "覆盖" in txt
    assert "不可采信" in txt or "失去意义" in txt


def test_event_summary_text_handles_empty():
    assert "没有" in ev.event_summary_text(pd.DataFrame())


# ================================================================= 安慰剂检验


def _make_noisy_nav(seed: int = 3, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "nav": 1.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.011, n)),
        }
    )


def test_placebo_rejects_bad_mode(impact_frame):
    nav = _make_noisy_nav()
    cal = ev.scheduled_events(nav["date"].iloc[0].date(), nav["date"].iloc[-1].date())
    with pytest.raises(ValueError):
        ev.placebo_test(nav, cal, mode="nonsense")


def test_placebo_empty_inputs():
    nav = _make_noisy_nav()
    assert ev.placebo_test(pd.DataFrame(), pd.DataFrame()) == {}
    assert ev.placebo_test(nav, pd.DataFrame()) == {}


@pytest.mark.parametrize("mode", ["shift", "uniform"])
def test_placebo_shape_and_ranges(mode):
    nav = _make_noisy_nav()
    cal = ev.scheduled_events(nav["date"].iloc[0].date(), nav["date"].iloc[-1].date())
    r = ev.placebo_test(
        nav,
        cal,
        before=0,
        after=1,
        horizon=1,
        min_importance=3,
        n_draws=40,
        max_shift=20,
        mode=mode,
    )
    assert r, "应返回结果"
    assert r["模式"] == mode
    assert 0.0 <= r["p_波动"] <= 1.0
    assert 0.0 <= r["p_方向"] <= 1.0
    assert r["随机波动比 P5"] <= r["随机波动比 P95"]
    assert r["n_draws"] >= 20
    assert isinstance(r["结论"], str) and r["结论"]


def test_placebo_is_deterministic():
    """固定随机种子时结果必须可复现——否则文档里引用的数字就没意义。"""
    nav = _make_noisy_nav()
    cal = ev.scheduled_events(nav["date"].iloc[0].date(), nav["date"].iloc[-1].date())
    a = ev.placebo_test(nav, cal, min_importance=3, n_draws=40)
    b = ev.placebo_test(nav, cal, min_importance=3, n_draws=40)
    assert a["p_波动"] == b["p_波动"]
    assert a["随机波动比中位数"] == b["随机波动比中位数"]


def test_placebo_median_is_near_one_for_random_dates():
    """均匀随机抽日期的零假设下，波动比中位数应接近 1。

    构造一段波动率恒定、彼此独立的白噪声收益：此时「哪些日子是事件窗口」
    完全无关紧要，波动比应当围绕 1 分布。若这个测试失败，说明安慰剂检验
    的统计口径本身有偏。
    """
    rng = np.random.default_rng(99)
    n = 2000
    dates = pd.bdate_range("2015-01-01", periods=n)
    # 收益独立同分布 -> 波动率没有聚集性，任何切分都不该产生系统差异
    nav = pd.DataFrame(
        {
            "date": dates,
            "nav": 1.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n)),
        }
    )
    cal = ev.scheduled_events(dates[0].date(), dates[-1].date())
    r = ev.placebo_test(
        nav, cal, before=0, after=1, horizon=1, min_importance=2, n_draws=100, mode="uniform"
    )
    assert r
    assert r["随机波动比中位数"] == pytest.approx(1.0, abs=0.15)
    assert 0.0 <= r["p_波动"] <= 1.0


# ================================================================= 最终判断


def test_verdict_text_without_placebo_defers(impact_frame):
    txt = ev.verdict_text(impact_frame, None)
    assert "安慰剂" in txt
    assert "还不能下判断" in txt


def test_verdict_text_middle_percentile_says_no_effect(impact_frame):
    """落在随机分布中部时，结论必须明确说「与随机日期没有区别」。"""
    txt = ev.verdict_text(
        impact_frame,
        {"模式": "uniform", "p_波动": 0.42, "n_draws": 200, "随机波动比中位数": 0.95},
    )
    assert "没有区别" in txt
    assert "不能提高方向判断" in txt


def test_verdict_text_tail_percentile_cautions_but_allows(impact_frame):
    txt = ev.verdict_text(
        impact_frame,
        {"模式": "shift", "p_波动": 0.01, "n_draws": 200, "随机波动比中位数": 1.00},
    )
    assert "尾部" in txt
    assert "相关" in txt and "因果" in txt


def test_placebo_summary_text_handles_empty():
    assert "样本不足" in ev.placebo_summary_text({})


def test_placebo_summary_text_mentions_mode():
    txt = ev.placebo_summary_text(
        {
            "模式": "uniform",
            "p_波动": 0.4,
            "n_draws": 200,
            "随机波动比中位数": 0.95,
            "随机波动比 P5": 0.8,
            "随机波动比 P95": 1.2,
            "真实波动比": 0.9,
            "结论": "X",
        }
    )
    assert "均匀随机" in txt


# ================================================================= 断网降级


def test_fetch_news_offline_returns_empty_with_columns(monkeypatch):
    """akshare 不可用时返回空表，不抛异常（界面不能因此崩掉）。"""
    monkeypatch.setitem(__import__("sys").modules, "akshare", None)
    df = ev.fetch_news(limit=10)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    for col in ("title", "date", "time", "source", "link"):
        assert col in df.columns


def test_fetch_econ_calendar_offline_returns_empty_with_columns(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "akshare", None)
    df = ev.fetch_econ_calendar("2025-09-01")
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "重要性" in df.columns
