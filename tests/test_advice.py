# -*- coding: utf-8 -*-
"""「操作建议」引擎的回归测试。

这一页是本项目唯一会**直接给结论**的地方，因此它比别处更需要被锁住。
测试守住四类约束：

1. **档位判定的优先级**：模型没有实测边际时，无论概率多好看都必须落在
   ``no_edge``，且仓位为 0 —— 这是整个项目最核心的主张，不能被 UI 改坏；
2. **建议必须是可执行的**：每一档都要同时具备「怎么做」与「什么情况下改主意」，
   缺一个就不算建议（只有动作没有退出条件的建议是有害的）；
3. **金额与仓位自洽**：``amount == capital * position``，且本金不参与档位判定；
4. **不给涨跌幅预测**：所有档位的输出里都不允许出现「预计…涨/跌 …%」这类字样。

引擎本身是纯函数，所以这些测试不联网、不训练，跑得很快。
"""

from __future__ import annotations

import copy

import pytest

from fund_signal.advice import EDGE_FLOOR, LEVELS, Advice, build_advice, position_fraction
from fund_signal.config import Config
from fund_signal.pipeline import run_analysis

ACTIONABLE = {"probe", "build", "cut", "watch"}


@pytest.fixture(scope="module")
def result():
    """一次真实的离线分析结果（其它测试在此基础上改指标来触发各档位）。"""
    cfg = Config(fund_code="000001", horizon=1, n_splits=3, model_type="logistic", realtime=False)
    return run_analysis(cfg, data_source="local")


class _WFStub:
    """只提供 ``build_advice`` 需要的那几个字段。

    为什么不用 ``deepcopy`` + 改字典：``WalkForwardResult.overall_metrics`` 的底层
    对象可能被多个实例共享，就地改它会把 module 级 fixture 一起污染，让测试变成
    「顺序相关」。换成独立 stub 之后，每个用例都拿到干净的输入。
    """

    def __init__(self, predictions, metrics: dict):
        self.predictions = predictions
        self.overall_metrics = metrics
        self.auc_summary = "stub"


def _stub_copy(result):
    """浅拷贝结果 + 独立的 wf stub —— 数据集与回测只读，不需要复制。"""
    r = copy.copy(result)
    r.wf = _WFStub(result.wf.predictions, dict(result.wf.overall_metrics))
    return r


@pytest.fixture
def patched(result):
    return _stub_copy(result)


def _force(r, *, edge: float, prob: float, acc: float = 0.55, base: float = 0.50):
    """把边际与最新概率改成指定值，其余（回测、成本、样本）保持真实。"""
    r.wf.overall_metrics.update({"acc_edge": edge, "accuracy": acc, "majority_acc": base})
    r.latest_prob = prob
    return r


# ------------------------------------------------------------------ 仓位映射
def test_position_fraction_bounds():
    assert position_fraction(0.40, 0.52, 0.62, 0.6) == 0.0
    assert position_fraction(0.52, 0.52, 0.62, 0.6) == 0.0
    assert position_fraction(0.62, 0.52, 0.62, 0.6) == pytest.approx(0.6)
    assert position_fraction(0.99, 0.52, 0.62, 0.6) == pytest.approx(0.6)
    assert position_fraction(float("nan"), 0.52, 0.62, 0.6) == 0.0


def test_position_fraction_is_linear_in_the_middle():
    mid = position_fraction(0.57, 0.52, 0.62, 0.6)
    assert mid == pytest.approx(0.3)  # 正好在中点 -> 仓位上限的一半


def test_position_fraction_respects_cap():
    assert position_fraction(0.80, 0.52, 0.62, 0.25) == pytest.approx(0.25)


# ------------------------------------------------------------------ ① 没有边际
def test_no_edge_forces_zero_position(patched):
    """最核心的一条：没有实测边际就不许给仓位，概率再高也一样。"""
    a = build_advice(_force(patched, edge=-0.01, prob=0.85), capital=6000.0)
    assert a.level == "no_edge"
    assert a.stance == LEVELS["no_edge"]
    assert a.position == 0.0
    assert a.amount == 0.0


def test_no_edge_still_gives_actionable_numbers(patched):
    """「不建议买」也必须是具体意见：要有钱数、要有可选的窄用法。"""
    a = build_advice(_force(patched, edge=-0.01, prob=0.43), capital=6000.0)
    joined = " ".join(a.actions)
    assert "6,000 元" in joined or "3,000 元" in joined or "1,800 元" in joined
    assert "2,000" not in joined  # 不该出现凭空的数字
    assert any("前 20%" in x for x in a.actions), "必须给出唯一有实测支撑的窄用法"


def test_edge_floor_boundary(patched):
    """刚好等于门槛不算过 —— 门槛是严格大于。"""
    assert build_advice(_force(patched, edge=EDGE_FLOOR, prob=0.70)).level == "no_edge"
    assert build_advice(_force(patched, edge=EDGE_FLOOR + 1e-6, prob=0.70)).level == "build"


# ------------------------------------------------------------------ ②③④ 有边际
def test_watch_when_prob_below_entry(patched):
    a = build_advice(_force(patched, edge=0.05, prob=0.45), capital=10000.0)
    assert a.level == "watch"
    assert a.position == 0.0
    assert a.amount == 0.0


def test_probe_when_prob_just_above_entry(patched):
    a = build_advice(_force(patched, edge=0.05, prob=0.56), capital=10000.0)
    assert a.level == "probe"
    assert a.position == pytest.approx(position_fraction(0.56, 0.52, 0.62, 0.6))
    assert 0.0 < a.position < 0.6


def test_build_reaches_position_cap(patched):
    a = build_advice(_force(patched, edge=0.05, prob=0.70), capital=10000.0)
    assert a.level == "build"
    assert a.position == pytest.approx(0.6)
    assert a.amount == pytest.approx(6000.0)


def test_cut_requires_holding(patched):
    """没仓位就没有减仓可言 —— 这一点必须靠 holding 开关区分。"""
    flat = build_advice(_force(_stub_copy(patched), edge=0.05, prob=0.40), holding=False)
    held = build_advice(_force(_stub_copy(patched), edge=0.05, prob=0.40), holding=True)
    assert flat.level == "watch"
    assert held.level == "cut"


# ------------------------------------------------------------------ 边界情形
def test_no_signal_when_prob_is_nan(patched):
    patched.latest_prob = float("nan")
    a = build_advice(patched)
    assert a.level == "no_signal"
    assert a.position == 0.0


def test_capital_scales_amount_only(patched):
    """本金只影响金额，绝不影响档位与仓位。"""
    small = build_advice(_force(_stub_copy(patched), edge=0.05, prob=0.56), capital=1000.0)
    big = build_advice(_force(_stub_copy(patched), edge=0.05, prob=0.56), capital=100000.0)
    assert small.level == big.level
    assert small.position == pytest.approx(big.position)
    assert big.amount == pytest.approx(small.amount * 100)


def test_amount_is_capital_times_position(patched):
    a = build_advice(_force(patched, edge=0.05, prob=0.58), capital=7777.0)
    assert a.amount == pytest.approx(a.capital * a.position)


# ------------------------------------------------------------------ 建议的完整性
def _all_branches(result) -> list[Advice]:
    out = []
    for edge, prob, holding in (
        (-0.01, 0.43, False),
        (0.05, 0.45, False),
        (0.05, 0.45, True),
        (0.05, 0.56, False),
        (0.05, 0.70, False),
    ):
        r = _force(_stub_copy(result), edge=edge, prob=prob)
        out.append(build_advice(r, capital=6000.0, holding=holding))
    nan_r = _stub_copy(result)
    nan_r.latest_prob = float("nan")
    out.append(build_advice(nan_r, capital=6000.0))
    return out


def test_every_branch_is_actionable(result):
    """每一档都必须同时给出「怎么做」和「什么时候改主意」。"""
    for a in _all_branches(result):
        assert a.level in LEVELS, f"未知档位 {a.level}"
        assert a.headline and a.headline.endswith(("。", "！")), a.headline
        assert a.actions, f"{a.level} 缺少「怎么做」"
        assert a.triggers, f"{a.level} 缺少「什么情况下回来重看」"
        assert a.risk, f"{a.level} 缺少风险提示"
        assert a.because, f"{a.level} 缺少依据"


def test_no_price_prediction_anywhere(result):
    """本项目明确不预测涨跌幅，输出里不允许出现这类表述。"""
    banned = ("预计上涨", "预计下跌", "预计涨", "预计跌", "目标价", "预期收益")
    for a in _all_branches(result):
        blob = a.text() + " ".join(a.because + a.actions + a.triggers) + a.risk
        for word in banned:
            assert word not in blob, f"{a.level} 出现了涨跌幅预测字样：{word}"


def test_text_version_is_plain(result):
    """命令行版必须是纯文本 —— 页面用的 <b> 不能泄漏到终端。"""
    for a in _all_branches(result):
        txt = a.text()
        assert "<b>" not in txt
        assert "</" not in txt
        assert "【操作建议】" in txt


def test_to_dict_roundtrip(result):
    a = build_advice(result, capital=6000.0)
    d = a.to_dict()
    assert d["level"] == a.level
    assert d["amount"] == a.amount
    assert set(d) >= {"level", "stance", "headline", "because", "actions", "triggers", "risk"}


def test_engine_follows_measured_edge(result):
    """集成检查：档位必须真的由「校准后的样本外边际」决定。

    这里刻意**不钉死**某个数值，因为同一只基金换模型 / 换折数，边际就会变
    （README 里 −0.0067 那组是 lightgbm + 5 折 + 校准）。真正要守住的是这条
    因果关系：过门槛才允许给仓位，没过门槛一律 no_edge 且仓位为 0。
    """
    edge = float(result.summary_metrics()["acc_edge"])
    a = build_advice(result, capital=6000.0)
    if edge > EDGE_FLOOR:
        assert a.level in ACTIONABLE, f"edge={edge} 过了门槛，却给出 {a.level}"
    else:
        assert a.level == "no_edge"
        assert a.position == 0.0
        assert "没有可检验的优势" in a.headline


def test_no_edge_never_returns_a_position(result):
    """跨参数横扫：只要边际没到门槛，仓位就必须是 0。"""
    for edge in (-0.05, -0.001, 0.0, 0.0199):
        a = build_advice(_force(_stub_copy(result), edge=edge, prob=0.99), capital=100000.0)
        assert a.level == "no_edge", f"edge={edge} 不该给操作建议"
        assert a.position == 0.0
        assert a.amount == 0.0


def test_selective_stats_are_reported(result):
    """窄用法的实测数字必须来自真实子集，不能是硬编码。"""
    a = build_advice(_force(_stub_copy(result), edge=-0.01, prob=0.43), capital=6000.0)
    assert a.level == "no_edge"
    joined = " ".join(a.actions)
    assert "前 20%" in joined and "前 5%" in joined
    assert "个样本日" in joined
