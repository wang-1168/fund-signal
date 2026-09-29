# -*- coding: utf-8 -*-
"""把统计信号翻译成「能照着做的结论」。

本项目对外始终不承诺收益，但「不承诺收益」不等于「只丢一个概率就完事」——
只给出 0.43 却不回答「所以呢」，等于把全部决策成本原样退还给使用者。
一个诚实的工具仍然应该给出意见，只是这个意见必须**可执行、带条件、且能被证伪**。

三条铁律：

1. **先看模型有没有边际。** 若样本外准确率没有超过「永远猜多数类」的基线，
   那么无论概率多好看，第一条建议就是「不要依据本信号操作」。
   这不是保守，这是这个项目用实测换来的结论。
2. **建议必须带条件。** 没有「触发什么就改主意」的建议不是建议，是口号。
   所以每条建议都要附上再评估条件。
3. **不预测涨跌幅。** 模型没有这个能力（README 已用 3 基金 × 4 周期 × 2 模型实测证明），
   因此建议只能落在「动作 / 仓位 / 再评估条件」上，绝不出现「预计上涨 X%」。
   凡是给出涨幅预测的基金工具，要么在骗人，要么在骗自己。

本模块只做机械映射，不接入任何主观判断，因此它的输出可以被完整复现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # 只为类型标注，避免把 akshare 这类重依赖拉进来
    from .pipeline import AnalysisResult

__all__ = ["Advice", "build_advice", "position_fraction", "LEVELS"]

_TAG_RE = re.compile(r"<[^>]+>")

# 边际门槛：样本外准确率相对多数类基线的超额。低于它视为「没有可检验的预测能力」。
EDGE_FLOOR = 0.02

# 各档位的短标签
LEVELS: dict[str, str] = {
    "no_signal": "暂无信号",
    "no_edge": "不建议操作",
    "watch": "观察等待",
    "probe": "小仓试探",
    "build": "分批建仓",
    "cut": "按规则减仓",
}


def position_fraction(prob: float, lo: float, hi: float, cap: float) -> float:
    """分段线性仓位映射：``p<=lo`` 空仓，``p>=hi`` 到上限，中间线性过渡。

    与界面「风控与仓位」页用的是同一套映射，保证两处数字一致。
    """
    if prob is None or np.isnan(prob):
        return 0.0
    p = float(prob)
    if p >= hi:
        return float(cap)
    if p <= lo:
        return 0.0
    return float(min(max(cap * (p - lo) / max(hi - lo, 1e-9), 0.0), 1.0))


@dataclass
class Advice:
    """一条可执行的结论。所有字段都是机械推导出来的，没有主观修饰。"""

    level: str
    stance: str
    headline: str
    because: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    risk: str = ""
    prob: float = float("nan")
    position: float = 0.0
    capital: float = 0.0
    holding: bool = False

    @property
    def amount(self) -> float:
        """建议投入金额 = 本金 × 建议仓位。"""
        return float(self.capital * self.position)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "stance": self.stance,
            "headline": self.headline,
            "because": list(self.because),
            "actions": list(self.actions),
            "triggers": list(self.triggers),
            "risk": self.risk,
            "prob": self.prob,
            "position": self.position,
            "amount": self.amount,
            "capital": self.capital,
            "holding": self.holding,
        }

    def text(self) -> str:
        """纯文本版，给命令行用。"""
        lines = [f"【操作建议】{self.stance}", f"  {self.headline}"]
        if self.because:
            lines.append("  依据：")
            lines += [f"    · {_strip_html(x)}" for x in self.because]
        if self.actions:
            lines.append("  怎么做：")
            lines += [f"    · {_strip_html(x)}" for x in self.actions]
        if self.triggers:
            lines.append("  什么情况下回来重看：")
            lines += [f"    · {_strip_html(x)}" for x in self.triggers]
        if self.risk:
            lines.append(f"  最需要防的：{_strip_html(self.risk)}")
        return "\n".join(lines)


def _pct(v: float, digits: int = 1) -> str:
    return "—" if v is None or np.isnan(v) else f"{v * 100:.{digits}f}%"


def _money(v: float) -> str:
    return f"{v:,.0f} 元"


def _strip_html(s: str) -> str:
    """命令行输出不认 HTML，把标签全部去掉。

    这里刻意用正则而不是逐个 ``replace`` —— 页面用的标签不止 ``<b>``，
    漏一个就会在终端里露出半截标签（``<u>`` 就是这么被漏掉的）。
    """
    return _TAG_RE.sub("", s)


def _selective_stats(result: AnalysisResult) -> dict:
    """复现「只下注模型最自信的前 N%」的实测准确率与超额。

    这是全项目唯一有实测支撑的「提高准确率」手段，因此它必须出现在建议里 ——
    即便结论是「不建议操作」，也要说清楚「如果非要用，唯一有依据的用法是什么」。
    """
    try:
        frame = result.dataset.frame
        pred = result.wf.predictions
        y = frame.loc[pred.index, "label"].astype(int)
        conf = (pred - 0.5).abs()
        out: dict = {}
        for q in (0.2, 0.05):
            mask = conf >= conf.quantile(1 - q)
            hit = ((pred[mask] >= 0.5).astype(int) == y[mask]).mean()
            base = max(float(y[mask].mean()), 1.0 - float(y[mask].mean()))
            out[q] = {
                "n": int(mask.sum()),
                "acc": float(hit),
                "base": base,
                "edge": float(hit) - base,
            }
        return out
    except Exception:
        return {}


def _annualized(total_return: float, years: float) -> float:
    """把累计收益折算成年化，用来回答「一直拿着会怎样」。"""
    if years <= 0 or total_return is None or np.isnan(total_return) or total_return <= -1:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def build_advice(
    result: AnalysisResult,
    *,
    capital: float = 10000.0,
    lo: float = 0.52,
    hi: float = 0.62,
    cap: float = 0.6,
    holding: bool = False,
) -> Advice:
    """把一次分析结果压成一条建议。

    Parameters
    ----------
    result:
        :func:`~fund_signal.pipeline.run_analysis` 的返回值。
    capital:
        可投入本金（元），只用于把比例换算成金额。
    lo, hi, cap:
        仓位映射参数，与界面「风控与仓位」页保持一致。
    holding:
        当前是否已经持有。只有为 True 时才会给出「减仓」这类建议 ——
        没有仓位就没有减仓可言。
    """
    m = result.summary_metrics()
    prob = float(m.get("latest_prob", float("nan")))
    auc = m.get("auc")
    acc_edge = m.get("acc_edge")
    base = m.get("majority_acc")
    acc = m.get("accuracy")
    horizon = result.config.horizon
    cost_bps = float(result.config.cost_bps)
    bt = result.backtest
    strat_dd = bt.strategy_metrics.get("max_drawdown")
    hold_dd = bt.benchmark_metrics.get("max_drawdown")
    strat_ret = bt.strategy_metrics.get("total_return")
    bench_ret = bt.benchmark_metrics.get("total_return")

    def _f(v, default=float("nan")):
        return default if v is None else float(v)

    pos = position_fraction(prob, lo, hi, cap)
    amt = capital * pos

    # 日频波动，用来衡量「成本到底有多贵」
    try:
        daily_vol = float(result.dataset.frame["ret_1"].std())
    except Exception:
        daily_vol = float("nan")
    round_trip = 2.0 * cost_bps / 10000.0
    cost_ratio = (
        round_trip / daily_vol if daily_vol == daily_vol and daily_vol > 0 else float("nan")
    )
    cost_line = (
        f"单边成本 {cost_bps:.0f} 基点，来回就是 {_pct(round_trip, 2)}；"
        f"该基金日波动约 {_pct(daily_vol, 2)}，一次进出相当于吃掉 "
        f"{cost_ratio:.2f} 个标准差的波动"
        if cost_ratio == cost_ratio
        else f"单边成本 {cost_bps:.0f} 基点，来回 {_pct(round_trip, 2)}"
    )

    triggers_common = [
        f"净值更新到新的交易日之后重新跑一次（信号只对特征日 "
        f"{result.latest_date.date() if result.latest_date is not None else '—'} 有效）",
        f"概率跌破空仓线 {lo:.2f}，视为信号失效",
        f"连续 {max(3, horizon * 3)} 个信号方向与实现方向相反，说明模型可能已经失效，"
        "应回到「🧪 模型评估」页看最近几折的 AUC",
    ]

    # 样本区间长度与「一直拿着」的年化 —— 用来给「不作为」标一个机会成本
    try:
        nav_dates = result.nav["date"]
        years = float((nav_dates.iloc[-1] - nav_dates.iloc[0]).days) / 365.25
    except Exception:
        years = float("nan")
    bench_ann = _annualized(_f(bench_ret), years)

    # 唯一有实测支撑的「提高准确率」手段：只下注最自信的前 N%
    selective = _selective_stats(result)
    top20, top5 = selective.get(0.2), selective.get(0.05)

    # ---------------- ① 没有信号 ----------------
    if prob != prob:
        return Advice(
            level="no_signal",
            stance=LEVELS["no_signal"],
            headline="拿不到当期信号，先别做任何动作。",
            because=["最新特征行构造失败（通常是当日净值尚未公布或数据缺口）。"],
            actions=["点侧边栏的「🔄 刷新数据」重试；仍失败就换 `local` 数据源确认真是数据问题。"],
            triggers=["净值公布后重新分析。"],
            risk="在拿不到信号的时候凭感觉操作 —— 那才是真正的赌博。",
            prob=prob,
            position=0.0,
            capital=capital,
            holding=holding,
        )

    edge_ok = bool(acc_edge is not None and not np.isnan(acc_edge) and acc_edge > EDGE_FLOOR)
    conf = abs(prob - 0.5)
    try:
        hist_conf = (result.wf.predictions - 0.5).abs()
        conf_pct = float((hist_conf < conf).mean() * 100)
    except Exception:
        conf_pct = float("nan")

    # ---------------- ② 有信号但模型没有边际：这一档最该说清楚 ----------------
    # 结论是「不建议操作」，但必须给出一个**具体**的意见：不动值多少、想动的话
    # 唯一有实测支撑的用法是什么、对应多少钱。只丢一句「别买」不是意见。
    if not edge_ok:
        because = [
            f"样本外准确率 {_pct(_f(acc), 1)}，多数类基线 {_pct(_f(base), 1)}，"
            f"超额 {_f(acc_edge) * 100:+.1f} 个百分点 —— 没越过 +{EDGE_FLOOR * 100:.0f}pp 的门槛，"
            "即模型的方向判断与「永远猜多数类」在统计上分不出差别",
            f"AUC {_f(auc):.4f}（0.5 = 抛硬币）",
            f"同一段样本上，策略累计 {_pct(_f(strat_ret), 1)} vs 买入持有 "
            f"{_pct(_f(bench_ret), 1)} —— 策略没有领先",
            f"当前概率 {_pct(prob, 1)} 看着"
            f"{'像偏多' if prob >= 0.5 else '像偏空'}，但一个没有边际的模型，"
            "它的概率不该被当成方向依据",
        ]
        if conf_pct == conf_pct and conf_pct >= 90:
            because.append(
                f"注意：这个概率的「自信度」排在历史前 {100 - conf_pct:.0f}%，"
                "而本项目的实测是<b>越自信不等于越准</b>（前 5% 置信度的超额反而为负）"
            )

        actions = [
            f"<b>默认动作：不操作。</b>把决定权交回给自己或别的依据 —— "
            f"不要因为这里显示 {_pct(prob, 0)} 就下单",
            "不操作的代价是可以算的：这只基金的历史年化是 "
            + (
                f"{_pct(bench_ann, 1)}（累计 {_pct(_f(bench_ret), 1)}，共 {years:.0f} 年）"
                if bench_ann == bench_ann
                else "—"
            )
            + "。如果你本来就不打算做择时，那「一直拿着」本身就是一个合法的方案，"
            "而不是「什么都没做」",
        ]
        # 想动手的话，唯一有实测支撑的窄用法
        if top20 and top5:
            probe_amt = capital * cap * 0.5
            trend = (
                "所以「更自信的一档看着更好」这件事本身<u>不可信</u>：同一段数据换个验证折数，"
                "这两个数都会变，前 5% 只有这么几个样本日，这种子集超额在如此小的样本上极不稳定。"
                if top5["edge"] >= top20["edge"]
                else "而且更自信的一档读数反而更差 —— 这正是本项目「越自信不等于越准」结论的来源。"
            )
            actions.append(
                "<b>如果非要动手</b>，全项目唯一有实测支撑的用法是「只下注模型最自信的少数几笔」，"
                f"而且必须把仓位压住。实测：最自信的前 20%（{top20['n']} 个样本日）准确率 "
                f"{_pct(top20['acc'], 1)}、超额 {top20['edge'] * 100:+.1f}pp；前 5%"
                f"（{top5['n']} 个样本日）准确率 {_pct(top5['acc'], 1)}、"
                f"超额 {top5['edge'] * 100:+.1f}pp。" + trend
            )
            actions.append(
                f"因此唯一可执行的形式是：<b>仓位不超过 {_money(probe_amt)}</b>"
                f"（本金 {_money(capital)} 的 {_pct(cap * 0.5, 0)}），"
                "且只在概率进入历史最自信区间时才动用；其余时间保持空仓。"
                "这不是一条推荐，而是「在所有用法里唯一没有被实测否掉的那一种」"
            )
            actions.append(
                "更省事、也往往更划算的做法：换一只基金、换预测周期，或打开"
                "「样本外概率校准 / 净化间隔」再跑一次 —— 有些组合是真的能过 +2pp 门槛的，"
                "那就用那个组合，而不是硬用眼前这个"
            )
        else:
            actions.append(
                "更省事的做法：换一只基金、换预测周期，或打开「样本外概率校准 / 净化间隔」"
                "再跑一次 —— 有些组合是真的能过 +2pp 门槛的"
            )

        return Advice(
            level="no_edge",
            stance=LEVELS["no_edge"],
            headline="本信号在样本外没有可检验的优势，因此不建议依据它买卖。",
            because=because,
            actions=actions,
            triggers=[
                "模型跑出超额 > +2pp（在「🎯 信号详情」页能看到超额），"
                "本建议会立刻改成有条件的操作方案，并给出具体建仓价位与仓位",
                *triggers_common[1:],
            ],
            risk=(
                "最该防的不是踏空，是「因为看到了一个数字就动手」。"
                f"这只基金历史最大回撤 {_pct(hold_dd, 0)} —— "
                "一次满仓的代价远大于错过一次反弹"
            ),
            prob=prob,
            position=0.0,
            capital=capital,
            holding=holding,
        )

    # ---------------- ③ 有边际：进入有条件的操作方案 ----------------
    because = [
        f"样本外准确率 {_pct(_f(acc), 1)} 超过多数类基线 {_pct(_f(base), 1)} "
        f"{_f(acc_edge) * 100:+.1f} 个百分点，越过了 +{EDGE_FLOOR * 100:.0f}pp 门槛",
        f"AUC {_f(auc):.4f}，最新上涨概率 {_pct(prob, 1)}（已做样本外校准）",
        f"按「概率 → 仓位」映射（空仓 <{lo:.2f} / 上限 >{hi:.2f}，仓位上限 {cap:.0%}），"
        f"当前对应仓位 {_pct(pos, 1)}",
        cost_line,
    ]
    if conf_pct == conf_pct and conf_pct >= 90:
        because.append(
            f"提醒：本概率的自信息度排在历史前 {100 - conf_pct:.0f}%，"
            "而实测表明「更自信」并不等于「更准」，所以不要因为这一条就放大仓位"
        )
    if top20 and top5:
        because.append(
            f"仓位不要压在最自信的那一小撮上：最自信的前 5%（{top5['n']} 个样本日）超额 "
            f"{top5['edge'] * 100:+.1f}pp，前 20%（{top20['n']} 个样本日）是 "
            f"{top20['edge'] * 100:+.1f}pp —— 子集越窄，读数越不稳，"
            "所以仓位上限比「模型说了多少」更值得信赖"
        )

    if prob >= hi:
        level, stance = "build", LEVELS["build"]
        headline = f"信号够强，可以按规则建到仓位上限（{_pct(cap, 0)}）。"
        actions = [
            f"第一笔只做一半：约 {_money(amt / 2)}（仓位 {_pct(pos / 2, 1)}）",
            f"第二笔在净值更新且概率仍在 {hi:.2f} 以上时补到 {_money(amt)}",
            f"单只基金仓位不要再往上加 —— 上限 {_pct(cap, 0)}，剩余本金留给别的标的或现金",
            f"给这笔仓位一个止损纪律：回撤超过 {abs(_f(strat_dd)) * 100 * 0.5:.0f}% 就减半，"
            "不要用「等它涨回来」替代纪律",
        ]
    elif prob > lo:
        level, stance = "probe", LEVELS["probe"]
        headline = f"信号刚过阈值，只值得小仓试探（约 {_pct(pos, 1)}）。"
        actions = [
            f"试探仓：{_money(amt)}（本金 {_money(capital)} 的 {_pct(pos, 1)}）",
            f"不要在同一个价位一次打满：概率到 {hi:.2f} 再考虑加到 {_money(capital * cap)}",
            "把这笔钱当成「验证模型是否还在工作」，而不是「这次要赚多少」",
        ]
    elif holding:
        level, stance = "cut", LEVELS["cut"]
        headline = f"概率已跌破空仓线 {lo:.2f}，按规则应该减仓，而不是继续等。"
        actions = [
            "先减到一半，剩下的看下一个信号",
            f"若净值更新后概率仍低于 {lo:.2f}，清掉剩余仓位",
            "不因为「已经亏了」而拖延 —— 这条规则的存在就是为了替代这种纠结",
        ]
    else:
        level, stance = "watch", LEVELS["watch"]
        headline = f"模型有边际，但概率还没到入手价位（需要 > {lo:.2f}），现在等待。"
        actions = [
            "维持空仓，把钱留在手上",
            f"设一个提醒：概率 ≥ {lo:.2f} 时回来按规则建仓，第一笔约 "
            f"{_money(capital * position_fraction(lo + 0.05, lo, hi, cap))}",
            "等待本身是一种操作，不需要每天都动手",
        ]

    triggers = list(triggers_common)
    if level == "cut":
        triggers.insert(0, f"概率回到 {hi:.2f} 以上，则本建议转为建仓")

    return Advice(
        level=level,
        stance=stance,
        headline=headline,
        because=because,
        actions=actions,
        triggers=triggers,
        risk=(
            f"这个模型给的是 {horizon} 个交易日的方向倾向，不是价格预测。"
            f"该基金历史最大回撤 {_pct(hold_dd, 0)}，"
            f"按当前 {_pct(pos, 1)} 仓位算，最坏情况在一段行情里大约亏 "
            f"{_money(abs(amt * _f(hold_dd, 0.0)))}"
            if amt > 0
            else "当前建议仓位为 0，没有新增风险敞口"
        ),
        prob=prob,
        position=pos,
        capital=capital,
        holding=holding,
    )
