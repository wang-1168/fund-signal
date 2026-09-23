# -*- coding: utf-8 -*-
"""
fund-signal 最简上手示例（无需网络，直接可跑）
==============================================

这个脚本演示三件事：

1. 怎么用 3 行代码拿到「未来一个交易日的上涨概率」；
2. 怎么读懂输出里的关键指标（AUC / acc_edge / 持仓日胜率）；
3. 怎么确信回测结果没有被「未来函数」污染。

运行方式
--------

    python examples/quickstart.py

    # 换成别的内置样例标的
    python examples/quickstart.py 320007
    python examples/quickstart.py 161725

    # 只用命令行参数也可以做到同样的事：
    # python -m fund_signal --code 000001 --data local --no-realtime

依赖
----

只需要 fund_signal 运行所需的最小依赖（numpy / pandas / scikit-learn / lightgbm）。
本示例使用仓库自带的离线样例数据，**不需要 akshare 联网，也不需要 streamlit**。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 `python examples/quickstart.py` 而不必先 pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fund_signal import Config, list_sample_codes, run_analysis  # noqa: E402

LINE = "=" * 70


def pct(x: float, digits: int = 2) -> str:
    """把小数格式化成百分比；NaN 显示为 —。"""
    try:
        if x != x:  # NaN
            return "—"
    except Exception:
        return "—"
    return f"{x * 100:.{digits}f}%"


def num(x: float, digits: int = 4) -> str:
    try:
        if x != x:
            return "—"
    except Exception:
        return "—"
    return f"{x:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    codes = list_sample_codes()
    code = argv[0] if argv else "000001"

    if code not in codes:
        print(f"内置样例里没有 {code}。可选：{', '.join(codes)}")
        return 2

    # ------------------------------------------------------------------
    # 第 1 步：配置
    # ------------------------------------------------------------------
    # 只改真正需要的参数。默认值已经是刻意调保守的：
    #   - horizon=1        预测下一个交易日
    #   - model_type       lightgbm（深度浅、叶子少，故意压低过拟合空间）
    #   - cost_bps=15      单边 15 个基点成本，只在仓位变化时计提
    #   - realtime=False   离线样例不需要实时接口
    cfg = Config(
        fund_code=code,
        horizon=1,
        model_type="lightgbm",
        cost_bps=15.0,
        n_splits=5,
        realtime=False,
    )

    print(LINE)
    print(f"  fund-signal 快速上手  |  标的 {code}  |  离线样例数据")
    print(LINE)

    # ------------------------------------------------------------------
    # 第 2 步：跑完整流程
    # ------------------------------------------------------------------
    # data_source="local" 读取 examples/sample_data/ 下的 CSV，完全离线。
    # 想用真实最新数据，把它改成 "akshare" 即可（需要联网 + akshare）。
    result = run_analysis(cfg, data_source="local")

    m = result.summary_metrics()
    c = result.wf.overall_metrics

    # ---------------------------------------------------------------- 概况
    nav = result.nav
    print("\n[1] 数据概况")
    print(
        f"    净值区间    {nav['date'].iloc[0].date()} ~ {nav['date'].iloc[-1].date()}"
        f"  （{len(nav)} 个交易日）"
    )
    print(f"    训练样本    {m['n_samples']} 行 × {m['n_features']} 个特征")
    print(f"    上涨占比    {pct(c.get('base_rate'))}   ← 这是「无脑买入」的基准胜率")

    # ---------------------------------------------------------------- 预测质量
    print("\n[2] 样本外预测质量（滚动前向验证，训练集永远早于预测集）")
    print(f"    AUC         {num(c.get('auc'))}   （0.5 = 完全没有预测能力）")
    print(f"    准确率      {pct(c.get('accuracy'))}")
    print(f"    多数类基线  {pct(c.get('majority_acc'))}   ← 永远猜「涨」的准确率")
    print(f"    超额        {pct(c.get('acc_edge'))}")
    print(f"    各折 AUC    {result.wf.auc_summary}")

    # `has_edge` 是项目里最重要的一句话结论：
    # True 才代表模型可能真的比「无脑猜多数类」强一点点。
    print(f"    是否有优势  {'✓ 有一点微弱优势' if result.has_edge else '✗ 没有，与抛硬币无异'}")

    # ---------------------------------------------------------------- 最新信号
    print("\n[3] 最新信号（要用来做决策的那一行）")
    if result.latest_date is not None:
        prob = result.latest_prob
        print(f"    特征日      {result.latest_date.date()}")
        print(f"    上涨概率    {pct(prob, 1)}")
        print(
            f"    解读        未来 {cfg.horizon} 个交易日"
            f"{'累计收涨' if cfg.horizon > 1 else '收涨'}的概率估计为 {pct(prob, 1)}"
        )
    else:
        print("    （历史数据不足，无法生成）")

    # ---------------------------------------------------------------- 回测
    b = result.backtest
    print(f"\n[4] 样本外回测（阈值 {b.threshold}，成本 {b.cost_bps:.0f}bps）")
    print(f"    策略累计收益  {pct(m['strategy_return'])}")
    print(f"    买入并持有    {pct(m['benchmark_return'])}")
    print(f"    超额收益      {pct(m['excess_return'])}")
    print(f"    最大回撤      {pct(m['max_drawdown'])}")
    print(f"    持仓时间占比  {pct(m['exposure'])}")
    print(f"    持仓日胜率    {pct(b.win_rate_active)}   ← 只看真正有仓位的交易日")

    # ---------------------------------------------------------------- 特征
    if len(result.importance):
        print("\n[5] 最重要的 6 个特征（归一化重要性）")
        for name, val in result.importance.head(6).items():
            bar = "█" * max(1, int(val * 100))
            print(f"    {name:<18} {val:6.3f}  {bar}")

    # ---------------------------------------------------------------- 提示
    if result.notes:
        print("\n[6] 过程中的提示")
        for note in result.notes:
            print(f"    · {note}")

    # ------------------------------------------------------------------
    # 第 3 步：为什么可以相信这个回测
    # ------------------------------------------------------------------
    print("\n[7] 这份结果为什么可信（三条硬约束）")
    print("    ① 标签的买入价用的是 nav[T+1]，不是 nav[T]。")
    print("       基金 T 日净值要到当晚 20:00 才公布，你只能按 T+1 净值成交。")
    print("       若错用 nav[T] 买入价，等于白送一天收益，结论会完全反向。")
    print("    ② 回测只吃滚动前向验证的「样本外预测」，不掺入任何样本内预测。")
    print("    ③ 特征严格只用到 T 日及更早；tests/ 里有截断回归测试锁死这一点。")

    print(f"\n{LINE}")
    print("  免责声明：本项目是量化方法论的演示工具，输出的是统计信号，")
    print("  不构成任何投资建议。基金净值短期方向接近随机，请勿据此交易。")
    print(LINE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
