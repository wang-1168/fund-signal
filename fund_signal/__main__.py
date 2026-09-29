# -*- coding: utf-8 -*-
"""
命令行入口。

用法示例
--------
::

    # 用默认参数分析（沪深300 为基准，预测下一交易日方向）
    python -m fund_signal --code 000001

    # 预测未来 5 个交易日，换用逻辑回归
    python -m fund_signal --code 110022 --horizon 5 --model logistic

    # 离线跑通（用随仓库附带的样例数据，不需要网络）
    python -m fund_signal --code 000001 --data local

    # 调整回测阈值与交易成本
    python -m fund_signal --code 000001 --threshold 0.55 --cost-bps 20
"""

from __future__ import annotations

import argparse
import sys

from .advice import build_advice
from .config import DEFAULT_BENCHMARKS, Config
from .pipeline import run_analysis
from .utils import human_number, log

LINE = "─" * 62


def _force_utf8() -> None:
    """Windows 终端下保证中文正常输出。"""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fund_signal",
        description="公募基金量化信号分析 —— 预测未来 N 个交易日的涨跌方向。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-c", "--code", required=True, help="6 位公募基金代码，例如 000001")
    p.add_argument("-H", "--horizon", type=int, default=1, help="预测未来多少个交易日（默认 1）")
    p.add_argument(
        "-m",
        "--model",
        default="lightgbm",
        choices=["lightgbm", "logistic"],
        help="模型类型（默认 lightgbm）",
    )
    p.add_argument(
        "-b",
        "--benchmark",
        default=DEFAULT_BENCHMARKS["沪深300"],
        help=f"基准指数代码（默认 {DEFAULT_BENCHMARKS['沪深300']} 沪深300）",
    )
    p.add_argument("-t", "--threshold", type=float, default=0.5, help="看多概率阈值（默认 0.5）")
    p.add_argument("-s", "--splits", type=int, default=5, help="滚动前向验证折数（默认 5）")
    p.add_argument(
        "--cost-bps", type=float, default=15.0, help="单边交易成本，单位基点（默认 15，即 0.15%%）"
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    p.add_argument(
        "--data",
        default="akshare",
        choices=["akshare", "local"],
        help="数据来源：akshare 联网 / local 离线样例",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="忽略本地缓存，强制重新取数（等同「刷新数据」）"
    )
    p.add_argument(
        "--no-realtime",
        action="store_true",
        help="关闭实时能力（不补齐最新净值、不抓盘中估值，速度更快）",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="可投入本金（元），仅用于把建议仓位换算成金额（默认 10000）",
    )
    p.add_argument(
        "--holding",
        action="store_true",
        help="当前已持有该基金 —— 只有开启后才会给出「按规则减仓」类建议",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)

    cfg = Config(
        fund_code=args.code,
        horizon=args.horizon,
        benchmark=args.benchmark,
        model_type=args.model,
        threshold=args.threshold,
        n_splits=args.splits,
        cost_bps=args.cost_bps,
        random_state=args.seed,
        use_cache=not args.no_cache,
        realtime=not args.no_realtime,
    )

    try:
        cfg.validate()
    except ValueError as err:
        print(f"参数错误：{err}", file=sys.stderr)
        return 2

    print(LINE)
    print(
        f"  fund-signal  |  基金 {cfg.fund_code}  |  horizon={cfg.horizon}  "
        f"模型={cfg.model_type}  阈值={cfg.threshold}"
    )
    print(LINE)

    try:
        result = run_analysis(cfg, data_source=args.data)
    except ImportError as err:
        print(f"\n依赖缺失：\n{err}", file=sys.stderr)
        return 3
    except Exception as err:
        print(f"\n分析失败：{err}", file=sys.stderr)
        log.debug("详细错误", exc_info=True)
        return 1

    m = result.summary_metrics()
    c = result.wf.overall_metrics

    # ---------------------------------------------------------- 数据概览
    nav = result.nav
    print("\n【基金概况】")
    print(f"  名称        {result.fund_name}")
    print(
        f"  净值区间    {nav['date'].iloc[0].date()} ~ {nav['date'].iloc[-1].date()}"
        f"（{len(nav)} 个交易日）"
    )
    print(f"  样本数      {m['n_samples']} 行 × {m['n_features']} 个特征")
    print(f"  上涨占比    {human_number(c.get('base_rate'), 4)}   ← 即「无脑买入」的基准胜率")

    # ---------------------------------------------------------- 数据新鲜度
    meta = result.realtime.get("meta", {})
    est = result.realtime_estimate
    age = result.data_age_days
    print("\n【数据新鲜度】")
    if result.last_nav_date is not None:
        if age <= 1:
            fresh = "最新交易日"
        elif age <= 7:
            fresh = f"{int(age)} 天前"
        else:
            fresh = f"已滞后 {int(age)} 天，建议加 --no-cache 刷新"
        print(f"  数据截至    {result.last_nav_date.date()}（{fresh}）")
    if meta:
        print(
            f"  申购/赎回   {meta.get('purchase_status', '—')} / "
            f"{meta.get('redeem_status', '—')}"
            f"   费率 {meta.get('fee', '—')}"
        )
    if est is not None:
        g = est.get("est_growth")
        extra = f"  （估算涨跌 {g:+.2%}）" if g is not None else ""
        print(f"  盘中估值    {est['est_nav']:.4f}{extra}   ← 第三方拟合，与最终净值有偏差")
    else:
        print("  盘中估值    —（数据源未覆盖该基金，或当前非交易时段）")

    # ---------------------------------------------------------- 模型表现
    print(f"\n【样本外预测质量】（滚动前向验证，{cfg.n_splits} 折）")
    print(f"  AUC         {human_number(c.get('auc'), 4)}   （0.5 = 无预测能力）")
    print(f"  准确率      {human_number(c.get('accuracy'), 4)}")
    print(f"  多数类基线  {human_number(c.get('majority_acc'), 4)}")
    print(
        f"  超额        {human_number(c.get('acc_edge'), 4)}"
        f"   {'✓ 有微弱优势' if result.has_edge else '✗ 未超过基线'}"
    )
    print(f"  各折 AUC    {result.wf.auc_summary}")

    # ---------------------------------------------------------- 最新信号
    print("\n【最新信号】")
    if result.latest_date is not None:
        p = result.latest_prob
        verdict = "偏多" if p >= cfg.threshold else "偏空/观望"
        print(f"  特征日      {result.latest_date.date()}")
        print(f"  上涨概率    {human_number(p, 4)}   →  {verdict}")
        print(
            f"  含义        预测未来 {cfg.horizon} 个交易日的"
            f"{'累计收涨' if cfg.horizon > 1 else '收涨'}概率"
        )
    else:
        print("  （无法生成，历史数据不足）")

    # ---------------------------------------------------------- 回测
    b = result.backtest
    print(
        f"\n【样本外回测】（阈值 {b.threshold}，成本 {b.cost_bps:.0f}bps，"
        f"额外执行延迟 {b.execution_lag} 日）"
    )
    print(b.summary().to_string(header=True))
    print(f"  持仓时间占比   {human_number(b.exposure, 4)}")
    print(f"  持仓日胜率     {human_number(b.win_rate_active, 4)}   ← 只看有仓位的日子")
    print(f"  超额收益       {human_number(b.excess_return, 4)}")

    # ---------------------------------------------------------- 特征
    if len(result.importance):
        print("\n【最重要的 8 个特征】")
        for name, val in result.importance.head(8).items():
            bar = "█" * max(1, int(val * 120))
            print(f"  {name:<18} {val:6.3f}  {bar}")

    # ---------------------------------------------------------- 提示
    if result.notes:
        print("\n【提示】")
        for n in result.notes:
            print(f"  · {n}")

    # ---------------------------------------------------------- 操作建议
    # 放在最后，是因为它是**结论**：上面所有数字都是它的证据。
    # 只输出一个概率而不回答「所以呢」，等于把决策成本原样退还给使用者。
    print(f"\n{LINE}")
    print(build_advice(result, capital=args.capital, holding=args.holding).text())
    print(LINE)

    print(f"\n{LINE}")
    print("  免责声明：本项目为量化分析工具，输出的是统计信号，不构成任何投资建议。")
    print("  基金净值短期走势接近随机，预测结果不应作为交易依据。")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
