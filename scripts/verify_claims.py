# -*- coding: utf-8 -*-
"""复现并校验 README / docs 中引用的全部实测数字。

为什么需要这个脚本
------------------
本项目把「结果诚实」当卖点，因此文档里的每一个数字都必须能被读者一键复现。
但历史上出现过一次事故：`calibrate`（样本外概率校准）从可选变成默认后，
README 里引用的 AUC / 准确率 / 策略收益**全部失效**，而文档没有同步——
这正是「结果诚实」类项目最致命的问题。

所以这里把文档里所有引用过的数字集中成一份**可执行的声明清单**，每次运行都重新
实算并逐项比对。任何一项漂移都会被显式标出来。

用法::

    python scripts/verify_claims.py           # 完整校验（含 24 组扫描，约 1 分钟）
    python scripts/verify_claims.py --quick   # 跳过 24 组扫描（约 10 秒）

全部计算基于仓库自带的离线样例数据（``--data local``），**不需要联网**，
因此任何人、任何机器上的结果都应当一致（仅受 LightGBM 版本微小浮点差异影响）。

退出码：存在漂移时为 1，否则为 0（可直接接入 CI）。
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from fund_signal import audit as ad  # noqa: E402
from fund_signal import data as dt  # noqa: E402
from fund_signal import features as ft  # noqa: E402
from fund_signal import model as md  # noqa: E402
from fund_signal.config import Config  # noqa: E402
from fund_signal.pipeline import run_analysis  # noqa: E402

logging.disable(logging.CRITICAL)

# 文档声明的容差：LightGBM 小版本之间会有约 0.01 的浮点差异，
# 因此只对「量级与方向」做断言，不对末位数字做断言。
TOL = 0.02


class Report:
    """收集「声明 vs 实算」的对照结果。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, bool]] = []
        self.drift = 0

    def check(
        self,
        section: str,
        claim: str,
        expected: float | str,
        actual: float | str,
        tol: float = TOL,
    ) -> None:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            ok = abs(float(actual) - float(expected)) <= tol
            exp_s, act_s = f"{expected:.4f}", f"{actual:.4f}"
        else:
            ok = str(expected) == str(actual)
            exp_s, act_s = str(expected), str(actual)
        if not ok:
            self.drift += 1
        self.rows.append((section, claim, exp_s, act_s, ok))

    def table(self, section: str) -> pd.DataFrame:
        rows = [r for r in self.rows if r[0] == section]
        return pd.DataFrame(
            [
                {
                    "文档声明": r[1],
                    "文档值": r[2],
                    "本次实算": r[3],
                    "一致": "✅" if r[4] else "⚠️ 漂移",
                }
                for r in rows
            ]
        )


def _analyze(code: str, horizon: int = 1, model: str = "lightgbm"):
    cfg = Config(
        fund_code=code,
        horizon=horizon,
        model_type=model,
        n_splits=5,
        calibrate=True,
        embargo=True,
        realtime=False,
        use_cache=False,
    )
    with redirect_stdout(io.StringIO()):
        return run_analysis(cfg, data_source="local")


def _dataset(code: str = "000001", horizon: int = 1):
    nav = dt.load_sample_nav(code)
    idx = dt.load_sample_index("sh000300")
    return ft.build_dataset(nav, idx, horizon=horizon)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--quick", action="store_true", help="跳过 24 组（3 基金 × 4 周期 × 2 模型）扫描"
    )
    args = ap.parse_args()

    rep = Report()

    # ---------------------------------------------------------- 1. 默认输出
    print("=" * 78)
    print("① 默认输出（README「先说清楚」章节）—— 000001，离线样例")
    print("=" * 78)
    print("   复现命令: python -m fund_signal --code 000001 --data local --no-realtime --no-cache")
    r = _analyze("000001")
    m = r.summary_metrics()
    bt = r.backtest
    for claim, exp, act in [
        ("n_samples", 5981, m["n_samples"]),
        ("AUC", 0.4994, m["auc"]),
        ("准确率", 0.4952, m["accuracy"]),
        ("多数类基线", 0.5018, m["majority_acc"]),
        ("超额 acc_edge", -0.0067, m["acc_edge"]),
        ("策略累计收益", -0.4970, m["strategy_return"]),
        ("买入持有", 0.2614, m["benchmark_return"]),
        ("持仓日胜率", 0.4794, bt.win_rate_active),
        ("持仓时间占比", 0.3892, bt.exposure),
    ]:
        rep.check("① 默认输出", claim, exp, act)
    print(rep.table("① 默认输出").to_string(index=False))
    print()

    # ---------------------------------------------------------- 2. 三只基金
    print("=" * 78)
    print("② 三只基金对照表（docs/methodology.md 第 7 节）")
    print("=" * 78)
    for code, name, n, auc, acc, maj, wr, sr, br in [
        ("000001", "华夏成长混合", 5981, 0.4994, 0.4952, 0.5018, 0.4794, -0.4970, 0.2614),
        ("320007", "诺安成长混合", 4227, 0.5050, 0.5014, 0.5142, 0.4806, 0.1280, 1.6343),
        ("161725", "招商中证白酒", 2728, 0.5364, 0.5367, 0.5565, 0.4757, -0.3268, -0.6554),
    ]:
        rr = _analyze(code)
        mm = rr.summary_metrics()
        for claim, exp, act in [
            (f"{code} {name} 样本数", n, mm["n_samples"]),
            (f"{code} {name} AUC", auc, mm["auc"]),
            (f"{code} {name} 准确率", acc, mm["accuracy"]),
            (f"{code} {name} 多数类基线", maj, mm["majority_acc"]),
            (f"{code} {name} 持仓日胜率", wr, rr.backtest.win_rate_active),
            (f"{code} {name} 策略收益", sr, mm["strategy_return"]),
            (f"{code} {name} 买入持有", br, mm["benchmark_return"]),
        ]:
            rep.check("② 三只基金", claim, exp, act)
    print(rep.table("② 三只基金").to_string(index=False))
    print()

    ds = _dataset()
    X = ds.frame[ds.feature_names]
    y = ds.frame["label"].astype(int)

    # ---------------------------------------------------------- 3. 24 组扫描
    print("=" * 78)
    print("③ 3 基金 × 4 周期 × 2 模型 = 24 组（README「能力边界」章节）")
    print("=" * 78)
    if args.quick:
        print("   （--quick 已跳过）")
    else:
        aucs, accs, edges = [], [], []
        for code in ("000001", "320007", "161725"):
            for h in (1, 2, 5, 10):
                for mt in ("lightgbm", "logistic"):
                    mm = _analyze(code, horizon=h, model=mt).summary_metrics()
                    aucs.append(mm["auc"])
                    accs.append(mm["accuracy"])
                    edges.append(mm["acc_edge"])
        n_over_055 = sum(1 for a in aucs if a > 0.55)
        n_edge_pos = sum(1 for e in edges if e > 0)
        for claim, exp, act in [
            ("AUC 均值", 0.5105, sum(aucs) / len(aucs)),
            ("AUC 下界", 0.4706, min(aucs)),
            ("AUC 上界", 0.5493, max(aucs)),
            ("准确率 均值", 0.5152, sum(accs) / len(accs)),
            ("准确率 下界", 0.4685, min(accs)),
            ("准确率 上界", 0.5581, max(accs)),
        ]:
            rep.check("③ 24 组", claim, exp, act)
        print(rep.table("③ 24 组").to_string(index=False))
        print(
            f"\n   组数 {len(aucs)} | AUC > 0.55 的组数 {n_over_055}/24 "
            f"| 超额 > 0 的组数 {n_edge_pos}/24"
        )
        print(
            f"   文档声明: 0/24 超过 0.55、4/24 超额为正 -> "
            f"{'✅ 一致' if n_over_055 == 0 and n_edge_pos == 4 else '⚠️ 漂移'}"
        )
        if n_over_055 != 0 or n_edge_pos != 4:
            rep.drift += 1
    print()

    # ---------------------------------------------------------- 4. 过拟合地形
    print("=" * 78)
    print("④ 过拟合 / 欠拟合复杂度地形（README「能力边界」章节）")
    print("=" * 78)
    cp = ad.complexity_path(X, y, n_splits=5, embargo=0, random_state=42)
    show = cp[["配置", "训练准确率", "样本外准确率", "过拟合差距"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tr_lo, tr_hi = cp["训练准确率"].iloc[0], cp["训练准确率"].iloc[-1]
    oos_lo, oos_hi = cp["样本外准确率"].min(), cp["样本外准确率"].max()
    for claim, exp, act in [
        ("训练准确率 起点", 0.557, tr_lo),
        ("训练准确率 终点", 1.000, tr_hi),
        ("样本外 最低", 0.512, oos_lo),
        ("样本外 最高", 0.524, oos_hi),
    ]:
        rep.check("④ 复杂度地形", claim, exp, act)
    print()
    print(rep.table("④ 复杂度地形").to_string(index=False))
    print()

    # ---------------------------------------------------------- 5. 泄漏审计
    print("=" * 78)
    print("⑤ 泄漏审计（docs/methodology.md 7.5.2）")
    print("=" * 78)
    la = ad.leakage_audit(ds.frame, ds.feature_names, horizon=1)
    print(
        la[["变体", "准确率", "AUC", "超额"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    d = dict(zip(la["变体"], la["准确率"], strict=True))
    normal = next(v for k, v in d.items() if k.startswith("①"))
    leaked = next(v for k, v in d.items() if k.startswith("②"))
    badfwd = next(v for k, v in d.items() if k.startswith("③"))
    for claim, exp, act in [
        ("① 正常 准确率", 0.5149, normal),
        ("② 答案入特征 准确率", 1.0000, leaked),
        ("③ 未来函数标签 准确率", 0.5207, badfwd),
    ]:
        rep.check("⑤ 泄漏审计", claim, exp, act)
    print()
    print(rep.table("⑤ 泄漏审计").to_string(index=False))
    print()

    # ---------------------------------------------------------- 6. 校准
    print("=" * 78)
    print("⑥ 概率校准前后 ECE（README「我们真正做了什么」）")
    print("=" * 78)
    wf = md.walk_forward_validate(
        X,
        y,
        model_type="lightgbm",
        n_splits=5,
        calibrate=True,
        embargo=1,
        random_state=42,
        verbose=False,
    )
    yt = y.loc[wf.predictions.index].to_numpy()
    ece_before = ad.calibration_diagnostics(wf.raw_predictions.to_numpy(), yt)["ece"]
    ece_after = ad.calibration_diagnostics(wf.predictions.to_numpy(), yt)["ece"]
    for claim, exp, act in [("ECE 校准前", 0.071, ece_before), ("ECE 校准后", 0.042, ece_after)]:
        rep.check("⑥ 校准", claim, exp, act)
    print(rep.table("⑥ 校准").to_string(index=False))
    print()

    # ---------------------------------------------------------- 7. 选择性预测
    print("=" * 78)
    print("⑦ 选择性预测（README FAQ / docs/methodology.md 7.5.3）")
    print("=" * 78)
    yser = pd.Series(yt, index=wf.predictions.index)
    sel = ad.selective_accuracy_table(wf.predictions, yser)
    print(sel.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    row20 = sel[sel["下注比例"].round(3) == 0.200]
    acc20 = float(row20["方向准确率"].iloc[0])
    edge20 = float(row20["超额"].iloc[0])
    rep.check("⑦ 选择性预测", "覆盖率 20% 准确率", 0.533, acc20)
    rep.check("⑦ 选择性预测", "覆盖率 20% 超额", 0.022, edge20)
    print()
    print(rep.table("⑦ 选择性预测").to_string(index=False))
    print()

    # ---------------------------------------------------------- 8. 理论换算
    print("=" * 78)
    print("⑧ AUC ↔ 准确率 理论换算（纯数学，不依赖数据）")
    print("=" * 78)
    rows = [
        {"AUC": a, "文档值": e, "实算": ad.auc_to_accuracy(a)}
        for a, e in [
            (0.500, 0.500),
            (0.520, 0.514),
            (0.550, 0.535),
            (0.600, 0.571),
            (0.700, 0.645),
            (0.900, 0.818),
            (0.965, 0.900),
        ]
    ]
    for r_ in rows:
        rep.check("⑧ 理论换算", f"AUC {r_['AUC']}", r_["文档值"], r_["实算"], tol=0.001)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    need = ad.accuracy_to_auc(0.90)
    print(f"\n   准确率 90% 需要 AUC = {need:.3f}（文档声明 0.965）")
    if abs(need - 0.965) > 0.001:
        rep.drift += 1
    print()

    # ---------------------------------------------------------- 汇总
    print("=" * 78)
    total = len(rep.rows)
    print(f"汇总：共校验 {total} 项，漂移 {rep.drift} 项")
    if rep.drift:
        print("⚠️  存在漂移，请更新 README / docs 中对应数字：")
        for sec, claim, exp, act, ok in rep.rows:
            if not ok:
                print(f"   - [{sec}] {claim}: 文档 {exp} / 实算 {act}")
    else:
        print("✅ 文档中引用的全部数字与实算一致")
    print("=" * 78)
    return 1 if rep.drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
