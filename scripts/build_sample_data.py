# -*- coding: utf-8 -*-
"""
生成随仓库分发的离线样例数据。

用途
----
1. 使用者网络不通时，可把数据源切成 ``local`` 直接体验完整流程；
2. CI 中的单元测试依赖这些数据，保证测试不依赖外网。

用法::

    python scripts/build_sample_data.py

数据来源：akshare（东方财富公开接口），仅用于功能演示。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在导入 akshare 之前清掉可能失效的本地代理
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)

from fund_signal import data as dt  # noqa: E402
from fund_signal.config import SAMPLE_DIR  # noqa: E402

# 选样标准：成立时间足够长（样本量够做时序验证）+ 类型有代表性
FUNDS: dict[str, str] = {
    "000001": "华夏成长混合（老牌混合型，2001 年成立）",
    "320007": "诺安成长混合（半导体主题，波动大）",
    "161725": "招商中证白酒指数（行业主题指数型）",
}

INDEXES: dict[str, str] = {
    "sh000300": "沪深300",
}


def main() -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0

    for code, desc in FUNDS.items():
        try:
            nav = dt.fetch_fund_nav(code)
            path = SAMPLE_DIR / f"{code}_nav.csv"
            nav.to_csv(path, index=False, encoding="utf-8")
            size_kb = path.stat().st_size / 1024
            print(f"[ok]   {code}  {desc}")
            print(
                f"       {len(nav)} 行 | {nav['date'].iloc[0].date()} ~ "
                f"{nav['date'].iloc[-1].date()} | {size_kb:.0f} KB -> {path.name}"
            )
        except Exception as err:
            failed += 1
            print(f"[fail] {code}  {desc}: {err}")

    for symbol, desc in INDEXES.items():
        try:
            idx = dt.fetch_index(symbol)
            path = SAMPLE_DIR / f"index_{symbol}.csv"
            idx.to_csv(path, index=False, encoding="utf-8")
            print(f"[ok]   {symbol}  {desc}: {len(idx)} 行 -> {path.name}")
        except Exception as err:
            failed += 1
            print(f"[fail] {symbol}  {desc}: {err}")

    print(f"\n完成，失败 {failed} 项。输出目录：{SAMPLE_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
