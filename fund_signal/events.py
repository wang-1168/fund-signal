# -*- coding: utf-8 -*-
"""重大事件日历与事件影响分析。

为什么单独有这个模块
--------------------
最自然的直觉是「把新闻和重大事件喂给模型，预测就更准」。本模块的存在是为了把这件事
**做对**，它把「事件」严格分成三层，因为三层的可用性完全不同：

====================  ============================  ================================
层次                  例子                          能不能进模型
====================  ============================  ================================
① 规则化日程事件       LPR 报价日、CPI 公布窗口、    **能**。时间点事先已知，不依赖
（``scheduled_events``）PMI、政治局会议、两会、      未来信息，是唯一合法的事件特征
                      指数调样、季报窗口、季末资金面
② 实时快讯/日历        财联社电报、东财快讯、         **不能**。当前快讯的「重要性」
（``fetch_news``）      百度经济日历（含预期值）      事后才看得清，事后选事件=事后诸葛
③ 历史事件影响         「CPI 公布日前后该基金         用来**检验①②值不值得用**，
（``event_impact``）    表现是否与非事件期不同」       不是特征本身
====================  ============================  ================================

设计原则：**先测，再信。** 任何事件相关的东西在进入模型前，都必须先用
``event_impact`` 证明它在样本外确实有区分度。本项目 v0.1.0 的定位是「结果诚实」，
所以这里不预设事件有用——测出来没用就如实说没用。

一个容易踩的坑
--------------
新闻的时间戳是**发布时刻**，而基金净值的时间戳是**T 日收盘后公布、T+1 日才可成交**。
把「T 日盘中看到的新闻」当成 T 日可交易信息，就是典型的未来函数。本模块所有
按事件日期对齐的分析，都以「事件日 T 的净值已知时，最早只能在 T+1 成交」为前提，
与 ``features.py`` 的标签口径保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger(__name__)

# ================================================================= 事件定义

# 重要性：3 = 能直接改变市场风险偏好；2 = 影响行业/风格；1 = 常规扰动
IMPORTANCE_LABELS = {3: "高", 2: "中", 1: "低"}


@dataclass(frozen=True)
class EventRule:
    """一条规则化的日程事件。

    Attributes
    ----------
    name:
        事件名，如 ``"LPR 报价"``。
    category:
        分类，用于聚合统计：``货币政策`` / ``经济数据`` / ``政策会议`` /
        ``市场制度`` / ``资金面``。
    importance:
        1~3，见 :data:`IMPORTANCE_LABELS`。
    precision:
        ``"exact"`` 表示发布日基本固定（如 LPR 每月 20 日）；
        ``"approx"`` 表示只是一个窗口（如 CPI 在每月 9~10 日，具体日浮动）。
        **回测时必须分开看待**，否则会把窗口噪声算成事件效应。
    note:
        给读者的一句话解释。
    """

    name: str
    category: str
    importance: int
    precision: str
    note: str


# 中国市场的确定性日程。刻意只收录「时间点事先可推」的事件。
CALENDAR_RULES: dict[str, EventRule] = {
    "lpr": EventRule(
        "LPR 报价",
        "货币政策",
        3,
        "exact",
        "每月 20 日公布 1 年期/5 年期以上 LPR，直接决定贷款与房贷利率中枢",
    ),
    "mlf": EventRule(
        "MLF/逆回购操作", "货币政策", 2, "exact", "月中 MLF 续作与公开市场操作，反映央行流动性态度"
    ),
    "pmi": EventRule(
        "制造业 PMI", "经济数据", 2, "exact", "每月最后一日公布，是当月最早的经济景气温标"
    ),
    "cpi": EventRule(
        "CPI / PPI 公布", "经济数据", 2, "approx", "每月 9~10 日公布上月物价，影响通胀与政策预期"
    ),
    "finance": EventRule(
        "金融数据（社融/信贷）",
        "经济数据",
        2,
        "approx",
        "每月 10~15 日公布，市场最关注的宽信用信号",
    ),
    "activity": EventRule(
        "经济数据（工业/消费/投资）", "经济数据", 2, "approx", "每月 15 日前后公布上月实体经济数据"
    ),
    "gdp": EventRule(
        "GDP 公布", "经济数据", 3, "approx", "每季度结束后约 17 日公布，全年最重要的总量数据"
    ),
    "politburo": EventRule(
        "政治局会议",
        "政策会议",
        3,
        "approx",
        "4/7/10/12 月下旬讨论经济工作，定调下一个季度政策方向",
    ),
    "cewc": EventRule(
        "中央经济工作会议", "政策会议", 3, "approx", "12 月中旬定调次年经济政策总基调"
    ),
    "lianghui": EventRule(
        "全国两会", "政策会议", 3, "approx", "3 月 5 日前后，公布增长目标与财政赤字率"
    ),
    "index_review": EventRule(
        "指数样本调整生效", "市场制度", 2, "approx", "6/12 月第二个周五后生效，被动资金集中调仓"
    ),
    "fund_report": EventRule(
        "基金定期报告披露", "市场制度", 1, "approx", "季报/年报披露窗口，机构调仓信息集中释放"
    ),
    "quarter_end": EventRule(
        "季末资金面", "资金面", 1, "exact", "季末银行考核，短端利率常阶段性走高"
    ),
}


def _month_shift(d: date, months: int) -> date:
    """把日期按月平移，自动处理跨年。"""
    m = d.month - 1 + months
    y = d.year + m // 12
    return date(y, m % 12 + 1, 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """某年第 n 个星期 weekday（0=周一）。用于指数调样等规则。"""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return date(year, month, 1 + offset + 7 * (n - 1))


def _snap(d: date, trade_dates: set | None) -> date | None:
    """把自然日吸附到最近的交易日（向后找，最多 10 天）。

    ``trade_dates`` 为 None 时退化为「跳过周末」，精度略低但离线可用。

    一个容易踩的坑：调用方传进来的交易日历往往只覆盖到**最后一个已公布净值日**
    （比如基金净值数据截止昨天）。若直接用它去吸附**未来**的事件，会找不到匹配而
    静默丢弃事件——「未来有什么大事」整块就空了。所以这里先判断日历是否覆盖到
    目标日期之后，不覆盖就退回周末规则，保证未来事件一定能算出来。
    """
    if trade_dates is not None and not trade_dates:
        trade_dates = None
    if trade_dates is not None and max(trade_dates) < d:
        trade_dates = None  # 日历不覆盖未来，退回周末规则

    for i in range(11):
        cand = d + timedelta(days=i)
        if trade_dates is None:
            if cand.weekday() < 5:
                return cand
        elif cand in trade_dates:
            return cand
    return None


def scheduled_events(
    start: date | str,
    end: date | str,
    trade_dates: pd.Series | set | None = None,
) -> pd.DataFrame:
    """生成 ``[start, end]`` 区间内的规则化日程事件。

    与「抓取新闻」不同，这里的事件是**算出来的**——因此：

    - 可以生成**未来**的事件（这是「提前知道有什么大事」的基础）；
    - 可以生成**历史**事件（这是回测事件影响的基础）；
    - 不依赖任何网络请求，CI 里可离线运行。

    Parameters
    ----------
    start, end:
        起止日期，``date`` 或 ``"YYYY-MM-DD"`` 字符串。
    trade_dates:
        可选的真实交易日集合，用于把事件吸附到实际交易日。
        传 ``None`` 则退化为跳过周末。

    Returns
    -------
    pandas.DataFrame
        列：``date`` / ``name`` / ``category`` / ``importance`` /
        ``importance_label`` / ``precision`` / ``note``。
    """
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    if end_d < start_d:
        raise ValueError(f"end ({end_d}) 不能早于 start ({start_d})")

    td: set | None
    if trade_dates is None:
        td = None
    elif isinstance(trade_dates, pd.Series):
        td = {pd.Timestamp(x).date() for x in trade_dates}
    else:
        td = {pd.Timestamp(x).date() for x in trade_dates}

    rows: list[dict] = []

    def add(rule_key: str, when: date, note_extra: str = "") -> None:
        r = CALENDAR_RULES[rule_key]
        d = _snap(when, td)
        if d is None or not (start_d <= d <= end_d):
            return
        rows.append(
            {
                "date": pd.Timestamp(d),
                "name": r.name,
                "category": r.category,
                "importance": r.importance,
                "importance_label": IMPORTANCE_LABELS[r.importance],
                "precision": r.precision,
                "note": r.note + note_extra,
            }
        )

    # 逐月遍历，覆盖跨年区间
    cur = date(start_d.year, start_d.month, 1)
    while cur <= end_d:
        y, m = cur.year, cur.month
        last_day = (date(y + (m // 12), m % 12 + 1, 1) - timedelta(days=1)).day

        add("lpr", date(y, m, 20))
        add("mlf", date(y, m, 15))
        add("pmi", date(y, m, last_day))
        add("cpi", date(y, m, 9), "（窗口，实际发布日会浮动）")
        add("finance", date(y, m, 12), "（窗口，实际发布日会浮动）")
        add("activity", date(y, m, 15), "（窗口，实际发布日会浮动）")

        if m in (1, 4, 7, 10):
            add("gdp", date(y, m, 17), "（窗口，实际发布日会浮动）")
        if m in (4, 7, 10, 12):
            add("politburo", date(y, m, 25))
        if m == 12:
            add("cewc", date(y, 12, 12))
        if m == 3:
            add("lianghui", date(y, 3, 5))

        # 指数调样：6 / 12 月第二个周五，通常次周五收市后生效
        if m in (6, 12):
            add("index_review", _nth_weekday(y, m, 4, 2) + timedelta(days=7))

        # 季报：4 / 7 / 10 / 翌年 1 月的 20 日前后
        if m in (1, 4, 7, 10):
            add("fund_report", date(y, m, 20), "（窗口）")

        # 季末最后 3 个自然日
        for back in range(3):
            add("quarter_end", date(y, m, last_day - back))

        cur = _month_shift(cur, 1)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["date", "importance"], ascending=[True, False]).reset_index(drop=True)


def upcoming_events(
    days: int = 10,
    min_importance: int = 2,
    trade_dates: pd.Series | set | None = None,
    as_of: date | str | None = None,
) -> pd.DataFrame:
    """未来 ``days`` 个自然日内的**重要**事件——「接下来会发生什么」。"""
    today = pd.Timestamp(as_of).date() if as_of is not None else date.today()
    df = scheduled_events(today, today + timedelta(days=days), trade_dates)
    if df.empty:
        return df
    return df[df["importance"] >= min_importance].reset_index(drop=True)


# ================================================================= 实时事件


def fetch_news(limit: int = 30, timeout: int | None = None) -> pd.DataFrame:
    """抓取市场级实时快讯（财联社 + 东财全球快讯）。

    这类数据用于**解释与预警**，不进入模型：快讯的「重要性」只有事后才看得清，
    用当天快讯去挑事件、再回测当时的收益，属于事后诸葛。

    网络失败时返回空表而不抛异常——界面上的事件面板不应该因为数据源抖动就崩掉。
    """
    frames: list[pd.DataFrame] = []
    try:
        import akshare as ak

        cls = ak.stock_info_global_cls(symbol="全部")
        cls = cls.rename(columns={"标题": "title", "发布日期": "date", "发布时间": "time"})
        cls["source"] = "财联社"
        frames.append(cls[["title", "date", "time", "source"]])
    except Exception as err:  # noqa: BLE001
        log.warning("财联社快讯获取失败：%s", err)

    try:
        import akshare as ak

        em = ak.stock_info_global_em()
        em = em.rename(columns={"标题": "title", "发布时间": "time", "链接": "link"})
        em["date"] = pd.to_datetime(em["time"]).dt.strftime("%Y-%m-%d")
        em["source"] = "东财"
        if "link" not in em.columns:
            em["link"] = ""
        frames.append(em[["title", "date", "time", "source", "link"]])
    except Exception as err:  # noqa: BLE001
        log.warning("东财快讯获取失败：%s", err)

    if not frames:
        return pd.DataFrame(columns=["title", "date", "time", "source", "link"])

    df = pd.concat(frames, ignore_index=True)
    for col in ("date", "time", "source", "title"):
        df[col] = df[col].astype(str)
    if "link" not in df.columns:
        df["link"] = ""
    df["link"] = df["link"].fillna("").astype(str)
    # 数据源会夹杂只有时间没有标题的空行，直接丢掉
    df["title"] = df["title"].str.strip()
    df = df[df["title"].ne("") & df["title"].ne("nan") & df["title"].ne("None")]
    return df.head(limit).reset_index(drop=True)


def fetch_econ_calendar(when: date | str | None = None) -> pd.DataFrame:
    """抓取指定日期的**经济日历**（含预期值 / 前值 / 重要性）。

    这是少数带**前瞻信息**（预期值）的公开数据源，因此「接下来要公布的、
    市场预期是多少」可以直接摆给人看。但同样不进入模型：预期值本身随日期变动，
    历史预期值拿不到，无法做无偏回测。
    """
    d = pd.Timestamp(when).date() if when is not None else date.today()
    try:
        import akshare as ak

        df = ak.news_economic_baidu(date=d.strftime("%Y%m%d"))
    except Exception as err:  # noqa: BLE001
        log.warning("经济日历获取失败：%s", err)
        return pd.DataFrame(
            columns=["日期", "时间", "地区", "事件", "公布", "预期", "前值", "重要性"]
        )
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["日期", "时间", "地区", "事件", "公布", "预期", "前值", "重要性"]
        )
    df = df.copy()
    df["重要性"] = pd.to_numeric(df["重要性"], errors="coerce").fillna(1).astype(int)
    return df.reset_index(drop=True)


# ================================================================= 事件影响


def event_impact(
    nav: pd.DataFrame,
    events: pd.DataFrame,
    before: int = 0,
    after: int = 1,
    horizon: int = 1,
    min_importance: int = 1,
) -> pd.DataFrame:
    """检验「事件窗口内该基金的表现是否与非事件期不同」。

    这是**决定事件值不值得用的唯一依据**。做法：

    1. 以事件日 T 为锚，取 ``[T-before, T+after]`` 的交易日窗口；
    2. 窗口内收益按「T 日净值已知 → T+1 才能成交」的同一口径计算
       （与 :mod:`fund_signal.features` 的标签定义一致，避免未来函数）；
    3. 把**事件窗口日**与**非事件日**分组对比：平均收益、上涨概率、
       日波动，并给出两者之差。

    默认 ``before=0, after=1``，即「事件日 + 事件后第一个交易日」。这是刻意收窄的：
    事件类别一多，宽窗口会把绝大多数交易日都圈进来，导致「非事件期」只剩零星样本，
    对照彻底失效（第一版用 ``before=1, after=3`` 时，000001 的事件窗口覆盖了 93% 的样本，
    所谓「胜率差 +4 个百分点」纯属采样偏差）。输出里的 ``覆盖率`` 列就是用来
    暴露这个问题的——**任何覆盖率超过 50% 的事件类别，其结论都不可采信**。

    Parameters
    ----------
    nav:
        含 ``date`` / ``nav`` 两列的净值表。
    events:
        :func:`scheduled_events` 的输出。
    before, after:
        事件窗口向前 / 向后各取多少个交易日。
    horizon:
        与模型预测周期保持一致，用于计算可交易收益。
    min_importance:
        只统计重要性不低于该值的事件。

    Returns
    -------
    pandas.DataFrame
        按事件名聚合：``样本数`` / ``覆盖率`` / ``窗口平均收益`` / ``非事件期平均收益`` /
        ``收益差`` / ``窗口上涨概率`` / ``非事件期上涨概率`` / ``胜率差`` /
        ``窗口日波动`` / ``非事件期日波动`` / ``波动比`` / ``重要性``。

    Notes
    -----
    ``precision="approx"`` 的事件（如 CPI 公布窗口）日期本身有 ±2 天浮动，
    其窗口统计会被窗口噪声稀释，读数时不要把它当成精确的事件效应。
    不同事件类别的窗口可能重叠，因此类别之间的横向比较只能是**定性**的。
    """
    if nav is None or nav.empty or events is None or events.empty:
        return pd.DataFrame()

    ev = events[events["importance"] >= min_importance]
    if ev.empty:
        return pd.DataFrame()

    n = nav.copy()
    n["date"] = pd.to_datetime(n["date"])
    n = n.sort_values("date").reset_index(drop=True)

    # 可交易收益：T+1 买入、T+1+horizon 卖出（与 features 标签同口径）
    n["fwd"] = n["nav"].shift(-(1 + horizon)) / n["nav"].shift(-1) - 1.0
    n["ret1"] = n["nav"].pct_change()
    idx_of = {d: i for i, d in enumerate(n["date"])}

    base = n.dropna(subset=["fwd"])
    base_ret = float(base["fwd"].mean())
    base_win = float((base["fwd"] > 0).mean())
    base_vol = float(base["ret1"].std())

    flagged = np.zeros(len(n), dtype=bool)
    rows: list[dict] = []

    for name, grp in ev.groupby("name", sort=False):
        hit = np.zeros(len(n), dtype=bool)
        for d in pd.to_datetime(grp["date"]):
            i = idx_of.get(d)
            if i is None:
                continue
            lo, hi = max(0, i - before), min(len(n) - 1, i + after)
            hit[lo : hi + 1] = True
        if hit.sum() == 0:
            continue
        flagged |= hit

        sub = n.loc[hit].dropna(subset=["fwd"])
        ctrl = n.loc[~hit].dropna(subset=["fwd"])
        if len(sub) < 20 or len(ctrl) < 20:
            continue

        w_ret, c_ret = float(sub["fwd"].mean()), float(ctrl["fwd"].mean())
        w_win, c_win = float((sub["fwd"] > 0).mean()), float((ctrl["fwd"] > 0).mean())
        w_vol = float(sub["ret1"].std())
        c_vol = float(ctrl["ret1"].std())

        rows.append(
            {
                "事件": name,
                "重要性": int(grp["importance"].iloc[0]),
                "精度": str(grp["precision"].iloc[0]),
                "样本数": int(len(sub)),
                "覆盖率": float(hit.sum() / len(n)),
                "窗口平均收益": w_ret,
                "非事件期平均收益": c_ret,
                "收益差": w_ret - c_ret,
                "窗口上涨概率": w_win,
                "非事件期上涨概率": c_win,
                "胜率差": w_win - c_win,
                "窗口日波动": w_vol,
                "非事件期日波动": c_vol,
                "波动比": w_vol / c_vol if c_vol else float("nan"),
            }
        )

    out = pd.DataFrame(rows)

    # 补一行「全事件窗口合并」，方便一眼对照。
    # 注意：这一行**不能**依赖"某个事件类别的样本数达标"——它是对全部事件窗口的整体
    # 描述。否则一旦类别多、单个类别样本少（比如只有一两个事件的短区间），
    # 整张表会因为所有类别都被过滤而变成空的，「事件到底有没有用」就彻底答不出来。
    if not flagged.any():
        return out
    merged = n.loc[flagged].dropna(subset=["fwd"])
    ctrl_all = n.loc[~flagged].dropna(subset=["fwd"])
    if merged.empty or ctrl_all.empty:
        return out

    # 波动比的分母可能为 0（极端构造的净值，如全程恒定），此时无量纲比值无意义
    _w_vol = float(merged["ret1"].std())
    _c_vol = float(ctrl_all["ret1"].std())

    summary = pd.DataFrame(
        [
            {
                "事件": "★ 全部事件窗口合并",
                "重要性": 0,
                "精度": "-",
                "样本数": int(len(merged)),
                "覆盖率": float(flagged.sum() / len(n)),
                "窗口平均收益": float(merged["fwd"].mean()),
                "非事件期平均收益": float(ctrl_all["fwd"].mean()),
                "收益差": float(merged["fwd"].mean() - ctrl_all["fwd"].mean()),
                "窗口上涨概率": float((merged["fwd"] > 0).mean()),
                "非事件期上涨概率": float((ctrl_all["fwd"] > 0).mean()),
                "胜率差": float((merged["fwd"] > 0).mean() - (ctrl_all["fwd"] > 0).mean()),
                "窗口日波动": _w_vol,
                "非事件期日波动": _c_vol,
                "波动比": _w_vol / _c_vol if _c_vol else float("nan"),
            }
        ]
    )
    if out.empty:
        out = summary
    else:
        out = pd.concat([summary, out.sort_values("重要性", ascending=False)], ignore_index=True)

    out.attrs["baseline"] = {
        "全样本平均收益": base_ret,
        "全样本上涨概率": base_win,
        "全样本日波动": base_vol,
    }
    out.attrs["coverage"] = float(flagged.sum() / len(n))
    return out.reset_index(drop=True)


def event_summary_text(impact: pd.DataFrame) -> str:
    """把 :func:`event_impact` 的结果压成一句人话。

    只描述**读数**（覆盖率、波动比、胜率差），不下最终判断——
    最终判断必须结合 :func:`placebo_test`，见 :func:`verdict_text`。
    原因是单独看这张表很容易得出「事件窗口波动更低 ⇒ 事件改变了风险」这种结论，
    而安慰剂检验恰好会否掉它。
    """
    if impact is None or impact.empty:
        return "没有足够的事件样本可供检验。"

    merged = impact[impact["事件"].str.startswith("★")]
    if merged.empty:
        return "没有足够的事件样本可供检验。"

    r = merged.iloc[0]
    coverage = float(r.get("覆盖率", 0.0))

    if coverage > 0.5:
        return (
            f"⚠️ 这些事件窗口覆盖了 {coverage * 100:.0f}% 的交易日，「非事件期」样本过少，"
            "对照已经失去意义。请调窄窗口或提高重要性门槛后重测——"
            "此时任何「胜率差」都不可采信。"
        )

    ratio = float(r["波动比"])
    win_diff = float(r["胜率差"])

    if ratio >= 1.15:
        vol_txt = f"波动明显放大（{ratio:.2f} 倍）"
    elif ratio >= 1.05:
        vol_txt = f"波动略有放大（{ratio:.2f} 倍）"
    elif ratio <= 0.9:
        vol_txt = f"波动反而更小（{ratio:.2f} 倍）"
    else:
        vol_txt = f"波动基本一致（{ratio:.2f} 倍）"

    if abs(win_diff) < 0.03:
        dir_txt = "方向上与非事件期没有实质差别"
    elif win_diff > 0:
        dir_txt = f"上涨概率高 {win_diff * 100:.1f} 个百分点"
    else:
        dir_txt = f"上涨概率低 {abs(win_diff) * 100:.1f} 个百分点"

    return f"事件窗口覆盖 {coverage * 100:.0f}% 的交易日：{vol_txt}，{dir_txt}。"


def verdict_text(impact: pd.DataFrame, placebo: dict | None = None) -> str:
    """合成**最终判断**：事件对方向与风险到底有没有可检验的作用。

    这是本项目对「要不要把事件加进预测」这个问题的正式回答，也是界面上唯一
    应该出现在最顶部的那句话。逻辑固定为：

    1. 先看方向：``event_impact`` 的胜率差；
    2. 再看风险：``event_impact`` 的波动比；
    3. **用安慰剂检验裁决 1 和 2 是否只是日历巧合**——这一步是决定性的。

    第 3 步不能省。实测三只样本基金的波动比都小于 1（看着像「事件让波动变小」），
    但把事件日随机化后重算，真实值全部落在随机分布的中部：所谓效应不存在。

    Parameters
    ----------
    impact:
        :func:`event_impact` 的输出。
    placebo:
        :func:`placebo_test` 的输出；为 ``None`` 时只给读数、不给判断。
    """
    base = event_summary_text(impact)

    if not placebo:
        return base + "（尚未做安慰剂检验，因此还不能下判断。）"

    p = float(placebo.get("p_波动", 0.5))
    mode = placebo.get("模式", "shift")
    mode_txt = "随机平移 ±20 个交易日" if mode == "shift" else "在样本区间内均匀随机取同样多的日期"
    draws = int(placebo.get("n_draws", 0))
    med = float(placebo.get("随机波动比中位数", float("nan")))

    if 0.05 < p < 0.95:
        tail = (
            f"把这批事件日**{mode_txt}**后重算 {draws} 次，波动比中位数 {med:.2f}，"
            f"真实值落在第 {p * 100:.0f} 百分位——**与随机日期没有区别**。"
            "结论：事件既不能提高方向判断，也不能证明它改变了风险。"
        )
    else:
        tail = (
            f"把这批事件日**{mode_txt}**后重算 {draws} 次，波动比中位数 {med:.2f}，"
            f"真实值落在第 {p * 100:.0f} 百分位——处在随机分布的尾部，"
            "说明事件窗口确实与平常不同。但它仍只是相关，不是因果，"
            "且样本有限，只应作为风险提示。"
        )

    return base + tail + "事件真正可靠的价值在**信息**：它的日期可以提前推算，"
    "所以能用来安排「什么时候下手」；而不是用来猜涨跌。"


# ================================================================= 安慰剂检验


def placebo_test(
    nav: pd.DataFrame,
    events: pd.DataFrame,
    before: int = 0,
    after: int = 1,
    horizon: int = 1,
    min_importance: int = 1,
    n_draws: int = 200,
    max_shift: int = 20,
    mode: str = "shift",
    random_state: int = 42,
) -> dict:
    """安慰剂检验：把事件日**随机化**后重算，看真实值是否只是日历巧合。

    为什么必须做这一步
    ------------------
    实测三只样本基金都得到「事件窗口波动比 < 1」（0.85~0.93），看起来像是个
    稳定结论。但这里有一个致命的替代解释：这些事件日**不是随机分布的**——
    LPR 固定在每月 20 日、两会固定在 3 月、中央经济工作会议固定在 12 月。
    如果 A 股本身在某些日历位置波动就偏低，那么「波动比 < 1」完全可能来自
    **日历位置**，而不是事件本身。

    所以这里做两个互补的对照（``mode``）：

    ``"shift"``（平移，默认）
        把每个事件日随机平移 ``[-max_shift, +max_shift]`` 个交易日。
        保留了「事件集中在月内特定位置」这一结构，用来检验
        **具体那几天**是否特殊（而非「那个日历位置」是否特殊）。

    ``"uniform"``
        在整个样本区间内**均匀随机**取同样多的日期。
        彻底打破日历结构，用来检验「这些日期整体上是否特殊」。
        这是更严格的零假设：若真实值落在它的中部，说明事件窗与随机窗无异。

    解读
    ----
    - 落在尾部（``p_波动`` 接近 0 或 1）→ 真实窗口确实特殊，值得进一步研究；
    - 落在中间（``p_波动`` ≈ 0.5）→ **与随机化没有区别**，
      所谓「事件效应」只是日历位置带来的假象。

    Returns
    -------
    dict
        ``模式`` / ``真实波动比`` / ``真实胜率差`` / ``随机波动比中位数`` /
        ``随机波动比 P5`` / ``随机波动比 P95`` / ``随机胜率差中位数`` /
        ``波动比分位`` / ``胜率差分位`` / ``p_波动`` / ``p_方向`` /
        ``n_draws`` / ``结论``。

    Notes
    -----
    ``p_波动`` 是单侧 p 值，定义为「随机化的波动比 ≤ 真实波动比」的比例。
    接近 0 或 1 都说明真实值在尾部；接近 0.5 则说明与随机化无异。
    """
    if mode not in ("shift", "uniform"):
        raise ValueError(f"mode 只能是 'shift' 或 'uniform'，收到 {mode!r}")
    if nav is None or nav.empty or events is None or events.empty:
        return {}

    ev_sel = events[events["importance"] >= min_importance]
    if ev_sel.empty:
        return {}

    n = nav.copy()
    n["date"] = pd.to_datetime(n["date"])
    n = n.sort_values("date").reset_index(drop=True)
    n["fwd"] = n["nav"].shift(-(1 + horizon)) / n["nav"].shift(-1) - 1.0
    n["ret1"] = n["nav"].pct_change()
    idx_of = {d: i for i, d in enumerate(n["date"])}

    positions = np.array(
        [i for i in (idx_of.get(d) for d in pd.to_datetime(ev_sel["date"])) if i is not None],
        dtype=int,
    )
    if len(positions) == 0:
        return {}

    def _stats(mask: np.ndarray) -> tuple[float, float]:
        sub = n.loc[mask].dropna(subset=["fwd"])
        ctrl = n.loc[~mask].dropna(subset=["fwd"])
        if sub.empty or ctrl.empty:
            return float("nan"), float("nan")
        s_vol, c_vol = float(sub["ret1"].std()), float(ctrl["ret1"].std())
        ratio = s_vol / c_vol if c_vol else float("nan")
        win = float((sub["fwd"] > 0).mean() - (ctrl["fwd"] > 0).mean())
        return ratio, win

    def _mask_at(pos: np.ndarray) -> np.ndarray:
        m = np.zeros(len(n), dtype=bool)
        for i in pos:
            lo, hi = max(0, i - before), min(len(n) - 1, i + after)
            m[lo : hi + 1] = True
        return m

    real_ratio, real_win = _stats(_mask_at(positions))

    rng = np.random.default_rng(random_state)
    k = len(positions)
    if mode == "shift":
        rows = np.concatenate(
            [
                rng.integers(1, max_shift + 1, size=(n_draws, k)),
                -rng.integers(1, max_shift + 1, size=(n_draws, k)),
            ],
            axis=0,
        )
        cand = positions[None, :] + rows
    else:
        cand = rng.integers(0, len(n), size=(n_draws, k))

    ratios, wins = [], []
    for row in cand:
        pos = np.clip(row, 0, len(n) - 1)
        r_, w_ = _stats(_mask_at(pos))
        if not np.isnan(r_):
            ratios.append(r_)
            wins.append(w_)

    if not ratios:
        return {}

    ratios_arr = np.asarray(ratios)
    wins_arr = np.asarray(wins)

    p_vol = float((ratios_arr <= real_ratio).mean())
    p_dir = float((wins_arr <= real_win).mean())

    if 0.05 < p_vol < 0.95:
        verdict = (
            "❌ 真实值与「把事件日随机化」没有区别 —— 所谓事件效应只是"
            "日历位置带来的假象，不能当信号用。"
        )
    else:
        verdict = (
            "⚠️ 真实值落在随机分布的尾部，事件窗口确实与平常不同；"
            "但样本仍有限，且相关性不等于因果，只应作为风险提示。"
        )

    return {
        "模式": mode,
        "真实波动比": real_ratio,
        "真实胜率差": real_win,
        "随机波动比中位数": float(np.median(ratios_arr)),
        "随机波动比 P5": float(np.percentile(ratios_arr, 5)),
        "随机波动比 P95": float(np.percentile(ratios_arr, 95)),
        "随机胜率差中位数": float(np.median(wins_arr)),
        "波动比分位": p_vol,
        "胜率差分位": p_dir,
        "p_波动": p_vol,
        "p_方向": p_dir,
        "n_draws": len(ratios),
        "结论": verdict,
    }


def placebo_summary_text(result: dict) -> str:
    """把 :func:`placebo_test` 的结果压成一句人话。"""
    if not result:
        return "样本不足，无法做安慰剂检验。"
    p = float(result["p_波动"])
    mode_txt = {
        "shift": "把每个事件日随机平移 ±20 个交易日",
        "uniform": "在样本区间内均匀随机取同样多的日期",
    }.get(result.get("模式", "shift"), "随机化事件日")
    return (
        f"{mode_txt}，重算 {result['n_draws']} 次：波动比中位数 "
        f"{result['随机波动比中位数']:.2f}"
        f"（5%~95% 区间 {result['随机波动比 P5']:.2f}~{result['随机波动比 P95']:.2f}），"
        f"真实值 {result['真实波动比']:.2f} 落在第 {p * 100:.0f} 百分位。{result['结论']}"
    )
