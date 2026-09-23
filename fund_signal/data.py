# -*- coding: utf-8 -*-
"""
数据获取层。

数据源：`akshare <https://akshare.akfamily.xyz/>`_（免费、开源、无需注册 token）。
本模块是「别人 clone 下来就能跑」的关键——因此做了三件事：

1. **带重试**：公开网页接口偶发超时属正常，自动重试避免用户手动重跑。
2. **带缓存**：默认缓存 12 小时，重复运行不会反复打接口，也便于离线复现。
3. **列名兼容**：akshare 不同版本返回的列名有中英文差异，此处统一归一化。
"""

from __future__ import annotations

import re

import pandas as pd

from .config import SAMPLE_DIR
from .utils import cache_key, cached_csv, get_logger, retry

log = get_logger()

CACHE_VERSION = "v2"  # 改动归一化逻辑时递增，避免读到旧格式缓存
DEFAULT_MAX_AGE_HOURS = 12.0

_AKSHARE_HINT = (
    "无法导入 akshare。请先安装：\n"
    "    pip install akshare\n"
    "若下载缓慢，可指定国内镜像源：\n"
    "    pip install akshare -i https://mirrors.aliyun.com/pypi/simple/"
)


def _ak():
    """延迟导入 akshare —— 只在真正需要取数时才导入，便于离线测试。"""
    try:
        import akshare as ak
    except ImportError as err:
        raise ImportError(_AKSHARE_HINT) from err
    return ak


# ------------------------------------------------------------------ 列名归一化
def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_nav(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """把净值表统一为 ``date`` / ``nav`` 两列。"""
    if df is None or len(df) == 0:
        raise ValueError(f"基金 {code} 返回空数据，请确认代码是否为公募基金代码。")

    date_col = _pick(df, ["净值日期", "date", "日期", "Date", "x"])
    nav_col = _pick(df, ["单位净值", "nav", "净值", "单位净值(元)", "y"])
    if date_col is None or nav_col is None:
        raise ValueError(
            f"无法识别净值表列名，实际列为 {list(df.columns)[:10]}。"
            "可能是 akshare 版本变更，请提 Issue 反馈。"
        )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "nav": pd.to_numeric(df[nav_col], errors="coerce"),
        }
    )
    out = out.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if out.empty:
        raise ValueError(f"基金 {code} 的净值数据为空。")
    return out


def _normalize_index(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把指数日线表统一为 ``date`` / ``close`` 两列。"""
    if df is None or len(df) == 0:
        raise ValueError(f"指数 {symbol} 返回空数据。")

    date_col = _pick(df, ["date", "日期", "Date"])
    close_col = _pick(df, ["close", "收盘", "收盘价"])
    if date_col is None or close_col is None:
        raise ValueError(f"无法识别指数表列名：{list(df.columns)[:10]}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    return out.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


# ------------------------------------------------------------------ 原始请求
@retry(times=3, delay=1.5)
def _request_fund_nav(code: str) -> pd.DataFrame:
    """请求基金单位净值走势。兼容 akshare 不同版本的参数命名。"""
    ak = _ak()
    attempts = (
        dict(symbol=code, indicator="单位净值走势"),
        dict(fund=code, indicator="单位净值走势"),
        dict(symbol=code, period="成立来", indicator="单位净值走势"),
    )
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            return ak.fund_open_fund_info_em(**kwargs)
        except TypeError as err:  # 参数名不匹配，换下一种
            last_err = err
            continue
    raise RuntimeError(f"调用 akshare 失败：{last_err}")


@retry(times=3, delay=1.5)
def _request_index(symbol: str) -> pd.DataFrame:
    ak = _ak()
    return ak.stock_zh_index_daily(symbol=symbol)


@retry(times=2, delay=1.0)
def _request_fund_name() -> pd.DataFrame:
    ak = _ak()
    return ak.fund_name_em()


# ------------------------------------------------------------------ 公开接口
def fetch_fund_nav(
    code: str, use_cache: bool = True, max_age_hours: float = DEFAULT_MAX_AGE_HOURS
) -> pd.DataFrame:
    """获取基金历史单位净值。

    Parameters
    ----------
    code:
        6 位基金代码，例如 ``"000001"``。
    use_cache:
        是否使用本地缓存。
    max_age_hours:
        缓存有效期（小时）。

    Returns
    -------
    DataFrame
        两列：``date``（交易日）与 ``nav``（单位净值），按日期升序。
    """
    if not (isinstance(code, str) and len(code) == 6 and code.isdigit()):
        raise ValueError(f"基金代码必须是 6 位数字字符串，收到 {code!r}")

    key = cache_key("nav", CACHE_VERSION, code)
    df = cached_csv(
        namespace="fund_nav",
        key=key,
        loader=lambda: _normalize_nav(_request_fund_nav(code), code),
        use_cache=use_cache,
        max_age_hours=max_age_hours,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)

    _sanity_check_nav(df, code)
    log.info(
        "基金 %s：%d 条净值，%s ~ %s",
        code,
        len(df),
        df["date"].iloc[0].date(),
        df["date"].iloc[-1].date(),
    )
    return df


def fetch_index(
    symbol: str, use_cache: bool = True, max_age_hours: float = DEFAULT_MAX_AGE_HOURS
) -> pd.DataFrame:
    """获取指数日线收盘价（用于构造市场环境特征）。

    Parameters
    ----------
    symbol:
        akshare 指数代码，如 ``"sh000300"``（沪深300）、``"sz399006"``（创业板指）。
    """
    key = cache_key("idx", CACHE_VERSION, symbol)
    df = cached_csv(
        namespace="index",
        key=key,
        loader=lambda: _normalize_index(_request_index(symbol), symbol),
        use_cache=use_cache,
        max_age_hours=max_age_hours,
    )
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def fetch_fund_name(code: str, use_cache: bool = True) -> str:
    """查询基金简称；失败时退化为返回代码本身（不阻断主流程）。"""
    try:
        key = cache_key("names", CACHE_VERSION, "all")
        df = cached_csv(
            namespace="fund_names",
            key=key,
            loader=_request_fund_name,
            use_cache=use_cache,
            max_age_hours=72.0,
        )
        code_col = _pick(df, ["基金代码", "code", "代码"])
        name_col = _pick(df, ["基金简称", "name", "简称"])
        if code_col is None or name_col is None:
            return code
        hit = df[df[code_col].astype(str).str.zfill(6) == code]
        if hit.empty:
            return code
        return str(hit.iloc[0][name_col])
    except Exception as err:
        log.warning("基金名称查询失败（不影响主流程）：%s", err)
        return code


def load_sample_nav(code: str = "000001") -> pd.DataFrame:
    """读取随仓库附带的离线样例数据。

    用于两种场景：网络不可用时跑通流程，以及 CI 中的离线测试。
    """
    path = SAMPLE_DIR / f"{code}_nav.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到离线样例数据 {path}。\n"
            "可执行 `python scripts/build_sample_data.py` 生成，"
            "或联网后直接调用 fetch_fund_nav()。"
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def load_sample_index(symbol: str = "sh000300") -> pd.DataFrame:
    """读取随仓库附带的离线指数数据（列：date / close）。"""
    path = SAMPLE_DIR / f"index_{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到离线指数数据 {path}。")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def list_sample_codes() -> list[str]:
    """列出仓库中可用的离线样例基金代码，供界面下拉展示。"""
    if not SAMPLE_DIR.exists():
        return []
    return sorted(p.name.split("_")[0] for p in SAMPLE_DIR.glob("*_nav.csv"))


# ------------------------------------------------------------------ 实时 / 最新
# 这两个接口返回的表格把日期写进了列名，例如 "2026-09-23-单位净值"，
# 因此需要用正则把日期解析出来，并按日期排序取最新的一列。
_DATE_NAV_PAT = re.compile(r"^(\d{4}-\d{2}-\d{2})-单位净值$")
_DATE_EST_PAT = re.compile(r"^(\d{4}-\d{2}-\d{2})-估算数据-估算值$")
_DATE_GROWTH_PAT = re.compile(r"^(\d{4}-\d{2}-\d{2})-估算数据-估算增长率$")
_DATE_PUB_PAT = re.compile(r"^(\d{4}-\d{2}-\d{2})-公布数据-单位净值$")


@retry(times=2, delay=1.0)
def _request_open_fund_daily() -> pd.DataFrame:
    return _ak().fund_open_fund_daily_em()


@retry(times=2, delay=1.0)
def _request_value_estimation() -> pd.DataFrame:
    return _ak().fund_value_estimation_em(symbol="全部")


def _today_key() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d")


def fetch_nav_snapshot(use_cache: bool = True) -> pd.DataFrame:
    """全市场开放式基金的最新净值快照。

    该表列名内嵌日期，因此缓存键按**自然日**生成——同一天内复用，
    跨天自动失效，保证拿到的始终是最新交易日的数据，不会读到昨天那份。
    """
    key = cache_key("snapshot", CACHE_VERSION, _today_key())
    return cached_csv(
        "fund_snapshot", key, _request_open_fund_daily, use_cache=use_cache, max_age_hours=6.0
    )


def fetch_latest_nav(code: str, use_cache: bool = True) -> dict | None:
    """获取该基金**最新一个交易日**的净值，用于把净值序列补到最新。

    Returns
    -------
    dict | None
        ``{"date": Timestamp, "nav": float}``；取不到时返回 None，
        调用方应保持原序列不变（此函数不抛异常，不阻断主流程）。
    """
    try:
        snap = fetch_nav_snapshot(use_cache)
    except Exception as err:
        log.warning("最新净值快照获取失败（不影响主流程）：%s", err)
        return None

    code_col = _pick(snap, ["基金代码", "code"])
    if code_col is None or snap.empty:
        return None

    dated = [(m.group(1), c) for c in snap.columns if (m := _DATE_NAV_PAT.match(str(c)))]
    if not dated:
        return None
    dated.sort(reverse=True)  # 日期新的在前

    row = snap[snap[code_col].astype(str).str.zfill(6) == code]
    if row.empty:
        return None
    rec = row.iloc[0]

    # 注意：当日净值通常在收盘后（约 20:00）才公布，未公布时该列是 NaN
    # （页面上显示为 "---"）。所以不能只看最新那一列，而要顺延到最近一个
    # 真正有数值的交易日——否则白天运行会永远拿不到数据。
    for date_str, col in dated:
        nav = pd.to_numeric(rec[col], errors="coerce")
        if pd.notna(nav) and float(nav) > 0:
            return {"date": pd.Timestamp(date_str), "nav": float(nav)}
    return None


def fetch_realtime_estimate(code: str, use_cache: bool = True) -> dict | None:
    """盘中实时估值。

    仅在交易时段（约 09:30–15:00）有数值，其余时段可能返回 ``---``。
    估值由第三方按持仓拟合，**与最终公布净值存在偏差**，仅供参考，
    不可作为交易依据。

    Returns
    -------
    dict | None
        ``{"date", "est_nav", "est_growth", "published_nav"}``
    """
    try:
        key = cache_key("estimate", CACHE_VERSION, _today_key(), pd.Timestamp.now().strftime("%H"))
        df = cached_csv(
            "fund_estimate", key, _request_value_estimation, use_cache=use_cache, max_age_hours=0.17
        )  # 约 10 分钟
    except Exception as err:
        log.warning("实时估值获取失败（不影响主流程）：%s", err)
        return None

    code_col = _pick(df, ["基金代码", "code"])
    if code_col is None or df.empty:
        return None

    def newest_col(pattern) -> tuple[str | None, str | None]:
        hits = [(m.group(1), c) for c in df.columns if (m := pattern.match(str(c)))]
        hits.sort(reverse=True)
        return hits[0] if hits else (None, None)

    d_est, c_est = newest_col(_DATE_EST_PAT)
    _, c_growth = newest_col(_DATE_GROWTH_PAT)
    _, c_pub = newest_col(_DATE_PUB_PAT)
    if c_est is None:
        return None

    row = df[df[code_col].astype(str).str.zfill(6) == code]
    if row.empty:
        return None
    rec = row.iloc[0]

    def to_float(col) -> float | None:
        if col is None:
            return None
        raw = str(rec[col]).replace("%", "").strip()
        if raw in ("", "---", "nan", "None"):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    est_nav = to_float(c_est)
    if est_nav is None:
        return None

    growth = to_float(c_growth)
    return {
        "date": pd.Timestamp(d_est) if d_est else pd.Timestamp.now().normalize(),
        "est_nav": est_nav,
        "est_growth": (growth / 100.0) if growth is not None else None,
        "published_nav": to_float(c_pub),
    }


def fetch_fund_meta(code: str, use_cache: bool = True) -> dict:
    """从最新净值快照里取该基金的名称、申赎状态、申购费率等元信息。

    比 :func:`fetch_fund_name` 更全，且复用同一份快照，不额外发请求。
    """
    info: dict = {"code": code, "name": code}
    try:
        snap = fetch_nav_snapshot(use_cache)
    except Exception as err:
        log.warning("基金元信息获取失败（不影响主流程）：%s", err)
        return info

    code_col = _pick(snap, ["基金代码", "code"])
    if code_col is None:
        return info

    row = snap[snap[code_col].astype(str).str.zfill(6) == code]
    if row.empty:
        return info
    rec = row.iloc[0]

    for key, cands in {
        "name": ["基金简称", "name"],
        "purchase_status": ["申购状态"],
        "redeem_status": ["赎回状态"],
        "fee": ["手续费"],
        "daily_growth": ["日增长率"],
    }.items():
        col = _pick(snap, cands)
        if col is not None:
            info[key] = str(rec[col])
    return info


# ------------------------------------------------------------------ 体检
def _sanity_check_nav(df: pd.DataFrame, code: str) -> None:
    """对净值序列做基本合理性检查，尽早发现拿到了「错误的品种」。"""
    if len(df) < 60:
        log.warning("基金 %s 仅 %d 条净值，历史过短，模型结果不可靠。", code, len(df))
    nav = df["nav"]
    if nav.max() <= 0:
        raise ValueError(f"基金 {code} 净值数据异常（存在非正值）。")
    if nav.std() < 1e-8:
        log.warning(
            "基金 %s 净值几乎恒定（标准差≈0）——很可能是一只货币基金，"
            "或该品种净值按「每万份收益」计价。这类品种不适合本框架建模。",
            code,
        )


__all__ = [
    "fetch_fund_nav",
    "fetch_index",
    "fetch_fund_name",
    "fetch_fund_meta",
    "fetch_nav_snapshot",
    "fetch_latest_nav",
    "fetch_realtime_estimate",
    "load_sample_nav",
    "load_sample_index",
    "list_sample_codes",
]
