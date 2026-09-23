# -*- coding: utf-8 -*-
"""
通用工具：日志、本地缓存、带重试的请求包装。
"""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd

from .config import CACHE_DIR

T = TypeVar("T")

_LOGGER_NAME = "fund_signal"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """返回统一格式的 logger（重复调用不会重复添加 handler）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = get_logger()


def retry(
    times: int = 3,
    delay: float = 1.5,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
):
    """失败重试装饰器。

    数据源（akshare 底层走的是公开网页接口）偶发超时或限流属于正常现象，
    自动重试可以显著提升「别人下载后直接跑通」的成功率。
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            wait = delay
            last_err: BaseException | None = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as err:  # noqa: PERF203
                    last_err = err
                    if attempt == times:
                        break
                    log.warning(
                        "%s 第 %d/%d 次失败：%s，%.1fs 后重试",
                        func.__name__,
                        attempt,
                        times,
                        err,
                        wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
            assert last_err is not None
            raise last_err

        return wrapper

    return decorator


def cache_key(*parts: Any) -> str:
    """根据调用参数生成稳定的缓存文件名。"""
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return digest


def cached_csv(
    namespace: str,
    key: str,
    loader: Callable[[], pd.DataFrame],
    use_cache: bool = True,
    max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """读取 / 写入 CSV 缓存。

    Parameters
    ----------
    namespace:
        缓存子目录名，例如 ``"fund_nav"``。
    key:
        缓存键（由 :func:`cache_key` 生成）。
    loader:
        缓存未命中时调用的数据加载函数，须返回 DataFrame。
    use_cache:
        为 False 时强制重新拉取并覆盖缓存。
    max_age_hours:
        缓存有效期；超过则视为过期。设为 0 表示永不过期。
    """
    directory = CACHE_DIR / namespace
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.csv"

    if use_cache and path.exists():
        if max_age_hours <= 0 or (time.time() - path.stat().st_mtime < max_age_hours * 3600):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    log.info("命中缓存 %s（%d 行）", path.name, len(df))
                    return df
            except Exception as err:  # 缓存损坏则忽略，重新拉取
                log.warning("缓存读取失败，将重新拉取：%s", err)

    df = loader()
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def human_number(value: float, digits: int = 2, suffix: str = "") -> str:
    """把 NaN / inf 安全地格式化为字符串，避免界面出现 'nan'。"""
    try:
        if value is None or pd.isna(value):
            return "—"
        if value in (float("inf"), float("-inf")):
            return "—"
        return f"{value:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"
