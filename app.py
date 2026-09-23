# -*- coding: utf-8 -*-
"""fund-signal · Streamlit 交互工作台。

启动::

    streamlit run app.py

这一版相比最初的骨架做了四件事
------------------------------
1. **视觉**：自绘 CSS 主题（KPI 卡片 / 区块标题 / 结论条 / 标签页），
   深浅色自适应；图表标题写「结论」而不是「图表类型」。
2. **密度**：总览页 8 张 KPI + 深度诊断评分卡 + 最新特征分位定位，
   一屏之内把「模型行不行、为什么」交代清楚。
3. **深度**：滚动 Spearman IC、概率校准曲线与 ECE、Bootstrap AUC 置信区间、
   成本瀑布拆解、阈值前沿散点、回撤持续期、特征分组贡献、滚动前向窗口图。
4. **动态**：真实进度条、参数预设、免重训的回测滑块、盘中估值自动刷新、
   Plotly 区间选择器 / 滚轮缩放 / 交互工具条。

配色遵循中国市场惯例：**红涨绿跌**（与欧美相反）。
本工具输出的是统计信号，不构成投资建议。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fund_signal import data as dt
from fund_signal.backtest import backtest_threshold, sweep_thresholds
from fund_signal.config import (
    CALENDAR_FEATURES,
    DEFAULT_BENCHMARKS,
    MARKET_FEATURES,
    PRICE_FEATURES,
    Config,
)
from fund_signal.metrics import information_coefficient
from fund_signal.pipeline import run_analysis

# ================================================================= 配色主题
# 每个主题都是一套「语义色」，而不是一堆散落的十六进制。
# 图表里所有颜色都必须从这里取，保证同一含义在整个界面里颜色一致。
LIGHT: dict[str, str] = {
    "base": "light",
    "up": "#D64545",  # 涨 —— 红
    "down": "#2F9E5F",  # 跌 —— 绿
    "accent": "#C9861A",  # 重点高亮（阈值线、当前值）
    "info": "#3E7CB1",  # 中性信息 / 基准
    "ok": "#3E7CB1",
    "warn": "#C9861A",
    "bad": "#7C8798",
    "neutral": "#9AA3AE",  # 弱化（对照、辅助线）
    "text": "#1B1F24",
    "muted": "#6B7280",
    "card": "#FFFFFF",
    "band": "#F5F7FA",
    "border": "#E4E8EE",
    "grid": "rgba(120,130,145,0.14)",
    "mid": "#F2F4F7",
    "hero": "linear-gradient(135deg,#F8FAFD 0%,#EEF3FA 58%,#E7EFF8 100%)",
}

DARK: dict[str, str] = {
    "base": "dark",
    "up": "#F26A6A",
    "down": "#4FC27E",
    "accent": "#F0B34C",
    "info": "#6FA0CC",
    "ok": "#6FA0CC",
    "warn": "#F0B34C",
    "bad": "#8A94A6",
    "neutral": "#6B7686",
    "text": "#E7EBF1",
    "muted": "#98A2B3",
    "card": "#151A22",
    "band": "#12161D",
    "border": "#262D38",
    "grid": "rgba(150,160,175,0.14)",
    "mid": "#1D232D",
    "hero": "linear-gradient(135deg,#171C25 0%,#141A23 55%,#111721 100%)",
}


def theme() -> dict[str, str]:
    """当前 Streamlit 主题对应的色板。"""
    try:
        dark = (st.get_option("theme.base") or "light").lower() == "dark"
    except Exception:  # 单元测试 / AppTest 环境下可能取不到选项
        dark = False
    return DARK if dark else LIGHT


# ================================================================= 样式
# 用 @名字@ 占位而不是 f-string —— CSS 里全是花括号，f-string 会互相打架。
_CSS = """
<style>
:root{
  --up:@up@; --down:@down@; --accent:@accent@; --good:@ok@; --warn:@warn@; --bad:@bad@;
  --neutral:@neutral@; --info:@info@; --text:@text@; --muted:@muted@;
  --card:@card@; --band:@band@; --border:@border@;
}
/* 收紧默认留白，换取信息密度 */
.block-container{padding-top:2.4rem;padding-bottom:3.2rem;max-width:1560px}
[data-testid="stHeader"]{background:transparent}

/* ---------- 顶部信息条 ---------- */
.fs-hero{display:flex;flex-wrap:wrap;gap:16px;align-items:center;justify-content:space-between;
  padding:18px 22px;border-radius:16px;border:1px solid var(--border);background:@hero@;
  margin:2px 0 14px}
.fs-hero-title{font-size:23px;font-weight:700;color:var(--text);letter-spacing:.2px;line-height:1.3}
.fs-code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px;
  color:var(--accent);border:1px solid var(--accent);border-radius:7px;
  padding:1px 8px;margin-left:8px;vertical-align:middle}
.fs-hero-sub{color:var(--muted);font-size:12.5px;margin-top:6px;line-height:1.6}
.fs-badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.fs-badge{font-size:11.5px;padding:4px 11px;border-radius:999px;border:1px solid var(--border);
  color:var(--muted);background:var(--card);white-space:nowrap}
.fs-badge.hot{border-color:var(--accent);color:var(--accent)}
.fs-badge.on{border-color:var(--good);color:var(--good)}

/* ---------- KPI 卡片 ---------- */
.fs-kpi{border:1px solid var(--border);border-radius:14px;padding:13px 15px;background:var(--card);
  transition:transform .12s ease,box-shadow .12s ease,border-color .12s ease;height:100%}
.fs-kpi:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(20,30,50,.10);
  border-color:var(--accent)}
.fs-kpi-l{font-size:12px;color:var(--muted);letter-spacing:.4px;line-height:1.4}
.fs-kpi-v{font-size:24px;font-weight:700;line-height:1.3;margin:3px 0 2px;
  font-variant-numeric:tabular-nums}
.fs-kpi-v.up{color:var(--up)} .fs-kpi-v.down{color:var(--down)}
.fs-kpi-v.flat{color:var(--text)} .fs-kpi-v.accent{color:var(--accent)}
.fs-kpi-d{font-size:12.5px;font-weight:600;line-height:1.4}
.fs-kpi-d.up{color:var(--up)} .fs-kpi-d.down{color:var(--down)}
.fs-kpi-d.flat{color:var(--muted)}
.fs-kpi-s{font-size:11.5px;color:var(--muted);margin-top:5px;line-height:1.5}

/* ---------- 区块标题 ---------- */
.fs-sec{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  margin:22px 0 10px;padding-left:11px;border-left:3px solid var(--accent)}
.fs-sec-t{font-size:17px;font-weight:700;color:var(--text)}
.fs-sec-d{font-size:12.5px;color:var(--muted)}

/* ---------- 结论条 ---------- */
.fs-note{border-radius:10px;padding:11px 15px;font-size:13.5px;line-height:1.7;
  border-left:4px solid var(--neutral);background:var(--band);color:var(--text);
  margin:9px 0 5px}
.fs-note.ok{border-left-color:var(--good)}
.fs-note.warn{border-left-color:var(--warn)}
.fs-note.bad{border-left-color:var(--bad)}
.fs-note.info{border-left-color:var(--info)}
.fs-note code{background:rgba(127,127,127,.14);padding:1px 5px;border-radius:4px;
  font-size:12.5px}
.fs-note b{color:var(--accent)}

/* ---------- 标签页（Streamlit 1.6x 用 data-testid 而不是 data-baseweb）---------- */
.stTabs [role="tablist"]{gap:5px;border-bottom:1px solid var(--border)}
.stTabs [data-testid="stTab"]{height:42px;padding:0 16px;border-radius:10px 10px 0 0;
  font-size:14px;font-weight:600;color:var(--muted)}
.stTabs [data-testid="stTab"]:hover{color:var(--text)}
.stTabs [data-testid="stTab"][aria-selected="true"]{color:var(--accent)!important}
.stTabs [data-testid="stTab"] p{font-size:inherit;font-weight:inherit}

/* ---------- 侧边栏 / 表格 / 滚动条 ---------- */
[data-testid="stSidebar"]{border-right:1px solid var(--border)}
[data-testid="stSidebar"] .block-container{padding-top:1.4rem}
[data-testid="stSidebar"] hr{margin:.7rem 0}
[data-testid="stDataFrame"]{border:1px solid var(--border);border-radius:10px}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:6px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}
</style>
"""


def inject_css() -> None:
    t = theme()
    css = _CSS
    for key, val in t.items():
        css = css.replace(f"@{key}@", val)
    st.markdown(css, unsafe_allow_html=True)


def _flat(html: str) -> str:
    """压掉换行 —— 否则 Streamlit 的 markdown 解析会给 HTML 套上 <p>，布局全乱。"""
    return " ".join(part.strip() for part in html.strip().splitlines())


def section(title: str, desc: str = "") -> None:
    """区块标题：带左侧主色竖条。"""
    d = f'<span class="fs-sec-d">{desc}</span>' if desc else ""
    st.markdown(_flat(f'<div class="fs-sec"><span class="fs-sec-t">{title}</span>{d}</div>'),
                unsafe_allow_html=True)


def note(html: str, tone: str = "info") -> None:
    """结论条。tone: info / ok / warn / bad。"""
    st.markdown(_flat(f'<div class="fs-note {tone}">{html}</div>'), unsafe_allow_html=True)


def hero(title: str, code: str, sub: str, badges: list[tuple[str, str]]) -> None:
    """顶部基金信息条。badges 为 (文本, 样式) 列表，样式 ∈ {"", "hot", "on"}。"""
    chips = "".join(
        f'<span class="fs-badge {cls}">{text}</span>' for text, cls in badges
    )
    st.markdown(
        _flat(
            f'<div class="fs-hero">'
            f'<div><div class="fs-hero-title">{title}<span class="fs-code">{code}</span></div>'
            f'<div class="fs-hero-sub">{sub}</div></div>'
            f'<div class="fs-badges">{chips}</div>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


# ================================================================= 数字格式
def fnum(v, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def fpct(v, digits: int = 2, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:+.{digits}f}%" if sign else f"{v * 100:.{digits}f}%"


def fint(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{int(v):,}"


def tone_of(v, *, zero_is_flat: bool = True, invert: bool = False) -> str:
    """把数值映射成 up / down / flat 三种语气（红涨绿跌）。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "flat"
    if zero_is_flat and abs(float(v)) < 1e-12:
        return "flat"
    good = float(v) > 0
    if invert:
        good = not good
    return "up" if good else "down"


def kpi(label: str, value: str, delta: str | None = None, *, tone: str = "flat",
        sub: str = "", help_text: str = "") -> None:
    """自绘 KPI 卡片 —— 比 st.metric 多一行脚注，且配色完全可控。"""
    tip = f' title="{help_text}"' if help_text else ""
    delta_html = f'<div class="fs-kpi-d {tone}">{delta}</div>' if delta else ""
    sub_html = f'<div class="fs-kpi-s">{sub}</div>' if sub else ""
    st.markdown(
        _flat(
            f'<div class="fs-kpi"{tip}>'
            f'<div class="fs-kpi-l">{label}</div>'
            f'<div class="fs-kpi-v {tone}">{value}</div>'
            f"{delta_html}{sub_html}</div>"
        ),
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict], cols: int = 4) -> None:
    """把 KPI 列表按 cols 列铺开，最后一行不足也安全。"""
    for start in range(0, len(items), cols):
        chunk = items[start: start + cols]
        for col, item in zip(st.columns(cols, gap="small"), chunk, strict=False):
            with col:
                kpi(**item)


# ================================================================= 图表工具
PLOTLY_CONFIG: dict = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def style_fig(fig, title: str = "", height: int = 380, *, legend: bool = False):
    """统一图表主题。title 请写结论，不要写图表类型。"""
    t = theme()
    fig.update_layout(
        title=dict(text=title, font=dict(size=13.5, color=t["text"]), x=0, xanchor="left"),
        height=height,
        margin=dict(l=6, r=8, t=54 if title else 20, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=t["text"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11.5)),
        hoverlabel=dict(font_size=12),
        bargap=0.32,
    )
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False, showline=False,
                     tickcolor=t["grid"], ticks="outside")
    fig.update_yaxes(gridcolor=t["grid"], zeroline=False, showline=False,
                     tickcolor=t["grid"], ticks="outside")
    return fig


def chart(fig, *, height: int | None = None, key: str | None = None) -> None:
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key)


def time_selector(fig, yaxis: str = "y") -> None:
    """给时间轴加「近3月/近1年/全部」快捷按钮。"""
    t = theme()
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1月", step="month", stepmode="backward"),
                dict(count=3, label="3月", step="month", stepmode="backward"),
                dict(count=6, label="6月", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(step="all", label="全部"),
            ],
            bgcolor=t["band"],
            activecolor=t["accent"],
            font=dict(color=t["text"], size=11),
            x=0,
            y=1.16,
            xanchor="left",
            yanchor="top",
        ),
        rangeselector_visible=True,
    )


# ================================================================= 深度分析工具
@st.cache_data(show_spinner=False, ttl=3600)
def bootstrap_auc(y_true: pd.Series, prob: pd.Series, n_boot: int = 400,
                  seed: int = 42) -> dict:
    """AUC 的 Bootstrap 置信区间。

    AUC = 0.53 到底算不算「有预测力」？单看数字没有意义——必须看它的抽样分布。
    这里对样本做 400 次有放回重抽，得到 AUC 的 95% 区间，
    并用「重抽中 AUC ≤ 0.5 的比例」近似单边 p 值。

    这是判断「模型是否只是运气好」最有说服力的一张图。
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if y.size < 30 or np.unique(y).size < 2:
        return {"lo": float("nan"), "hi": float("nan"), "median": float("nan"),
                "p_le_half": float("nan"), "n": 0, "samples": np.array([])}

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, y.size, y.size)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(float(roc_auc_score(yy, p[idx])))
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "median": float("nan"),
                "p_le_half": float("nan"), "n": 0, "samples": np.array([])}

    arr = np.asarray(vals)
    return {
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "median": float(np.median(arr)),
        "p_le_half": float((arr <= 0.5).mean()),
        "n": int(arr.size),
        "samples": arr,
    }


def calibration_table(prob, y_true, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """概率校准表 + 期望校准误差 ECE。

    一个 AUC 尚可的模型，如果概率全都挤在 0.48~0.52，
    那它输出的「概率」其实不能当概率用。ECE 量化了这个偏差：
    ``Σ (各档样本占比 × |该档实际频率 − 该档平均预测概率|)``。
    """
    p = pd.Series(np.asarray(prob, dtype=float))
    y = pd.Series(np.asarray(y_true, dtype=float))
    df = pd.DataFrame({"p": p.to_numpy(), "y": y.to_numpy()}).dropna()
    if df.empty:
        return pd.DataFrame(), float("nan")
    try:
        df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame(), float("nan")
    g = (df.groupby("bin", observed=True)
           .agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
           .reset_index())
    ece = float((g["n"] / g["n"].sum() * (g["obs"] - g["pred"]).abs()).sum())
    return g, ece


def rolling_spearman(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """逐窗口计算 Spearman 秩相关（滚动 IC）。

    为什么不用 ``rolling().corr()``？那个算的是 Pearson。
    IC 在量化里的定义就是秩相关，两者在金融收益这种厚尾数据上差别不小。
    样本量在这里只有几千行，逐窗口排秩的开销完全可以接受。
    """
    out = np.full(len(a), np.nan)
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    min_ok = max(10, window // 3)
    for i in range(window - 1, len(av)):
        x = av[i - window + 1: i + 1]
        y = bv[i - window + 1: i + 1]
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < min_ok:
            continue
        xr = pd.Series(x[m]).rank().to_numpy()
        yr = pd.Series(y[m]).rank().to_numpy()
        if xr.std() < 1e-12 or yr.std() < 1e-12:
            continue
        out[i] = float(np.corrcoef(xr, yr)[0, 1])
    return pd.Series(out, index=a.index)


def drawdown_episodes(equity: pd.Series, top: int = 3) -> pd.DataFrame:
    """识别最深的若干段回撤，返回开始 / 谷底 / 恢复日期与持续天数。

    「最大回撤 -18%」这种单点数字隐瞒了过程：是三个月阴跌，还是三天插水？
    两者的持有体验完全不同。
    """
    eq = equity.dropna()
    if len(eq) < 5:
        return pd.DataFrame()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    episodes: list[dict] = []
    in_dd = False
    start = trough = eq.index[0]
    min_dd = 0.0
    for date, val in dd.items():
        if val < -1e-9:
            if not in_dd:
                in_dd, start, min_dd = True, date, val
                trough = date
            if val < min_dd:
                min_dd, trough = val, date
        elif in_dd:
            episodes.append(
                {"开始": start, "谷底": trough, "恢复": date, "最大回撤": min_dd,
                 "持续(交易日)": int(eq.index.get_loc(date) - eq.index.get_loc(start))}
            )
            in_dd = False
    if in_dd:
        episodes.append(
            {"开始": start, "谷底": trough, "恢复": None, "最大回撤": min_dd,
             "持续(交易日)": int(len(eq) - 1 - eq.index.get_loc(start))}
        )
    if not episodes:
        return pd.DataFrame()
    return (pd.DataFrame(episodes)
              .sort_values("最大回撤")
              .head(top)
              .reset_index(drop=True))


FEATURE_GROUPS: dict[str, list[str]] = {
    "价格 / 动量": PRICE_FEATURES,
    "市场环境": MARKET_FEATURES,
    "日历效应": CALENDAR_FEATURES,
}


# ================================================================= 侧边栏
PRESETS: dict[str, dict] = {
    "快速体验（逻辑回归 3 折）": dict(horizon=1, n_splits=3, model_type="logistic",
                                      init_ratio=0.5),
    "标准（LightGBM 5 折）": dict(horizon=1, n_splits=5, model_type="lightgbm",
                                  init_ratio=0.5),
    "严格 · 深度（LightGBM 8 折 / 2 日）": dict(horizon=2, n_splits=8,
                                                model_type="lightgbm", init_ratio=0.6),
}

QUICK_CODES: dict[str, str] = {
    "000001 华夏成长": "000001",
    "161725 招商中证白酒": "161725",
    "320007 诺安成长": "320007",
    "110022 易方达消费行业": "110022",
    "003096 中欧医疗健康": "003096",
}


def _apply_quick_code() -> None:
    picked = st.session_state.get("fs_quick_code")
    if picked:
        st.session_state["fs_code"] = QUICK_CODES[picked]


def render_sidebar() -> dict:
    sb = st.sidebar
    sb.markdown("### ⚙️ 控制台")
    sb.caption("改动任何参数后，重新点击「开始分析」即可。")

    # ---------------- 参数预设 ----------------
    preset = sb.selectbox(
        "参数预设",
        ["自定义", *PRESETS],
        key="fs_preset",
        help="选择后自动套用下方参数；套用后仍可逐项微调。",
    )
    if preset != "自定义" and st.session_state.get("fs_preset_applied") != preset:
        for key, val in PRESETS[preset].items():
            st.session_state[f"fs_{key}"] = val
        st.session_state["fs_preset_applied"] = preset
        st.rerun()

    sb.divider()

    # ---------------- ① 标的 ----------------
    sb.markdown("**① 标的**")
    sb.pills("快捷选择", list(QUICK_CODES), key="fs_quick_code",
             on_change=_apply_quick_code, label_visibility="collapsed")
    code = sb.text_input("基金代码", value="000001", key="fs_code",
                         help="6 位公募基金代码。离线样例仅含 000001 / 161725 / 320007。")
    bench_label = sb.selectbox("基准指数", list(DEFAULT_BENCHMARKS.keys()), key="fs_bench",
                               help="用于构造市场环境与相对强弱特征")
    data_source = sb.radio(
        "数据来源", ["akshare", "local"], horizontal=True, key="fs_source",
        help="akshare = 联网实时取数（含盘中估值）；local = 仓库自带离线样例，断网也能跑",
    )

    sb.divider()

    # ---------------- ② 预测任务 ----------------
    sb.markdown("**② 预测任务**")
    horizon = sb.slider("预测周期（交易日）", 1, 20, 1, key="fs_horizon",
                        help="预测未来多少个交易日的涨跌方向")
    init_ratio = sb.slider("初始训练集占比", 0.3, 0.8, 0.5, 0.05, key="fs_init_ratio",
                           help="滚动前向验证中，前多少比例的历史只用于训练、从不出现在预测集")

    sb.divider()

    # ---------------- ③ 模型 ----------------
    sb.markdown("**③ 模型**")
    model_type = sb.selectbox("模型类型", ["lightgbm", "logistic"], key="fs_model_type",
                              help="LightGBM 梯度提升树；logistic 为带标准化的逻辑回归基线")
    n_splits = sb.slider("滚动验证折数", 2, 10, 5, key="fs_n_splits")
    seed = sb.number_input("随机种子", value=42, step=1, key="fs_seed")

    sb.divider()

    # ---------------- ④ 成本与实时 ----------------
    sb.markdown("**④ 成本与实时**")
    cost_bps = sb.slider("单边成本（基点）", 0.0, 100.0, 15.0, 1.0, key="fs_cost_bps",
                         help="1 基点 = 0.01%。C 类基金免申购费但有销售服务费")
    realtime = sb.toggle("启用实时数据", value=True, key="fs_realtime",
                         help="把净值序列补齐到最新交易日，并尝试抓取盘中估值")
    auto_refresh = sb.toggle(
        "盘中估值自动刷新", value=False, key="fs_auto",
        help="开启后每 60 秒重新拉取一次实时估值（仅 akshare 数据源有效）",
    )

    sb.divider()
    c1, c2 = sb.columns(2)
    run = c1.button("🚀 开始分析", type="primary", width="stretch", key="fs_run")
    refresh = c2.button("🔄 刷新数据", width="stretch", key="fs_refresh",
                        help="绕过本地缓存，强制重新向数据源拉取")

    sb.caption(
        "⚠️ 本工具用于方法论演示与学习，输出为统计信号，"
        "**不构成任何投资建议**。基金净值短期走势接近随机。"
    )
    return dict(
        code=code,
        benchmark=DEFAULT_BENCHMARKS[bench_label],
        benchmark_label=bench_label,
        data_source=data_source,
        horizon=horizon,
        init_ratio=init_ratio,
        model_type=model_type,
        n_splits=n_splits,
        seed=int(seed),
        cost_bps=cost_bps,
        realtime=realtime,
        auto_refresh=auto_refresh,
        run=run,
        refresh=refresh,
    )


# ================================================================= 分析调度
_CACHE_LIMIT = 8


def _analysis_params(params: dict, use_cache: bool) -> dict:
    """只挑真正影响训练结果的参数；阈值 / 成本放在回测层另算，不触发重训。"""
    return dict(
        code=params["code"],
        horizon=int(params["horizon"]),
        model_type=params["model_type"],
        benchmark=params["benchmark"],
        n_splits=int(params["n_splits"]),
        cost_bps=float(params["cost_bps"]),
        seed=int(params["seed"]),
        data_source=params["data_source"],
        init_ratio=float(params["init_ratio"]),
        use_cache=bool(use_cache),
        realtime=bool(params["realtime"]),
    )


def _run_pipeline(p: dict):
    cfg = Config(
        fund_code=p["code"],
        horizon=p["horizon"],
        benchmark=p["benchmark"],
        model_type=p["model_type"],
        n_splits=p["n_splits"],
        cost_bps=p["cost_bps"],
        random_state=p["seed"],
        train_ratio=p["init_ratio"],
        use_cache=p["use_cache"],
        realtime=p["realtime"],
    )
    return run_analysis(cfg, data_source=p["data_source"])


def obtain_result(p: dict):
    """取结果：命中会话缓存直接返回，否则带真实进度条跑一遍流水线。

    这里刻意不用 ``st.cache_data`` 缓存最终结果——它的返回值缓存无法驱动
    进度回调，首次联网取数时界面会长时间无反馈。改成会话级缓存
    （``st.session_state``）后，既能在同参数下秒开，又能把 ``run_analysis``
    的 ``progress`` 回调实时画到进度条上。

    数据层仍然有磁盘缓存（``Config(use_cache=True)``），所以重复运行不会
    反复打接口。
    """
    key = tuple(sorted(p.items()))
    store: OrderedDict = st.session_state.setdefault("fs_store", OrderedDict())
    if key in store:
        store.move_to_end(key)
        return store[key]

    cfg = Config(
        fund_code=p["code"], horizon=p["horizon"], benchmark=p["benchmark"],
        model_type=p["model_type"], n_splits=p["n_splits"], cost_bps=p["cost_bps"],
        random_state=p["seed"], train_ratio=p["init_ratio"],
        use_cache=p["use_cache"], realtime=p["realtime"],
    )
    with st.status("正在分析…", expanded=True) as status:
        bar = st.progress(0.0, text="准备中…")

        def on_progress(msg: str, frac: float) -> None:
            bar.progress(min(max(float(frac), 0.0), 1.0), text=msg)

        result = run_analysis(cfg, data_source=p["data_source"], progress=on_progress)
        bar.progress(1.0, text="完成")
        status.update(label="分析完成", state="complete", expanded=False)

    store[key] = result
    while len(store) > _CACHE_LIMIT:
        store.popitem(last=False)
    return result


# ================================================================= 欢迎页
def render_welcome() -> None:
    hero(
        "fund-signal",
        "量化信号工作台",
        "从数据获取、特征工程，到滚动前向验证与样本外回测的完整链路。"
        "所有指标都来自「用过去预测未来」的样本外结果。",
        [("walk-forward 验证", "hot"), ("T+1 执行延迟", ""), ("含交易成本", ""),
         ("红涨绿跌", "")],
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        kpi("严格时序验证", "walk-forward", "训练数据永远早于预测样本",
            sub="滚动前向推进，杜绝信息穿越", tone="flat")
    with c2:
        kpi("可交易收益", "T+1 执行", "买入价 = T+1 日净值",
            sub="不假设「看完净值还能按净值成交」", tone="flat")
    with c3:
        kpi("诚实评估", "对基线", "多数类基线 + AUC 置信区间",
            sub="直接告诉你模型到底有没有优势", tone="flat")
    with c4:
        kpi("成本内建", "双边计费", "默认 15 基点单边",
            sub="忽略成本的回测没有意义", tone="flat")

    note(
        "<b>先读这段再动手。</b> 公募基金净值的日频方向预测，在学术与业界都缺少"
        "可靠证据支持（弱式有效市场）。本项目的价值在于<b>演示一套严谨的量化研究流程</b>，"
        "以及<b>展示「预测准确率略高于 50% 却依然亏钱」这一真实规律</b>，"
        "而不是提供一个能赚钱的策略。",
        "warn",
    )

    section("上手三步", "在左侧控制台完成设置")
    a, b, c = st.columns(3, gap="small")
    with a:
        note("<b>① 选标的</b><br>点快捷按钮，或直接填 6 位基金代码。"
             "断网时把「数据来源」切成 <code>local</code> 用离线样例。", "info")
    with b:
        note("<b>② 选预设</b><br>想要立刻看到东西就选「快速体验」；"
             "做正式结论选「严格 · 深度」。", "info")
    with c:
        note("<b>③ 点分析</b><br>首次联网取数需要几十秒，之后同参数秒开。"
             "结果会缓存 8 组，方便来回对比。", "info")


# ================================================================= 总览
def render_freshness(res) -> None:
    """数据新鲜度 + 盘中估值（可选自动刷新）。"""
    last = res.last_nav_date
    if last is None:
        return

    age = res.data_age_days
    est = res.realtime_estimate
    meta = res.realtime.get("meta", {})
    code = res.config.fund_code
    auto = bool(st.session_state.get("fs_auto", False))
    remote = bool(res.config.realtime)

    def freshness_card() -> None:
        if age <= 1:
            kpi("数据新鲜度", f"{last.date()}", "已是最新交易日",
                sub="净值于每交易日约 20:00 后公布", tone="up")
        elif age <= 3:
            kpi("数据新鲜度", f"{last.date()}", f"{int(age)} 天前",
                sub="仍在合理范围", tone="flat")
        else:
            kpi("数据新鲜度", f"{last.date()}", f"滞后 {int(age)} 天",
                sub="建议点「🔄 刷新数据」", tone="down")

    def realtime_cards(payload: dict) -> None:
        items = []
        if payload:
            growth = payload.get("est_growth")
            items.append(dict(
                label="盘中实时估值",
                value=f"{payload['est_nav']:.4f}",
                delta=(f"{growth:+.2%}" if growth is not None else None),
                tone=tone_of(growth),
                sub=f"估值日 {pd.Timestamp(payload['date']).date()}　"
                    "第三方按持仓拟合，与公布净值有偏差",
            ))
        else:
            items.append(dict(
                label="盘中实时估值", value="—",
                sub="数据源未覆盖该基金，或当前非交易时段",
            ))
        items.append(dict(label="申购状态", value=meta.get("purchase_status", "—"),
                          sub=f"赎回状态 {meta.get('redeem_status', '—')}"))
        items.append(dict(label="申购费率", value=meta.get("fee", "—"),
                          sub="C 类通常免申购费、另收销售服务费"))
        kpi_row(items, cols=3)

    left, right = st.columns([1, 2.6], gap="small")

    with left:
        freshness_card()
        if remote:
            tag = "每 60 秒自动刷新" if auto else "实时数据已启用"
            st.caption(f"· {tag}")

    with right:
        if auto and remote:
            # 用 fragment 让这一块独立于整页重跑，避免打断用户操作
            @st.fragment(run_every=60)
            def _rt(_code: str, _fallback: dict) -> None:
                payload = _fallback
                try:
                    fresh = dt.fetch_realtime_estimate(_code, use_cache=True)
                    if fresh:
                        payload = fresh
                except Exception:  # 网络抖动不该影响界面
                    payload = _fallback
                realtime_cards(payload)

            _rt(code, est or {})
        else:
            realtime_cards(est or {})

    st.caption(
        f"本地取数时间 {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}　·　"
        f"基准 {res.config.benchmark}　·　"
        f"数据源 {'akshare（联网）' if remote else 'local（离线样例）'}"
    )


def render_diagnosis(res, bt) -> None:
    """深度诊断评分卡 —— 把「模型到底行不行」拆成 9 条可核查的判据。"""
    frame = res.dataset.frame
    prob = res.wf.predictions
    y_true = frame.loc[prob.index, "label"].astype(int)
    fwd = frame.loc[prob.index, "fwd_ret"]

    boot = bootstrap_auc(y_true, prob, seed=res.config.random_state)
    _, ece = calibration_table(prob, y_true, n_bins=10)
    ic = information_coefficient(prob, fwd)
    folds = pd.DataFrame(res.wf.fold_metrics)
    fold_auc = folds["auc"].dropna() if not folds.empty else pd.Series(dtype=float)
    fold_hit = float((fold_auc > 0.5).mean()) if len(fold_auc) else float("nan")
    fold_std = float(fold_auc.std(ddof=1)) if len(fold_auc) > 1 else float("nan")

    base_rate = float(y_true.mean())
    prob_spread = float(prob.std(ddof=1))
    sw = sweep_thresholds(prob, fwd, cost_bps=bt.cost_bps, execution_lag=bt.execution_lag)
    beat_ratio = float((sw["excess_vs_hold"] > 0).mean()) if not sw.empty else float("nan")

    def verdict(ok: bool, warn: bool = False) -> str:
        if ok:
            return "✅ 成立"
        return "⚠️ 边界" if warn else "❌ 不成立"

    rows = [
        {"判据": "AUC 显著高于 0.5", "数值": f"{fnum(res.auc)}　95%CI [{fnum(boot['lo'],3)}, {fnum(boot['hi'],3)}]",
         "判定": verdict(boot["lo"] > 0.5, boot["hi"] > 0.5),
         "说明": "置信区间下界高于 0.5，才能说「不是运气」"},
        {"判据": "超额准确率 > 2%", "数值": fnum(res.wf.overall_metrics.get("acc_edge"), 4),
         "判定": verdict(float(res.wf.overall_metrics.get("acc_edge") or 0) > 0.02),
         "说明": "相对「永远猜多数类」基线的增量"},
        {"判据": "折间稳定性（各折 AUC > 0.5 占比）", "数值": fpct(fold_hit, 0),
         "判定": verdict(fold_hit >= 0.6, fold_hit >= 0.4),
         "说明": "若只有个别折有效，多半是过拟合"},
        {"判据": "折间 AUC 波动", "数值": fnum(fold_std, 4),
         "判定": verdict(fold_std < 0.05, fold_std < 0.08),
         "说明": "标准差越小，结论越可复现"},
        {"判据": "|IC| ≥ 0.03", "数值": fnum(ic, 4),
         "判定": verdict(abs(ic) >= 0.03, abs(ic) >= 0.015),
         "说明": "不依赖阈值的单调关系；低于 0.03 基本等同噪音"},
        {"判据": "概率校准 ECE < 0.05", "数值": fnum(ece, 4),
         "判定": verdict(ece < 0.05, ece < 0.08),
         "说明": "概率能否当概率用，而不只是排序"},
        {"判据": "概率分化度（输出标准差）", "数值": fnum(prob_spread, 4),
         "判定": verdict(prob_spread > 0.05, prob_spread > 0.03),
         "说明": "全都挤在 0.5 附近 = 模型不敢表态"},
        {"判据": "阈值稳健（扫描中跑赢持有的比例）", "数值": fpct(beat_ratio, 0),
         "判定": verdict(beat_ratio >= 0.6, beat_ratio >= 0.4),
         "说明": "只在某一阈值上亮眼，就是过拟合的信号"},
        {"判据": "样本量 ≥ 350 行", "数值": fint(len(frame)),
         "判定": verdict(len(frame) >= 350, len(frame) >= 200),
         "说明": "小样本下 AUC 的抽样噪声极大"},
    ]
    cards = pd.DataFrame(rows)

    pass_n = int(cards["判定"].str.startswith("✅").sum())
    warn_n = int(cards["判定"].str.startswith("⚠️").sum())
    total = len(cards)

    if pass_n >= 6:
        tone, head = "ok", "多项判据成立"
    elif pass_n + warn_n >= 5:
        tone, head = "warn", "结论不稳，多数判据只在边界上"
    else:
        tone, head = "bad", "基本没有可利用的统计优势"

    note(
        f"<b>{head}</b>：{total} 条判据中 <b>{pass_n} 条成立</b>、{warn_n} 条处于边界、"
        f"{total - pass_n - warn_n} 条不成立。"
        f"上涨基准率 {fpct(base_rate, 1)}——意味着「永远猜涨」就能拿到 "
        f"{fpct(res.wf.overall_metrics.get('majority_acc'), 1)} 的准确率，"
        "任何模型都必须先跨过这条线。"
        + ("" if pass_n >= 6 else
           " <b>在这种情况下不要相信回测曲线</b>——"
           "策略赚钱往往来自市场本身的 beta，而不是模型的 alpha。"),
        tone,
    )
    st.dataframe(cards, width="stretch", hide_index=True)


def render_latest_feature_position(res) -> None:
    """最新特征行在历史分布中的位置 —— 模型此刻「看到了什么」。"""
    latest = res.dataset.latest
    if not len(latest):
        return
    frame = res.dataset.frame
    names = res.feature_names
    row = latest.iloc[-1]
    hist = frame[names]
    mean = hist.mean()
    std = hist.std(ddof=1).replace(0.0, np.nan)
    z = ((row - mean) / std).dropna()
    pct = pd.Series(
        {name: float((hist[name] < row[name]).mean()) for name in names}, dtype=float
    )

    tbl = pd.DataFrame({
        "特征": z.index,
        "当前值": row[z.index].to_numpy(),
        "历史均值": mean[z.index].to_numpy(),
        "z 分数": z.to_numpy(),
        "历史分位": pct[z.index].to_numpy(),
    })
    tbl["历史分位"] = tbl["历史分位"].map(lambda v: fpct(v, 0))
    tbl = tbl.reindex(tbl["z 分数"].abs().sort_values(ascending=False).index)

    top = tbl.head(8)
    fig = go.Figure(
        go.Bar(
            x=top["z 分数"][::-1],
            y=top["特征"][::-1],
            orientation="h",
            marker_color=[theme()["up"] if v >= 0 else theme()["down"]
                          for v in top["z 分数"][::-1]],
            text=[f"{v:+.2f}σ · 分位 {p}" for v, p in
                  zip(top["z 分数"][::-1], top["历史分位"][::-1], strict=True)],
            textposition="outside",
            hovertemplate="%{y}<br>z = %{x:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=theme()["neutral"], line_width=1)
    fig.update_xaxes(title="z 分数（正 = 高于历史均值）")
    lim = max(2.5, float(top["z 分数"].abs().max()) * 1.35)
    fig.update_xaxes(range=[-lim, lim])
    chart(style_fig(fig, f"此刻最「异常」的 8 个特征（特征日 {res.latest_date.date()}）", 360))
    st.caption(
        "z 分数衡量当前值偏离历史均值多少个标准差。偏离越大，模型这一期的输入"
        "越处于历史少见的区域——**样本外外推的风险也越高**。"
    )


def render_overview(res) -> None:
    m = res.summary_metrics()
    nav = res.nav
    bt = res.backtest
    prob = res.latest_prob

    badges = [
        (f"{res.config.model_type}", "hot"),
        (f"预测周期 {res.config.horizon} 日", ""),
        (f"{res.config.n_splits} 折 walk-forward", ""),
        (f"单边 {res.config.cost_bps:.0f} 基点", ""),
        (f"{m['n_features']} 特征", ""),
    ]
    hero(
        res.fund_name,
        res.config.fund_code,
        f"净值区间 {nav['date'].iloc[0].date()} ~ {nav['date'].iloc[-1].date()}　·　"
        f"{fint(len(nav))} 个交易日　·　可用样本 {fint(m['n_samples'])} 行"
        f"　·　模型 {res.config.model_type}",
        badges,
    )

    render_freshness(res)

    # ---------------- 最新信号 ----------------
    section("最新信号", f"特征日 {res.latest_date.date() if res.latest_date is not None else '—'}")
    left, right = st.columns([1, 2.1], gap="small")

    thr = res.config.threshold
    with left:
        if not np.isnan(prob):
            direction = "偏多" if prob >= thr else "偏空 / 观望"
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(prob) * 100.0,
                    number=dict(suffix="%", font=dict(size=34, color=theme()["text"])),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100],
                                  tickfont=dict(size=10, color=theme()["muted"])),
                        bar=dict(color=theme()["up"] if prob >= thr else theme()["down"],
                                 thickness=0.34),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 35], color=theme()["band"]),
                            dict(range=[35, 50], color=theme()["mid"]),
                            dict(range=[50, 65], color=theme()["band"]),
                            dict(range=[65, 100], color=theme()["mid"]),
                        ],
                        threshold=dict(line=dict(color=theme()["accent"], width=3),
                                       thickness=0.86, value=thr * 100),
                    ),
                )
            )
            style_fig(fig, "", 230)
            fig.update_layout(margin=dict(l=18, r=18, t=8, b=4))
            chart(fig)
            st.caption(f"竖线为看多阈值 {thr:.2f}　→　**{direction}**")
        else:
            note("未能构造最新特征行，无法给出当期信号。", "bad")

    with right:
        rows = [
            dict(label="样本外 AUC", value=fnum(m["auc"]),
                 delta=f"相对 0.5 超额 {fnum((m['auc'] or np.nan) - 0.5, 4)}",
                 tone=tone_of((m["auc"] or np.nan) - 0.5, zero_is_flat=False),
                 sub="0.5 = 与抛硬币无异"),
            dict(label="准确率 vs 基线", value=fpct(m["accuracy"], 1),
                 delta=f"超额 {fnum(m['acc_edge'], 4)}",
                 tone=tone_of(m["acc_edge"]),
                 sub=f"多数类基线 {fpct(m['majority_acc'], 1)}"),
            dict(label="策略累计收益", value=fpct(m["strategy_return"], 1),
                 delta=f"买入持有 {fpct(m['benchmark_return'], 1)}",
                 tone=tone_of(m["strategy_return"]),
                 sub=f"超额 {fpct(m['excess_return'], 1)}"),
            dict(label="夏普比率", value=fnum(m["sharpe"], 2),
                 tone=tone_of(m["sharpe"]), sub="无风险利率按 0 处理"),
            dict(label="最大回撤", value=fpct(m["max_drawdown"], 1),
                 tone="down" if (m["max_drawdown"] or 0) < -0.15 else "flat",
                 sub="越大越考验持有耐心"),
            dict(label="持仓时间占比", value=fpct(m["exposure"], 1),
                 tone="flat", sub=f"样本外 {fint(bt.strategy_metrics.get('n_days'))} 个交易日"),
        ]
        kpi_row(rows, cols=3)

    if res.has_edge:
        note(
            f"模型准确率 {fpct(m['accuracy'])} 高于多数类基线 {fpct(m['majority_acc'])}，"
            f"超额 <b>{fnum(m['acc_edge'], 4)}</b>——存在微弱但可测的统计信号。"
            "注意：微弱信号通常<b>不足以覆盖交易成本</b>，请看回测页的净结果与成本瀑布。",
            "ok",
        )
    else:
        note(
            f"模型准确率 {fpct(m['accuracy'])} <b>未明显超过</b>多数类基线 "
            f"{fpct(m['majority_acc'])}（超额 {fnum(m['acc_edge'], 4)}）。"
            "这说明该基金在所选周期上的方向变化接近随机——"
            "<b>这是最常见、也最诚实的结果</b>。",
            "warn",
        )

    for item in res.notes:
        st.caption(f"· {item}")

    # ---------------- 深度诊断 ----------------
    section("深度诊断", "9 条可核查的判据，而不是一句「效果不错」")
    render_diagnosis(res, bt)

    # ---------------- 净值 ----------------
    section("净值走势", f"{nav['date'].iloc[0].date()} 起")
    navi = nav.copy()
    navi["ma60"] = navi["nav"].rolling(60, min_periods=10).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=navi["date"], y=navi["nav"], mode="lines", name="单位净值",
        line=dict(color=theme()["info"], width=2),
        fill="tozeroy", fillcolor="rgba(91,127,166,0.10)",
        hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=navi["date"], y=navi["ma60"], mode="lines", name="60日均线",
        line=dict(color=theme()["accent"], width=1.4, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>MA60 %{y:.4f}<extra></extra>",
    ))
    style_fig(fig, "", 400, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="单位净值")
    fig.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.06)
    time_selector(fig)
    chart(fig)
    st.caption("拖拽下方滑块可缩放区间，也可用图上方的快捷按钮切换时间窗口。")

    # ---------------- 最新特征定位 ----------------
    section("模型此刻「看到」了什么", "最新特征行相对历史分布的位置")
    render_latest_feature_position(res)


# ================================================================= 信号详情
def render_prediction(res) -> None:
    wf = res.wf
    prob = wf.predictions
    frame = res.dataset.frame
    y_true = frame.loc[prob.index, "label"].astype(int)
    fwd = frame.loc[prob.index, "fwd_ret"]
    t = theme()

    section("样本外预测概率", "每个点都是「用过去预测未来」的产物，不含训练集自预测")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prob.index, y=prob.values, mode="lines", name="上涨概率",
        line=dict(color=t["accent"], width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>概率 %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[prob.index.min(), prob.index.max()], y=[0.5, 0.5], mode="lines",
        name="0.5 中性线", line=dict(color=t["neutral"], dash="dot", width=1),
        hoverinfo="skip",
    ))
    fig.add_hline(y=res.config.threshold, line_dash="dash", line_color=t["info"],
                  annotation_text=f"阈值 {res.config.threshold:.2f}",
                  annotation_position="top left")
    style_fig(fig, "", 380, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="上涨概率", range=[0, 1])
    time_selector(fig)
    chart(fig)

    c1, c2 = st.columns(2, gap="small")

    with c1:
        fig = go.Figure(go.Histogram(
            x=prob.values, nbinsx=40, marker_color=t["accent"], opacity=0.85,
            hovertemplate="概率 %{x:.3f}<br>%{y} 次<extra></extra>",
        ))
        fig.add_vline(x=0.5, line_dash="dot", line_color=t["neutral"])
        style_fig(fig, "概率分布越集中，模型越「不敢表态」", 320)
        fig.update_xaxes(title="上涨概率")
        fig.update_yaxes(title="样本数", rangemode="tozero")
        chart(fig)
        spread = float(prob.std(ddof=1))
        st.caption(
            f"概率标准差 {fnum(spread, 4)}。"
            + ("分布明显张开，模型愿意给出差异化判断。" if spread > 0.05
               else "分布几乎贴着 0.5，说明模型的判断与常数无异——这与弱信号是一致的。")
        )

    with c2:
        grp, _ = calibration_table(prob, y_true, n_bins=10)
        if grp.empty:
            note("样本不足，无法计算校准曲线。", "warn")
        else:
            grp = grp.assign(count=grp["n"])
            fig = go.Figure()
            mx = float(max(grp["pred"].max(), grp["obs"].max())) * 1.08
            fig.add_trace(go.Scatter(
                x=[0, mx], y=[0, mx], mode="lines", name="完美校准",
                line=dict(color=t["neutral"], dash="dash", width=1.2), hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=grp["pred"], y=grp["obs"], mode="lines+markers",
                name="实际频率",
                line=dict(color=t["accent"], width=2),
                marker=dict(size=8, color=t["accent"]),
                customdata=grp["n"].to_numpy(),
                hovertemplate="预测 %{x:.3f}<br>实际 %{y:.3f}<br>样本 %{customdata}<extra></extra>",
            ))
            style_fig(fig, "校准曲线：模型说的 60%，是否真有 60% 会涨？", 320, legend=True)
            fig.update_xaxes(title="分档平均预测概率")
            fig.update_yaxes(title="分档实际上涨频率")
            chart(fig)
            _, ece = calibration_table(prob, y_true, 10)
            st.caption(
                f"期望校准误差 ECE = {fnum(ece, 4)}。"
                + ("概率读数基本可信。" if ece < 0.05
                   else "偏离对角线较多，概率只能当排序看，不能直接当胜率用。")
            )

    # ---------------- 概率分档的实际表现 ----------------
    section("概率分档 → 实际涨跌", "若模型有效，各档上涨率应单调上升")
    n_bins = st.segmented_control("分档数", [5, 10], default=10, key="fs_pred_bins")
    n_bins = int(n_bins or 10)
    try:
        bins = pd.qcut(prob, n_bins, labels=False, duplicates="drop")
        grp = (pd.DataFrame({"bin": bins, "y": y_true.to_numpy(), "r": fwd.to_numpy()})
                 .groupby("bin", observed=True)
                 .agg(up_rate=("y", "mean"), mean_ret=("r", "mean"), n=("y", "size")))
        grp.index = [f"Q{i + 1}" for i in range(len(grp))]
        base = float(y_true.mean())

        c1, c2 = st.columns(2, gap="small")
        with c1:
            fig = go.Figure(go.Bar(
                x=grp.index, y=grp["up_rate"],
                marker_color=[t["up"] if v >= base else t["down"] for v in grp["up_rate"]],
                text=[f"{v:.1%}<br>n={int(n)}"
                      for v, n in zip(grp["up_rate"], grp["n"], strict=True)],
                textposition="outside",
                hovertemplate="%{x}<br>上涨率 %{y:.1%}<extra></extra>",
            ))
            fig.add_hline(y=base, line_dash="dash", line_color=t["neutral"],
                          annotation_text=f"整体上涨率 {base:.1%}")
            style_fig(fig, "各档实际上涨频率", 340)
            fig.update_yaxes(title="上涨频率", tickformat=".0%", rangemode="tozero")
            chart(fig)

        with c2:
            fig = go.Figure(go.Bar(
                x=grp.index, y=grp["mean_ret"],
                marker_color=[t["up"] if v >= 0 else t["down"] for v in grp["mean_ret"]],
                text=[fpct(v, 2, sign=True) for v in grp["mean_ret"]],
                textposition="outside",
                hovertemplate="%{x}<br>平均收益 %{y:.3%}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
            style_fig(fig, "各档平均可交易收益", 340)
            fig.update_yaxes(title="平均未来收益", tickformat=".2%")
            chart(fig)

        mono = float(pd.Series(grp["up_rate"].to_numpy()).corr(
            pd.Series(np.arange(len(grp))), method="spearman"))
        note(
            f"分档上涨率与档位序号的秩相关为 <b>{fnum(mono, 3)}</b>。"
            + ("越接近 1，说明概率越高、真实上涨频率也越高，模型的排序能力成立。"
               if mono > 0.7 else
               "排序关系不单调——高概率档并没有对应更高的真实上涨频率，"
               "说明模型输出的「概率」排序能力有限。"),
            "ok" if mono > 0.7 else "warn",
        )
    except ValueError as err:
        st.info(f"无法分档：{err}")

    # ---------------- 滚动 IC ----------------
    section("滚动 IC：信号是否在衰减", "逐 60 日窗口的 Spearman 秩相关")
    c1, c2 = st.columns([3, 1], gap="small")
    with c2:
        win = st.slider("窗口（交易日）", 20, 180, 60, 10, key="fs_ic_win")
    roll = rolling_spearman(prob, fwd, window=int(win))
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll, mode="lines", name="滚动 IC",
            line=dict(color=t["accent"], width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>IC %{y:.3f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
        style_fig(fig, f"滚动 {int(win)} 日 IC（正 = 概率与未来收益同向）", 340)
        fig.update_yaxes(title="Spearman IC")
        chart(fig)

    valid = roll.dropna()
    if len(valid) > 5:
        pos_ratio = float((valid > 0).mean())
        note(
            f"滚动窗口中有 <b>{fpct(pos_ratio, 0)}</b> 的时间 IC 为正，"
            f"均值 {fnum(valid.mean(), 4)}，波动 {fnum(valid.std(ddof=1), 4)}。"
            + ("IC 长期在 0 附近来回穿越——这是「信号极弱」的典型形态，"
               "意味着任何单次交易的结果都主要由噪声决定。"
               if abs(valid.mean()) < 0.03 else
               "IC 存在一定正向偏移，但仍需扣除交易成本后再判断是否有净收益。"),
            "info",
        )

    # ---------------- 仓位时间轴 ----------------
    section("信号触发的持仓区间", "把「什么时候在场」摊开看")
    daily = res.backtest.daily
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["position"], mode="lines", name="仓位",
        line=dict(color=t["up"], width=1.2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(214,69,69,0.22)",
        hovertemplate="%{x|%Y-%m-%d}<br>仓位 %{y:.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=daily.index, y=(daily["prob"] - 0.5) * 2, mode="lines", name="概率（映射到 ±1）",
        line=dict(color=t["info"], width=1.1, dash="dot"), opacity=0.75,
        hovertemplate="%{x|%Y-%m-%d}<br>偏移 %{y:.2f}<extra></extra>",
    ))
    style_fig(fig, "红色的「在场」区间与概率偏移是否同步？", 300, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="仓位 / 概率偏移", range=[-1.15, 1.35])
    time_selector(fig)
    chart(fig)

    ic = information_coefficient(prob, fwd)
    kpi_row([
        dict(label="信息系数 IC（全样本）", value=fnum(ic, 4),
             delta="不依赖阈值的诚实指标",
             tone=tone_of(ic, zero_is_flat=False),
             sub="|IC| < 0.03 基本等同于噪音"),
        dict(label="持仓时间占比", value=fpct(res.backtest.exposure, 1),
             tone="flat", sub="阈值越高，在场时间越短"),
        dict(label="样本外交易日数", value=fint(len(prob)),
             tone="flat", sub=f"占全样本 {fpct(len(prob) / max(len(frame), 1), 0)}"),
        dict(label="概率标准差", value=fnum(float(prob.std(ddof=1)), 4),
             tone="flat", sub="越大说明模型越愿意表态"),
    ], cols=4)


# ================================================================= 模型评估
def render_model(res) -> None:
    wf = res.wf
    frame = res.dataset.frame
    prob = wf.predictions
    y_true = frame.loc[prob.index, "label"].astype(int)
    t = theme()
    folds = pd.DataFrame(wf.fold_metrics)

    section("滚动前向窗口", "灰色 = 训练区间，红色 = 该折的测试区间（永远在训练之后）")
    if not folds.empty:
        fig = go.Figure()
        for i, row in folds.iterrows():
            label = f"折 {int(row['fold'])}"
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(row["train_end"]) - pd.Timedelta(
                    days=int(row["n_train"]) * 365 / 244),
                   pd.Timestamp(row["train_end"])],
                y=[label, label], mode="lines",
                line=dict(color=t["bad"], width=13),
                name="训练窗口", showlegend=(i == 0),
                hovertemplate=f"{label}<br>训练至 {row['train_end']}"
                              f"<br>{int(row['n_train'])} 行<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(row["test_start"]), pd.Timestamp(row["test_end"])],
                y=[label, label], mode="lines",
                line=dict(color=t["up"], width=13),
                name="测试窗口", showlegend=(i == 0),
                hovertemplate=f"{label}<br>{row['test_start']} ~ {row['test_end']}"
                              f"<br>{int(row['n_test'])} 行<extra></extra>",
            ))
        style_fig(fig, "训练窗口随折数推进而变长，测试段依次向前滚动", 300, legend=True)
        fig.update_yaxes(title="", autorange="reversed")
        fig.update_xaxes(title="时间")
        chart(fig)

    c1, c2 = st.columns([1.15, 1], gap="small")

    with c1:
        if not folds.empty:
            aucs = folds["auc"].to_numpy(dtype=float)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=folds["fold"].astype(str), y=aucs, mode="lines+markers",
                name="AUC", line=dict(color=t["neutral"], width=1.2, dash="dot"),
                marker=dict(size=11,
                            color=[t["up"] if a >= 0.5 else t["down"] for a in aucs]),
                text=[f"{a:.3f}" for a in aucs], textposition="top center",
                hovertemplate="折 %{x}<br>AUC %{y:.4f}<extra></extra>",
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color=t["neutral"],
                          annotation_text="0.5 = 无预测能力")
            lo = min(0.46, float(np.nanmin(aucs)) - 0.03)
            hi = max(0.58, float(np.nanmax(aucs)) + 0.05)
            style_fig(fig, "各折 AUC：点越分散，说明结论越不稳", 340)
            fig.update_yaxes(title="AUC", range=[lo, hi])
            chart(fig)

    with c2:
        boot = bootstrap_auc(y_true, prob, seed=res.config.random_state)
        if boot["n"]:
            fig = go.Figure(go.Histogram(
                x=boot["samples"], nbinsx=36, marker_color=t["info"], opacity=0.85,
                hovertemplate="AUC %{x:.3f}<br>%{y} 次<extra></extra>",
            ))
            fig.add_vline(x=0.5, line_dash="dash", line_color=t["neutral"],
                          annotation_text="0.5")
            fig.add_vline(x=res.auc, line_color=t["accent"], line_width=2.5,
                          annotation_text=f"实测 {res.auc:.3f}")
            style_fig(fig, f"Bootstrap AUC 分布（{boot['n']} 次重抽）", 340)
            fig.update_xaxes(title="AUC")
            fig.update_yaxes(title="出现次数", rangemode="tozero")
            chart(fig)
            sig = "显著" if boot["lo"] > 0.5 else "不显著"
            note(
                f"AUC 的 95% 置信区间为 <b>[{fnum(boot['lo'], 3)}, {fnum(boot['hi'], 3)}]</b>，"
                f"重抽中 {fpct(boot['p_le_half'], 1)} 次不高于 0.5。"
                f"结论：<b>{sig}</b>。"
                + ("" if boot["lo"] > 0.5 else
                   " 区间跨越 0.5 意味着「这点预测力可能纯属抽样噪声」——"
                   "这是判断模型价值最有说服力的一张图。"),
                "ok" if boot["lo"] > 0.5 else "warn",
            )
        else:
            note("样本不足，无法做 Bootstrap 置信区间。", "warn")

    if not folds.empty:
        section("各折明细")
        cols = ["fold", "train_end", "test_start", "test_end", "n_train", "n_test",
                "auc", "accuracy", "majority_acc", "acc_edge", "brier"]
        st.dataframe(
            folds[cols],
            width="stretch",
            hide_index=True,
            column_config={
                "fold": st.column_config.NumberColumn("折", format="%d"),
                "train_end": "训练至",
                "test_start": "测试起",
                "test_end": "测试止",
                "n_train": st.column_config.NumberColumn("训练样本", format="%d"),
                "n_test": st.column_config.NumberColumn("测试样本", format="%d"),
                "auc": st.column_config.NumberColumn("AUC", format="%.4f"),
                "accuracy": st.column_config.NumberColumn("准确率", format="%.4f"),
                "majority_acc": st.column_config.NumberColumn("多数类基线", format="%.4f"),
                "acc_edge": st.column_config.NumberColumn("超额", format="%+.4f"),
                "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
            },
        )

    section("判别能力", "ROC 看排序能力，PR 看「看多信号」的命中质量")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        try:
            fpr, tpr, _ = roc_curve(y_true, prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="随机猜测",
                line=dict(color=t["neutral"], dash="dash", width=1.2), hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"AUC = {res.auc:.4f}",
                line=dict(color=t["accent"], width=2.6),
                fill="tozeroy", fillcolor="rgba(201,134,26,0.10)",
                hovertemplate="假正率 %{x:.3f}<br>真正率 %{y:.3f}<extra></extra>",
            ))
            style_fig(fig, "ROC 曲线（越贴近左上角越好）", 360, legend=True)
            fig.update_xaxes(title="假正率（误报）", range=[0, 1])
            fig.update_yaxes(title="真正率（命中）", range=[0, 1])
            chart(fig)
        except ValueError as err:
            st.info(f"无法绘制 ROC：{err}")

    with c2:
        try:
            pre, rec, _ = precision_recall_curve(y_true, prob)
            ap = float(average_precision_score(y_true, prob))
            base = float(y_true.mean())
            fig = go.Figure()
            fig.add_hline(y=base, line_dash="dash", line_color=t["neutral"],
                          annotation_text=f"基准率 {base:.1%}")
            fig.add_trace(go.Scatter(
                x=rec, y=pre, mode="lines", name=f"AP = {ap:.4f}",
                line=dict(color=t["info"], width=2.6),
                fill="tozeroy", fillcolor="rgba(62,124,177,0.10)",
                hovertemplate="召回 %{x:.3f}<br>精确 %{y:.3f}<extra></extra>",
            ))
            style_fig(fig, "PR 曲线（不平衡数据下比 ROC 更敏感）", 360, legend=True)
            fig.update_xaxes(title="召回率", range=[0, 1])
            fig.update_yaxes(title="精确率", range=[0, 1.02])
            chart(fig)
            st.caption(
                f"平均精确率 AP = {fnum(ap, 4)}，上涨基准率 {fpct(base, 1)}。"
                + ("AP 高于基准率，看多信号略优于随机抽样。" if ap > base
                   else "AP 不高于基准率，说明「看多」信号并没有筛出更好的样本。")
            )
        except ValueError as err:
            st.info(f"无法绘制 PR：{err}")

    section("特征重要性", "按贡献占比归一化，可跨模型比较")
    if len(res.importance):
        c1, c2 = st.columns([1.4, 1], gap="small")
        with c1:
            topn = st.slider("显示前 N 个特征", 5, min(30, len(res.importance)), 15,
                             key="fs_topn")
            top = res.importance.head(int(topn))[::-1]
            fig = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation="h",
                marker_color=[t["accent"] if i < 3 else t["info"]
                              for i in range(len(top) - 1, -1, -1)],
                text=[f"{v:.1%}" for v in top.values], textposition="outside",
                hovertemplate="%{y}<br>贡献占比 %{x:.2%}<extra></extra>",
            ))
            style_fig(fig, f"贡献最大的 {int(topn)} 个特征（前 3 名高亮）",
                       max(340, 24 * int(topn)))
            fig.update_xaxes(title="贡献占比", tickformat=".0%")
            chart(fig)

        with c2:
            grp = pd.Series({
                name: float(res.importance.reindex(cols).fillna(0.0).sum())
                for name, cols in FEATURE_GROUPS.items()
            }).sort_values(ascending=False)
            grp = grp[grp > 0]
            fig = go.Figure(go.Bar(
                x=grp.values, y=grp.index, orientation="h",
                marker_color=t["info"],
                text=[f"{v:.1%}" for v in grp.values], textposition="outside",
                hovertemplate="%{y}<br>合计占比 %{x:.2%}<extra></extra>",
            ))
            style_fig(fig, "特征分组的合计贡献", 320)
            fig.update_xaxes(title="合计贡献占比", tickformat=".0%")
            chart(fig)
            dominant = grp.index[0]
            note(
                f"贡献最大的分组是 <b>{dominant}</b>（占 "
                f"{fpct(float(grp.iloc[0]), 1)}）。"
                + ("模型主要依赖价格自身的动量与波动，市场环境信息只是配角。"
                   if dominant.startswith("价格") else
                   "模型较依赖外部市场环境，需注意基准指数取数失败时信号会退化。"),
                "info",
            )
    else:
        st.info("当前模型不支持特征重要性输出。")

    section("混淆矩阵", f"阈值 = {res.config.threshold:.2f}")
    pred = (prob >= res.config.threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    total = max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    c1, c2 = st.columns([1.1, 1], gap="small")
    with c1:
        fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["预测跌", "预测涨"], y=["实际跌", "实际涨"],
            text=[[f"TN<br>{tn}<br>{fpct(tn / total, 1)}",
                   f"FP<br>{fp}<br>{fpct(fp / total, 1)}"],
                  [f"FN<br>{fn}<br>{fpct(fn / total, 1)}",
                   f"TP<br>{tp}<br>{fpct(tp / total, 1)}"]],
            texttemplate="%{text}", textfont=dict(size=12),
            colorscale="Blues", showscale=False,
            hovertemplate="%{y} / %{x}<br>%{z} 次<extra></extra>",
        ))
        style_fig(fig, "", 320)
        chart(fig)
    with c2:
        kpi_row([
            dict(label="精确率 Precision", value=fpct(prec, 1),
                 sub="看多信号里真正上涨的比例"),
            dict(label="召回率 Recall", value=fpct(rec, 1),
                 sub="真实上涨里被抓住的比例"),
            dict(label="F1", value=fnum(f1, 4), sub="精确与召回的调和平均"),
            dict(label="预测看多占比", value=fpct((tp + fp) / total, 1),
                 sub=f"实际上涨占比 {fpct((tp + fn) / total, 1)}"),
        ], cols=2)


# ================================================================= 回测
def render_backtest(res) -> None:
    frame = res.dataset.frame
    prob = res.wf.predictions
    fwd = frame.loc[prob.index, "fwd_ret"]
    t = theme()

    section("交互回测", "以下滑块不会重新训练模型，只重算回测，可实时观察敏感度")
    c1, c2, c3 = st.columns(3, gap="small")
    thr = c1.slider("看多阈值", 0.30, 0.70, 0.50, 0.01, key="bt_thr")
    cost = c2.slider("单边成本（基点）", 0.0, 100.0, float(res.config.cost_bps), 1.0,
                     key="bt_cost")
    lag = c3.slider("额外执行延迟（交易日）", 0, 5, 0, 1, key="bt_lag",
                    help="0 = 信号次日按净值成交（已内含在收益定义中）；调大做压力测试")

    bt = backtest_threshold(prob, fwd, threshold=float(thr), cost_bps=float(cost),
                            execution_lag=int(lag))

    kpi_row([
        dict(label="策略累计收益", value=fpct(bt.strategy_metrics["total_return"], 1),
             tone=tone_of(bt.strategy_metrics["total_return"]),
             sub=f"年化 {fpct(bt.strategy_metrics['annual_return'], 1)}"),
        dict(label="买入持有", value=fpct(bt.benchmark_metrics["total_return"], 1),
             tone=tone_of(bt.benchmark_metrics["total_return"]),
             sub=f"年化 {fpct(bt.benchmark_metrics['annual_return'], 1)}"),
        dict(label="超额收益", value=fpct(bt.excess_return, 1),
             delta="跑赢持有" if bt.excess_return > 0 else "跑输持有",
             tone=tone_of(bt.excess_return, zero_is_flat=False),
             sub="扣费后的净差额"),
        dict(label="持仓日胜率", value=fpct(bt.win_rate_active, 1),
             tone=tone_of(bt.win_rate_active - 0.5, zero_is_flat=False),
             sub="只统计有仓位的日子，比全样本胜率更反映择时"),
        dict(label="最大回撤", value=fpct(bt.strategy_metrics["max_drawdown"], 1),
             tone="down" if bt.strategy_metrics["max_drawdown"] < -0.15 else "flat",
             sub=f"持有买入回撤 {fpct(bt.benchmark_metrics['max_drawdown'], 1)}"),
        dict(label="持仓占比", value=fpct(bt.exposure, 1),
             tone="flat", sub=f"交易 {fint(bt.strategy_metrics['n_trades'])} 次"),
    ], cols=3)

    # ---------------- 净值 + 回撤 ----------------
    eq = pd.DataFrame({
        "策略": (1 + bt.daily["strategy_ret"]).cumprod(),
        "买入持有": (1 + bt.daily["benchmark_ret"]).cumprod(),
    })
    dd = eq["策略"] / eq["策略"].cummax() - 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.07,
                        subplot_titles=("净值曲线（起点归一）", "策略回撤（水下曲线）"))
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq["策略"], name="策略", line=dict(color=t["up"], width=2.2),
        hovertemplate="%{x|%Y-%m-%d}<br>策略 %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq["买入持有"], name="买入持有",
        line=dict(color=t["info"], width=2.2),
        hovertemplate="%{x|%Y-%m-%d}<br>持有 %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, name="回撤", line=dict(color=t["down"], width=1.4),
        fill="tozeroy", fillcolor="rgba(47,158,95,0.20)",
        hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.2%}<extra></extra>",
    ), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    style_fig(fig, "", 540, legend=True)
    fig.update_layout(hovermode="x unified", margin=dict(t=34))
    for ann in fig.layout.annotations[:2]:
        ann.font.size = 12.5
        ann.x = 0
        ann.xanchor = "left"
    chart(fig)

    # ---------------- 成本拆解 ----------------
    section("成本拆解", "毛收益 → 扣掉交易成本 → 净收益")
    gross_ret = float((1 + bt.daily["position"] * bt.daily["benchmark_ret"]).prod() - 1)
    net_ret = float(bt.strategy_metrics["total_return"])
    drag = gross_ret - net_ret
    turnover_sum = float(bt.daily["turnover"].sum())
    cost_sum = float(bt.daily["cost"].sum())

    c1, c2 = st.columns([1.3, 1], gap="small")
    with c1:
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["毛收益（不计成本）", "交易成本拖累", "净收益"],
            y=[gross_ret, -drag, net_ret],
            text=[fpct(gross_ret, 2, sign=True), fpct(-drag, 2, sign=True),
                  fpct(net_ret, 2, sign=True)],
            textposition="outside",
            connector=dict(line=dict(color=t["neutral"], width=1, dash="dot")),
            increasing=dict(marker=dict(color=t["up"])),
            decreasing=dict(marker=dict(color=t["down"])),
            totals=dict(marker=dict(color=t["accent"])),
            hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
        style_fig(fig, "成本吃掉多少收益？", 360)
        fig.update_yaxes(title="累计收益", tickformat=".0%")
        chart(fig)
    with c2:
        cover = (gross_ret / drag) if drag > 0 else float("inf")
        kpi_row([
            dict(label="累计换手（仓位变动量）", value=f"{turnover_sum:.1f}",
                 sub=f"平均每次变动 1.0，共 {fint(bt.strategy_metrics['n_trades'])} 次交易"),
            dict(label="累计成本", value=fpct(-cost_sum, 2, sign=True),
                 tone="down" if cost_sum > 0 else "flat",
                 sub=f"单边 {cost:.0f} 基点 × {turnover_sum:.1f} 换手"),
            dict(label="成本拖累占毛收益", value=fpct(drag / gross_ret, 1) if gross_ret else "—",
                 tone="down" if drag > 0 else "flat",
                 sub="越高说明策略越「手忙脚乱」"),
            dict(label="毛收益对成本的覆盖倍数", value=(f"{cover:.2f}×" if np.isfinite(cover) else "∞"),
                 tone=tone_of(cover - 1 if np.isfinite(cover) else 1, zero_is_flat=False),
                 sub="< 1 表示成本直接吃穿收益"),
        ], cols=2)
        if drag > 0 and abs(drag) > abs(gross_ret) * 0.5:
            note(
                f"成本拖累 <b>{fpct(drag, 2)}</b> 已超过毛收益的一半。"
                "在日频调仓的策略里这是常态：<b>信号越弱、交易越频繁，成本占比越高</b>。"
                "降低成本的可行方向是拉长预测周期、提高阈值减少交易次数。",
                "warn",
            )

    # ---------------- 阈值敏感度 ----------------
    section("阈值敏感度", "只在某一个阈值上亮眼、相邻阈值立刻崩掉，多半是过拟合")
    sw = sweep_thresholds(prob, fwd, cost_bps=float(cost), execution_lag=int(lag))
    if not sw.empty:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sw["threshold"], y=sw["total_return"], mode="lines+markers",
                name="策略累计收益", line=dict(color=t["up"], width=2.4),
                marker=dict(size=7),
                hovertemplate="阈值 %{x:.2f}<br>累计收益 %{y:.2%}<extra></extra>",
            ))
            fig.add_hline(y=float(bt.benchmark_metrics["total_return"]),
                          line_dash="dash", line_color=t["info"],
                          annotation_text="买入持有")
            fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
            fig.add_vline(x=float(thr), line_color=t["accent"], line_width=1.6,
                          annotation_text="当前")
            style_fig(fig, "不同阈值下的策略累计收益", 340)
            fig.update_xaxes(title="看多阈值")
            fig.update_yaxes(title="累计收益", tickformat=".0%")
            chart(fig)

        with c2:
            fig = go.Figure(go.Scatter(
                x=sw["exposure"], y=sw["total_return"],
                mode="markers+lines",
                marker=dict(size=11, color=sw["threshold"],
                            colorscale=[[0, t["info"]], [1, t["accent"]]],
                            showscale=True,
                            colorbar=dict(title="阈值", thickness=12, len=0.7)),
                line=dict(color=t["neutral"], width=1, dash="dot"),
                name="阈值路径",
                text=[f"{v:.2f}" for v in sw["threshold"]],
                hovertemplate="阈值 %{text}<br>持仓占比 %{x:.1%}"
                              "<br>累计收益 %{y:.2%}<extra></extra>",
            ))
            fig.add_hline(y=float(bt.benchmark_metrics["total_return"]),
                          line_dash="dash", line_color=t["info"])
            style_fig(fig, "阈值前沿：持仓越久收益越高，还是越少越稳？", 340)
            fig.update_xaxes(title="持仓时间占比", tickformat=".0%")
            fig.update_yaxes(title="累计收益", tickformat=".0%")
            chart(fig)

        st.dataframe(
            sw.assign(持仓占比=sw["exposure"] * 100).drop(columns=["exposure"]),
            width="stretch", hide_index=True,
            column_config={
                "threshold": st.column_config.NumberColumn("阈值", format="%.2f"),
                "持仓占比": st.column_config.NumberColumn("持仓占比%", format="%.1f"),
                "total_return": st.column_config.NumberColumn("累计收益", format="%.4f"),
                "annual_return": st.column_config.NumberColumn("年化收益", format="%.4f"),
                "sharpe": st.column_config.NumberColumn("夏普", format="%.2f"),
                "max_drawdown": st.column_config.NumberColumn("最大回撤", format="%.4f"),
                "n_trades": st.column_config.NumberColumn("交易次数", format="%d"),
                "excess_vs_hold": st.column_config.NumberColumn("超额", format="%+.4f"),
            },
        )

    # ---------------- 月度 / 年度 ----------------
    section("分时间段表现", "好策略不应只在某一段行情里有效")
    c1, c2 = st.columns(2, gap="small")
    with c1:
        monthly = (1 + bt.daily["strategy_ret"]).resample("ME").prod() - 1
        if len(monthly) > 3:
            piv = pd.DataFrame({"y": monthly.index.year, "m": monthly.index.month,
                                "r": monthly.values}).pivot(index="y", columns="m",
                                                            values="r")
            fig = go.Figure(go.Heatmap(
                z=piv.values, x=[f"{c}月" for c in piv.columns],
                y=piv.index.astype(str),
                colorscale=[[0, t["down"]], [0.5, t["mid"]], [1, t["up"]]],
                zmid=0, colorbar=dict(title="收益", thickness=12),
                hovertemplate="%{y}年%{x}<br>%{z:.2%}<extra></extra>",
            ))
            style_fig(fig, "月度收益热力图（红涨绿跌）", 340)
            chart(fig)
        else:
            st.info("样本区间过短，无法做月度聚合。")

    with c2:
        yr = pd.DataFrame({
            "策略": (1 + bt.daily["strategy_ret"]).resample("YE").prod() - 1,
            "买入持有": (1 + bt.daily["benchmark_ret"]).resample("YE").prod() - 1,
        })
        yr.index = [str(i.year) for i in yr.index]
        if len(yr) >= 1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=yr.index, y=yr["策略"], name="策略", marker_color=t["up"],
                text=[fpct(v, 1, sign=True) for v in yr["策略"]],
                textposition="outside",
                hovertemplate="%{x} 策略 %{y:.2%}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=yr.index, y=yr["买入持有"], name="买入持有", marker_color=t["info"],
                text=[fpct(v, 1, sign=True) for v in yr["买入持有"]],
                textposition="outside",
                hovertemplate="%{x} 持有 %{y:.2%}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
            style_fig(fig, "逐年收益对比", 340, legend=True)
            fig.update_layout(barmode="group")
            fig.update_yaxes(title="年度收益", tickformat=".0%")
            chart(fig)
            win_years = int((yr["策略"] > yr["买入持有"]).sum())
            st.caption(
                f"{len(yr)} 个年度中，策略有 {win_years} 年跑赢买入持有。"
                "一年两年的时间窗口很容易被单一行情主导，别急着下结论。"
            )

    # ---------------- 回撤持续期 ----------------
    section("回撤的「形状」", "同样是 -15%，三个月阴跌和三天插水的持有体验完全不同")
    eps = drawdown_episodes(eq["策略"], top=5)
    if not eps.empty:
        eps_disp = eps.copy()
        eps_disp["最大回撤"] = eps_disp["最大回撤"].map(lambda v: fpct(v, 2))
        eps_disp["开始"] = eps_disp["开始"].map(lambda v: str(pd.Timestamp(v).date()))
        eps_disp["谷底"] = eps_disp["谷底"].map(lambda v: str(pd.Timestamp(v).date()))
        eps_disp["恢复"] = eps_disp["恢复"].map(
            lambda v: "尚未恢复" if v is None or pd.isna(v) else str(pd.Timestamp(v).date()))
        st.dataframe(eps_disp, width="stretch", hide_index=True)
        longest = eps.loc[eps["持续(交易日)"].idxmax()]
        end_raw = longest["恢复"]
        end_txt = ("尚未恢复" if pd.isna(end_raw)
                   else str(pd.Timestamp(end_raw).date()))
        note(
            f"持续最久的一段回撤：从 <b>{pd.Timestamp(longest['开始']).date()}</b> 开始，"
            f"{'至今尚未恢复' if pd.isna(end_raw) else f'{end_txt} 才收复'}，"
            f"最长水下 <b>{int(longest['持续(交易日)'])}</b> 个交易日，"
            f"期间最深 <b>{fpct(float(longest['最大回撤']), 2)}</b>。"
            "如果你的持有耐心撑不过这一段，那么再漂亮的年化收益也拿不到手。",
            "info",
        )
    else:
        st.info("样本区间内没有明显回撤。")


# ================================================================= 数据探索
def render_data(res) -> None:
    nav = res.nav
    frame = res.dataset.frame
    t = theme()

    section("收益分布与波动", "净值序列本身的统计特征")
    df = nav.copy()
    df["日收益率"] = df["nav"].pct_change()
    ret = df["日收益率"].dropna()

    c1, c2, c3, c4 = st.columns(4, gap="small")
    ann_ret = float((nav["nav"].iloc[-1] / nav["nav"].iloc[0]) ** (244 / max(len(nav), 1)) - 1)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(244))
    skew = float(ret.skew())
    kurt = float(ret.kurtosis())
    with c1:
        kpi("区间年化收益", fpct(ann_ret, 1), tone=tone_of(ann_ret), sub="按 244 交易日折算")
    with c2:
        kpi("年化波动率", fpct(ann_vol, 1), sub=f"日波动 {fpct(ret.std(ddof=1), 2)}")
    with c3:
        kpi("偏度", fnum(skew, 2), tone="flat", sub="负偏 = 极端下跌更常见")
    with c4:
        kpi("峰度", fnum(kurt, 2), tone="flat", sub="远大于 0 说明厚尾")

    c1, c2 = st.columns(2, gap="small")
    with c1:
        fig = go.Figure(go.Histogram(
            x=ret, nbinsx=80, marker_color=t["info"], opacity=0.85,
            hovertemplate="%{x:.2%}<br>%{y} 天<extra></extra>",
        ))
        fig.add_vline(x=0, line_color=t["neutral"], line_width=1)
        style_fig(fig, "日收益率分布：是否对称、尾部有多厚", 330)
        fig.update_xaxes(title="日收益率", tickformat=".1%")
        fig.update_yaxes(title="天数", rangemode="tozero")
        chart(fig)
    with c2:
        roll = df.set_index("date")["日收益率"].rolling(60).std() * np.sqrt(244)
        fig = go.Figure(go.Scatter(
            x=roll.index, y=roll, mode="lines", line=dict(color=t["accent"], width=1.6),
            fill="tozeroy", fillcolor="rgba(201,134,26,0.14)",
            hovertemplate="%{x|%Y-%m-%d}<br>年化波动 %{y:.1%}<extra></extra>",
        ))
        style_fig(fig, "滚动 60 日年化波动率：风险并不恒定", 330)
        fig.update_yaxes(title="年化波动率", tickformat=".0%")
        time_selector(fig)
        chart(fig)

    section("基金 vs 基准", "基准由建模区间内的日收益重构，仅覆盖样本外区间")
    if "idx_ret_1" in frame.columns:
        fund_lvl = frame["nav"] / frame["nav"].iloc[0]
        bench_lvl = (1 + frame["idx_ret_1"].fillna(0.0)).cumprod()
        bench_lvl = bench_lvl / bench_lvl.iloc[0]
        rel = fund_lvl / bench_lvl
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.64, 0.36],
                            vertical_spacing=0.08,
                            subplot_titles=("净值归一对比", "相对强弱（基金 / 基准）"))
        fig.add_trace(go.Scatter(
            x=fund_lvl.index, y=fund_lvl, name="基金", line=dict(color=t["up"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>基金 %{y:.3f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bench_lvl.index, y=bench_lvl, name="基准指数",
            line=dict(color=t["info"], width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>基准 %{y:.3f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=rel.index, y=rel, name="相对强弱", line=dict(color=t["accent"], width=1.6),
            fill="tozeroy", fillcolor="rgba(201,134,26,0.12)",
            hovertemplate="%{x|%Y-%m-%d}<br>比值 %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=1.0, line_dash="dot", line_color=t["neutral"], row=2, col=1)
        style_fig(fig, "", 460, legend=True)
        fig.update_layout(hovermode="x unified", margin=dict(t=34))
        for ann in fig.layout.annotations[:2]:
            ann.font.size = 12.5
            ann.x = 0
            ann.xanchor = "left"
        chart(fig)
        excess = float(fund_lvl.iloc[-1] - bench_lvl.iloc[-1])
        note(
            f"样本外区间内，基金累计 {fpct(float(fund_lvl.iloc[-1] - 1), 1)}、"
            f"基准 {fpct(float(bench_lvl.iloc[-1] - 1), 1)}，"
            f"相对基准{'跑赢' if excess > 0 else '跑输'} <b>{fpct(abs(excess), 1)}</b>。"
            "先看清这个差额，再去评价模型——"
            "<b>很多时候策略赚钱只是因为它恰好在一只上涨的基金上做多</b>。",
            "ok" if excess > 0 else "info",
        )
    else:
        st.info("本次分析未取得基准指数（取数失败或已跳过市场特征），无法做基准对比。")

    section("特征与未来收益的关系", "单变量看，特征到底有没有区分力")
    c1, c2 = st.columns([1.2, 2], gap="small")
    with c1:
        pick = st.selectbox("选择特征", res.feature_names, key="fs_feat")
        nb = st.segmented_control("分箱数", [5, 10], default=5, key="fs_feat_bins")
        nb = int(nb or 5)
    with c2:
        sub = frame[[pick, "fwd_ret"]].dropna()
        try:
            sub = sub.assign(bin=pd.qcut(sub[pick], nb, labels=False, duplicates="drop"))
            g = (sub.groupby("bin", observed=True)
                    .agg(ret=("fwd_ret", "mean"), n=("fwd_ret", "size")))
            labels = [f"Q{i + 1}" for i in range(len(g))]
            fig = go.Figure(go.Bar(
                x=labels, y=g["ret"],
                marker_color=[t["up"] if v >= 0 else t["down"] for v in g["ret"]],
                text=[f"{fpct(v, 2, sign=True)}<br>n={int(n)}"
                      for v, n in zip(g["ret"], g["n"], strict=True)],
                textposition="outside",
                hovertemplate="%{x}<br>平均未来收益 %{y:.3%}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
            fig.add_hline(y=float(sub["fwd_ret"].mean()), line_dash="dash",
                          line_color=t["neutral"], annotation_text="整体均值")
            style_fig(fig, f"{pick} 分 {nb} 档后的平均未来收益", 330)
            fig.update_yaxes(title="平均未来收益", tickformat=".2%")
            chart(fig)
            spread = float(g["ret"].max() - g["ret"].min())
            st.caption(
                f"最高档与最低档的平均收益相差 {fpct(spread, 2)}。"
                "这个差距越大，该特征作为单变量的区分力越强——"
                "但要注意：多特征模型里，单变量强的特征未必最终贡献大。"
            )
        except ValueError as err:
            st.info(f"无法分箱：{err}")

    section("特征相关性", "高度相关的特征会让树模型的重要性被稀释")
    X = frame[res.feature_names]
    corr = X.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="相关系数", thickness=12),
        hovertemplate="%{y} ↔ %{x}<br>相关 %{z:.2f}<extra></extra>",
    ))
    style_fig(fig, "", 620)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    chart(fig)

    mask = ~np.eye(len(corr), dtype=bool)
    pairs = corr.where(mask).abs().stack()
    if len(pairs):
        pairs = pairs.sort_values(ascending=False)
        top = pairs.head(3)
        lines = "　·　".join(f"{a} ↔ {b}（{fnum(v, 2)}）" for (a, b), v in top.items())
        note(
            f"相关性最高（取绝对值）的三对特征：{lines}。"
            "如果一对特征相关系数超过 0.9，可以考虑只保留其中一个，"
            "或者用它们的差值构造新特征。",
            "info",
        )


# ================================================================= 数据与导出
def render_raw(res) -> None:
    frame = res.dataset.frame

    section("特征与标签数据集", "可直接下载用于自己的建模实验")
    note(
        "列 <code>fwd_ret</code> 已经包含执行延迟（以 T+1 日净值为买入价），"
        "<code>label</code> 是它的符号。<b>不要</b>用 <code>nav</code> 或 "
        "<code>fwd_ret</code> 之外的信息去拟合——更不要把它们放进特征集，"
        "那是最典型的未来函数。",
        "warn",
    )
    st.dataframe(
        frame.tail(300),
        width="stretch",
        column_config={
            "date": st.column_config.DatetimeColumn("日期", format="YYYY-MM-DD"),
        },
    )

    c1, c2, c3 = st.columns(3, gap="small")
    c1.download_button(
        "⬇️ 完整数据集 (CSV)",
        data=frame.to_csv().encode("utf-8-sig"),
        file_name=f"fund_signal_{res.config.fund_code}_dataset.csv",
        mime="text/csv", width="stretch",
    )
    wf_tbl = res.wf.predictions.to_frame("prob").join(frame[["label", "fwd_ret", "nav"]])
    c2.download_button(
        "⬇️ 样本外预测 (CSV)",
        data=wf_tbl.to_csv().encode("utf-8-sig"),
        file_name=f"fund_signal_{res.config.fund_code}_oos.csv",
        mime="text/csv", width="stretch",
    )
    c3.download_button(
        "⬇️ 各折指标 (CSV)",
        data=pd.DataFrame(res.wf.fold_metrics).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"fund_signal_{res.config.fund_code}_folds.csv",
        mime="text/csv", width="stretch",
    )

    section("本次运行的参数")
    st.json(res.config.to_dict())

    section("方法论速览", "为什么这些数字可以被信任（或者说，为什么不能）")
    with st.expander("① 为什么必须用滚动前向验证，而不用普通交叉验证", expanded=False):
        st.markdown(
            "金融时间序列有强自相关。随机划分训练 / 测试集，会把「未来」的样本"
            "拿去训练、再用「过去」的样本测试，AUC 轻松做到 0.7+，但实盘一文不值。\n\n"
            "本项目的做法：预留前 `初始训练集占比` 的样本只用于训练，"
            "把剩余样本等分成 `折数` 段，对每一段都只用它**之前**的全部数据训练。"
            "所以预测序列是「逐段实盘模拟」的产物。"
        )
    with st.expander("② 为什么买入价是 T+1 日净值", expanded=False):
        st.markdown(
            "公募基金 T 日净值在当晚才公布。投资者在 T+1 日 15:00 前下单，"
            "按 **T+1 日净值**成交。\n\n"
            "所以 T 日的信号只能赚到 `nav[T+1] → nav[T+1+horizon]` 这一段。"
            "如果用 `nav[T]` 当买入价，等于假设「看完 T 日净值还能按 T 日净值成交」——"
            "凭空多赚一天，这是开源量化项目里最常见的回测虚高来源。"
        )
    with st.expander("③ 为什么一定要看「多数类基线」", expanded=False):
        st.markdown(
            "如果一只基金历史上 55% 的交易日是上涨的，那么「永远猜涨」这个"
            "什么都不做的策略，准确率就有 55%。\n\n"
            "模型准确率 54% 听起来不错，实际上**还不如抛硬币**。"
            "所以本项目同时输出 `accuracy`、`majority_acc` 与两者的差 `acc_edge`——"
            "只有 `acc_edge` 才是有意义的量。"
        )
    with st.expander("④ 为什么准确率略高于 50% 依然可能亏钱", expanded=False):
        st.markdown(
            "三个原因叠加：\n\n"
            "1. **成本**：每次调仓都要付钱。日频调仓、单边 15 基点，"
            "一年下来成本轻而易举超过 5%。\n"
            "2. **不对称**：猜对时赚 1%，猜错时亏 2%，那么 53% 的胜率照样亏。\n"
            "3. **波动**：胜率只描述频率，不描述幅度。\n\n"
            "这就是为什么本项目把「成本拆解」单独做了一张图——"
            "**毛收益看起来很美，扣掉成本才是你真正拿到的东西**。"
        )
    with st.expander("⑤ 为什么 AUC 需要置信区间", expanded=False):
        st.markdown(
            "AUC = 0.53 这个数字本身没有意义，因为它是**一个样本上的一次抽样**。"
            "换一段历史，它可能变成 0.47。\n\n"
            "本项目用 Bootstrap（有放回重抽 400 次）给出 AUC 的 95% 区间。"
            "**如果区间跨越 0.5，就不能声称模型有预测能力**——"
            "哪怕点估计是 0.53。"
        )
    with st.expander("⑥ 这个项目不能做什么", expanded=False):
        st.markdown(
            "- 不能预测净值点位，只能给出方向概率，且以「历史统计规律会延续」为前提；\n"
            "- 不能处理基金持仓风格漂移、基金经理更换、限购等结构性变化；\n"
            "- 不能替代资产配置决策。择时只是投资里很小的一块。\n\n"
            "一句话：**这是一个教学与研究方法论工具，不是一个赚钱工具。**"
        )


# ================================================================= 主流程
def main() -> None:
    st.set_page_config(
        page_title="fund-signal · 基金量化信号工作台",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    params = render_sidebar()
    run_clicked = params.pop("run")
    refresh_clicked = params.pop("refresh")

    if run_clicked or refresh_clicked:
        p = _analysis_params(params, use_cache=not refresh_clicked)
        try:
            result = obtain_result(p)
            st.session_state["fs_result"] = result
            st.session_state["fs_active"] = p
            if refresh_clicked:
                st.toast("已绕过缓存，重新拉取数据", icon="🔄")
        except ImportError as err:
            st.error(f"依赖缺失：\n\n```\n{err}\n```")
            st.stop()
        except Exception as err:
            st.error(f"分析失败：{err}")
            st.info(
                "常见原因：\n"
                "- 基金代码有误（应为 6 位数字）\n"
                "- 该基金历史过短，样本不足以做时序验证（需要 ≥ 350 行）\n"
                "- 网络不通导致取数失败（可把「数据来源」切为 `local` 试试）\n"
                "- 离线样例仅覆盖 000001 / 161725 / 320007"
            )
            st.stop()

    result = st.session_state.get("fs_result")
    if result is None:
        render_welcome()
        return

    tabs = st.tabs(["📊 总览", "🎯 信号详情", "🧪 模型评估", "💰 回测",
                    "🔍 数据探索", "📋 数据与导出"])
    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_prediction(result)
    with tabs[2]:
        render_model(result)
    with tabs[3]:
        render_backtest(result)
    with tabs[4]:
        render_data(result)
    with tabs[5]:
        render_raw(result)

    st.divider()
    st.caption(
        "**免责声明**　本项目为量化研究方法论演示，所有输出均为基于历史数据的"
        "统计结果，不构成投资建议。基金净值短期走势接近随机，回测表现不代表未来收益。"
        "投资有风险，决策需谨慎。"
    )


if __name__ == "__main__":
    main()
else:  # Streamlit 以 __main__ 执行脚本，这里做双保险
    main()
