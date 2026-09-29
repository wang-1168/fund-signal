# -*- coding: utf-8 -*-
"""fund-signal · Streamlit 交互工作台。

启动::

    streamlit run app.py

这一版相比最初的骨架做了五件事
------------------------------
1. **视觉**：自绘 CSS 主题 + **粒子背景**（canvas 粒子网络，可开关）+
   玻璃拟态卡片 + 渐变流光；深浅色自适应；图表标题写「结论」而不是「图表类型」。
2. **密度**：总览页多张 KPI + 深度诊断评分卡 + 最新特征分位定位，
   一屏之内把「模型行不行、为什么」交代清楚。
3. **深度**：滚动 Spearman IC、概率校准曲线与 ECE、Bootstrap AUC 置信区间、
   成本瀑布拆解、阈值前沿散点、回撤持续期、特征分组贡献、滚动前向窗口图。
4. **动态**：真实进度条、参数预设、免重训的回测滑块、盘中估值自动刷新、
   Plotly 区间选择器 / 滚轮缩放 / 交互工具条。
5. **诚实**：新增「精度审计」与「风控与仓位」两页。
   前者把「准确率能做到多少、高准确率是怎么骗出来的」用可复现的实验摊开讲
   （AUC→准确率理论换算、过拟合/欠拟合复杂度地形、泄漏审计、置信度分层）；
   后者把校准后的概率映射成仓位，并给出压力测试。

配色遵循中国市场惯例：**红涨绿跌**（与欧美相反）。
本工具输出的是统计信号，不构成投资建议。
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from fund_signal import audit
from fund_signal import data as dt
from fund_signal import events as ev
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
    # ---- 粒子 / 玻璃拟态层 ----
    "bg": "#FFFFFF",  # 页面底色（粒子画布会用它打底）
    "sidebar": "#F7F9FC",
    "glass": "rgba(255,255,255,0.72)",
    "glass_hi": "rgba(255,255,255,0.92)",
    "sheen": "rgba(201,134,26,0.10)",
    "p_dot": "rgba(62,124,177,0.55)",  # 粒子点
    "p_line": "rgba(62,124,177,0.16)",  # 粒子连线
    "p_glow1": "rgba(201,134,26,0.10)",  # 光斑一
    "p_glow2": "rgba(62,124,177,0.10)",  # 光斑二
    "p_bg": "#FBFCFE",
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
    "bg": "#0D1117",
    "sidebar": "#11161D",
    "glass": "rgba(21,26,34,0.68)",
    "glass_hi": "rgba(28,35,45,0.88)",
    "sheen": "rgba(240,179,76,0.12)",
    "p_dot": "rgba(111,160,204,0.60)",
    "p_line": "rgba(111,160,204,0.18)",
    "p_glow1": "rgba(240,179,76,0.10)",
    "p_glow2": "rgba(111,160,204,0.12)",
    "p_bg": "#0D1117",
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
  --bg:@bg@; --glass:@glass@; --glass-hi:@glass_hi@; --sheen:@sheen@;
  --sidebar:@sidebar@;
}

/* ---------- 让粒子画布透出来：把 Streamlit 自身的底色全部透明化 ---------- */
html{background:var(--bg)!important}
body{background:transparent!important}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent!important}
[data-testid="stSidebar"]{background:var(--sidebar)!important;
  border-right:1px solid var(--border)}
[data-testid="stSidebar"] .block-container,[data-testid="stSidebar"]>div{
  background:transparent!important}

/* 收紧默认留白，换取信息密度 */
.block-container{padding-top:2.2rem;padding-bottom:3.2rem;max-width:1600px}

/* ---------- 顶部信息条（含粒子画布容器） ---------- */
.fs-hero{position:relative;overflow:hidden;display:flex;flex-wrap:wrap;gap:14px;
  align-items:center;justify-content:space-between;
  padding:20px 24px;border-radius:18px;border:1px solid var(--border);
  background:transparent;margin:2px 0 16px;
  box-shadow:0 10px 34px rgba(20,30,50,.07)}
/* 半透明的渐变底 —— 让背后的粒子网络透出来 */
.fs-hero::before{content:"";position:absolute;inset:0;background:@hero@;opacity:.80}
.fs-hero::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(120% 90% at 100% 0%,var(--sheen) 0%,transparent 62%)}
.fs-hero>div{position:relative;z-index:1}
.fs-hero-title{font-size:23px;font-weight:750;color:var(--text);letter-spacing:.2px;
  line-height:1.3;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.fs-code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px;
  color:var(--accent);border:1px solid var(--accent);border-radius:8px;
  padding:2px 9px;background:var(--glass-hi);letter-spacing:.6px}
.fs-hero-sub{color:var(--muted);font-size:12.5px;margin-top:7px;line-height:1.65;
  max-width:76ch}
.fs-badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;
  position:relative;z-index:2}
.fs-badge{font-size:11.5px;padding:5px 12px;border-radius:999px;
  border:1px solid var(--border);color:var(--muted);background:var(--glass-hi);
  backdrop-filter:blur(10px);white-space:nowrap;letter-spacing:.2px}
.fs-badge.hot{border-color:var(--accent);color:var(--accent)}
.fs-badge.on{border-color:var(--good);color:var(--good)}
.fs-badge.live{border-color:var(--up);color:var(--up)}

/* ---------- KPI 卡片（玻璃拟态） ---------- */
.fs-kpi{position:relative;overflow:hidden;border:1px solid var(--border);border-radius:15px;
  padding:14px 16px;background:var(--glass);backdrop-filter:blur(14px) saturate(1.35);
  -webkit-backdrop-filter:blur(14px) saturate(1.35);height:100%;
  transition:transform .16s cubic-bezier(.2,.7,.3,1),box-shadow .16s ease,
             border-color .16s ease;
  animation:fsRise .42s cubic-bezier(.2,.7,.3,1) both}
.fs-kpi::before{content:"";position:absolute;left:0;right:0;top:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);
  opacity:0;transition:opacity .18s ease}
.fs-kpi:hover{transform:translateY(-3px);
  box-shadow:0 14px 30px rgba(20,30,50,.13),0 0 0 1px var(--sheen);
  border-color:var(--accent)}
.fs-kpi:hover::before{opacity:1}
.fs-kpi-l{font-size:11.5px;color:var(--muted);letter-spacing:.5px;line-height:1.45;
  text-transform:none}
.fs-kpi-v{font-size:25px;font-weight:750;line-height:1.28;margin:4px 0 2px;
  font-variant-numeric:tabular-nums;letter-spacing:-.4px}
.fs-kpi-v.up{color:var(--up)} .fs-kpi-v.down{color:var(--down)}
.fs-kpi-v.flat{color:var(--text)} .fs-kpi-v.accent{color:var(--accent)}
.fs-kpi-v.info{color:var(--info)}
.fs-kpi-d{font-size:12.5px;font-weight:650;line-height:1.45}
.fs-kpi-d.up{color:var(--up)} .fs-kpi-d.down{color:var(--down)}
.fs-kpi-d.flat{color:var(--muted)}
.fs-kpi-s{font-size:11.5px;color:var(--muted);margin-top:6px;line-height:1.55;
  border-top:1px dashed var(--border);padding-top:6px}

/* ---------- 区块标题 ---------- */
.fs-sec{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  margin:24px 0 11px;padding-left:12px;position:relative}
.fs-sec::before{content:"";position:absolute;left:0;top:2px;bottom:2px;width:3px;
  border-radius:3px;background:linear-gradient(180deg,var(--accent),var(--info))}
.fs-sec-t{font-size:17px;font-weight:750;color:var(--text);letter-spacing:.15px}
.fs-sec-d{font-size:12.5px;color:var(--muted)}

/* ---------- 结论条 ---------- */
.fs-note{position:relative;border-radius:11px;padding:12px 16px;font-size:13.5px;
  line-height:1.75;border:1px solid var(--border);border-left:4px solid var(--neutral);
  background:var(--glass);backdrop-filter:blur(10px);color:var(--text);margin:10px 0 6px}
.fs-note.ok{border-left-color:var(--good)}
.fs-note.warn{border-left-color:var(--warn)}
.fs-note.bad{border-left-color:var(--bad)}
.fs-note.info{border-left-color:var(--info)}
.fs-note code{background:rgba(127,127,127,.14);padding:1px 6px;border-radius:5px;
  font-size:12.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.fs-note b{color:var(--accent)}
.fs-note .hi{color:var(--accent);font-weight:700}

/* ---------- 标签页（Streamlit 1.6x 用 data-testid）---------- */
.stTabs [role="tablist"]{gap:4px;border-bottom:1px solid var(--border);
  flex-wrap:wrap;row-gap:2px}
.stTabs [data-testid="stTab"]{height:43px;padding:0 15px;border-radius:11px 11px 0 0;
  font-size:13.8px;font-weight:650;color:var(--muted);transition:all .14s ease}
.stTabs [data-testid="stTab"]:hover{color:var(--text);background:var(--sheen)}
.stTabs [data-testid="stTab"][aria-selected="true"]{color:var(--accent)!important;
  background:var(--glass)}
.stTabs [data-testid="stTab"][aria-selected="true"]::after{content:"";position:absolute;
  left:10px;right:10px;bottom:0;height:2px;border-radius:2px;background:var(--accent)}
.stTabs [data-testid="stTab"] p{font-size:inherit;font-weight:inherit}

/* ---------- 粒子背景层 ----------
   把「含 iframe 的那个元素容器」拽成全屏固定层，画布画在 iframe 内部。
   用 :has() 精确命中，不依赖版本相关的 data-testid；
   z-index:-1 + pointer-events:none 保证它在内容之下、且不拦截任何交互。 */
[data-testid="stElementContainer"]:has(iframe){
  position:fixed!important;left:0!important;top:0!important;
  width:100vw!important;max-width:none!important;height:100vh!important;
  z-index:-1!important;pointer-events:none!important;
  margin:0!important;padding:0!important;overflow:hidden!important;
  background:transparent!important;
}
[data-testid="stElementContainer"]:has(iframe) iframe{
  width:100%!important;height:100%!important;border:0!important;
  display:block!important;
}

/* ---------- 表格 / 滚动条 / 控件 ---------- */
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--border);
  border-radius:11px;overflow:hidden}
[data-testid="stMetric"]{background:var(--glass);border:1px solid var(--border);
  border-radius:13px;padding:11px 14px;backdrop-filter:blur(10px)}
[data-testid="stExpander"]{border:1px solid var(--border);border-radius:11px;
  background:var(--glass);backdrop-filter:blur(8px);overflow:hidden}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:6px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}
.js-plotly-plot{border-radius:12px}

@keyframes fsRise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
@keyframes fsGlow{0%,100%{opacity:.55}50%{opacity:1}}
.fs-pulse{animation:fsGlow 2.6s ease-in-out infinite}
</style>
"""


def inject_css() -> None:
    t = theme()
    css = _CSS
    for key, val in t.items():
        css = css.replace(f"@{key}@", val)
    st.markdown(css, unsafe_allow_html=True)


# ================================================================= 粒子背景
# 全页 canvas 粒子网络。
#
# 实现方式的取舍（踩过的坑，留档）
# --------------------------------
# ① 不能用 ``st.markdown(unsafe_allow_html=True)``：Streamlit 用 React 的
#    ``dangerouslySetInnerHTML`` 渲染，**插进去的 ``<script>`` 不会执行**。
# ② 不该让脚本去改 ``window.parent.document``：组件 iframe 被挂载的时机、
#    ``sandbox`` 属性、以及 ``height=0`` 时干脆不挂载，都会让它时灵时不灵。
# ③ 最终方案：**把组件自己的 iframe 变成全屏背景层**。
#    画布画在 iframe 内部，脚本完全自洽；外层用一条 CSS 把承载 iframe 的
#    ``stElementContainer`` 拽成 ``position:fixed`` 铺满视口。
#    用 ``:has(iframe)`` 精确选中「含 iframe 的那个容器」，不依赖任何
#    版本相关的 ``data-testid``。
#
# 层级：container 用 ``z-index:-1`` + ``pointer-events:none``，
# 配合 ``html`` 上色、``.stApp`` 透明，于是它落在「页面底色之上、所有内容之下」，
# 且不拦截任何鼠标事件。若 CSS 未生效（极旧浏览器不支持 ``:has()``），
# 页面只是少一层背景，其余视觉不受影响。
_PARTICLES_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;padding:0;height:100%;width:100%;overflow:hidden;
  background:__BG__}
canvas{display:block;width:100%;height:100%}
</style></head><body><canvas id="fs-canvas"></canvas>
<script>
(function(){
  var CFG = __CFG__;
  var cv = document.getElementById('fs-canvas');
  var ctx = cv.getContext('2d');
  var DPR = Math.min(window.devicePixelRatio || 1, 2);
  var VW = 0, VH = 0, parts = [], glows = [], phase = 0;

  function resize() {
    VW = window.innerWidth; VH = window.innerHeight;
    cv.width = Math.round(VW * DPR); cv.height = Math.round(VH * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

    var n = Math.round(VW * VH / 24000 * (CFG.density || 1));
    n = Math.max(28, Math.min(96, n));
    parts = [];
    for (var i = 0; i < n; i++) {
      parts.push({
        x: Math.random() * VW, y: Math.random() * VH,
        vx: (Math.random() - .5) * .26, vy: (Math.random() - .5) * .26,
        r: Math.random() * 1.5 + .55
      });
    }
    var R = Math.max(VW, VH);
    glows = [[VW * .12, VH * .08, R * .34],
             [VW * .88, VH * .24, R * .30],
             [VW * .48, VH * .95, R * .34]];
  }

  window.addEventListener('resize', resize);

  var LINK = CFG.link || 140;

  function loop() {
    phase += .0032;

    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = CFG.bg;
    ctx.fillRect(0, 0, VW, VH);

    // --- 光斑：缓慢漂移的低透明度径向渐变，营造呼吸感 ---
    ctx.globalCompositeOperation = 'lighter';
    for (var g = 0; g < glows.length; g++) {
      var gx = glows[g][0] + Math.sin(phase + g * 2.1) * 44;
      var gy = glows[g][1] + Math.cos(phase * .8 + g * 1.7) * 36;
      var rad = glows[g][2];
      var grd = ctx.createRadialGradient(gx, gy, 0, gx, gy, rad);
      grd.addColorStop(0, g === 1 ? CFG.glow2 : CFG.glow1);
      grd.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(gx, gy, rad, 0, 6.2832); ctx.fill();
    }
    ctx.globalCompositeOperation = 'source-over';

    // --- 平移（越界后从另一侧回绕）---
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      p.x += p.vx; p.y += p.vy;
      if (p.x < -20) p.x = VW + 20; else if (p.x > VW + 20) p.x = -20;
      if (p.y < -20) p.y = VH + 20; else if (p.y > VH + 20) p.y = -20;
    }

    // --- 连线：先做轴对齐快速排除，再算平方距离，避免开方 ---
    ctx.lineWidth = 1;
    ctx.strokeStyle = CFG.line;
    ctx.beginPath();
    for (var a = 0; a < parts.length; a++) {
      var pa = parts[a];
      for (var b = a + 1; b < parts.length; b++) {
        var pb = parts[b];
        var ex = pa.x - pb.x, ey = pa.y - pb.y;
        if (ex > LINK || ex < -LINK || ey > LINK || ey < -LINK) continue;
        if (ex * ex + ey * ey > LINK * LINK) continue;
        ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y);
      }
    }
    ctx.stroke();

    // --- 粒子点 ---
    ctx.fillStyle = CFG.dot;
    for (var k = 0; k < parts.length; k++) {
      var q = parts[k];
      ctx.beginPath(); ctx.arc(q.x, q.y, q.r, 0, 6.2832); ctx.fill();
    }

    requestAnimationFrame(loop);
  }

  resize();
  requestAnimationFrame(loop);
})();
</script></body></html>"""


def particle_background(enabled: bool = True) -> None:
    """注入全页粒子背景（占位元素在 DOM 里，真正的绘制发生在被放大的 iframe 内）。"""
    t = theme()
    cfg = {
        "bg": t["p_bg"],
        "dot": t["p_dot"] if enabled else "rgba(0,0,0,0)",
        "line": t["p_line"] if enabled else "rgba(0,0,0,0)",
        "glow1": t["p_glow1"] if enabled else "rgba(0,0,0,0)",
        "glow2": t["p_glow2"] if enabled else "rgba(0,0,0,0)",
        "link": 140,
        "density": 1.0,
    }
    html = _PARTICLES_HTML.replace("__CFG__", json.dumps(cfg)).replace("__BG__", t["p_bg"])
    try:
        embed = getattr(st, "iframe", None)  # Streamlit >= 1.63
        if embed is not None:
            embed(html, height=900)
        else:  # pragma: no cover - 兼容旧版
            components.html(html, height=900, scrolling=False)
    except Exception:  # pragma: no cover - AppTest / 无浏览器环境
        pass


def _flat(html: str) -> str:
    """压掉换行 —— 否则 Streamlit 的 markdown 解析会给 HTML 套上 <p>，布局全乱。"""
    return " ".join(part.strip() for part in html.strip().splitlines())


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)


def _bold(html: str) -> str:
    """把 Markdown 粗体 ``**x**`` 转成 HTML ``<b>x</b>``。

    ``section()`` / ``note()`` 是**裸 HTML 渲染**（``unsafe_allow_html=True`` 下
    Streamlit 不再解析 Markdown），所以正文里写 ``**重点**`` 会原样显示成星号。
    把模块函数产生的说明文字（如 ``events.verdict_text``）直接塞进结论条时尤其
    容易踩这个坑，这里统一兜住。
    """
    return _BOLD_RE.sub(r"<b>\1</b>", html)


def section(title: str, desc: str = "") -> None:
    """区块标题：带左侧主色竖条。"""
    d = f'<span class="fs-sec-d">{_bold(desc)}</span>' if desc else ""
    st.markdown(
        _flat(f'<div class="fs-sec"><span class="fs-sec-t">{title}</span>{d}</div>'),
        unsafe_allow_html=True,
    )


def note(html: str, tone: str = "info") -> None:
    """结论条。tone: info / ok / warn / bad。"""
    st.markdown(_flat(f'<div class="fs-note {tone}">{_bold(html)}</div>'), unsafe_allow_html=True)


def hero(title: str, code: str, sub: str, badges: list[tuple[str, str]]) -> None:
    """顶部基金信息条。badges 为 (文本, 样式) 列表，样式 ∈ {"", "hot", "on"}。"""
    chips = "".join(f'<span class="fs-badge {cls}">{text}</span>' for text, cls in badges)
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


def kpi(
    label: str,
    value: str,
    delta: str | None = None,
    *,
    tone: str = "flat",
    sub: str = "",
    help_text: str = "",
) -> None:
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
        chunk = items[start : start + cols]
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
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11.5)
        ),
        hoverlabel=dict(font_size=12),
        bargap=0.32,
    )
    fig.update_xaxes(
        gridcolor=t["grid"], zeroline=False, showline=False, tickcolor=t["grid"], ticks="outside"
    )
    fig.update_yaxes(
        gridcolor=t["grid"], zeroline=False, showline=False, tickcolor=t["grid"], ticks="outside"
    )
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
def bootstrap_auc(y_true: pd.Series, prob: pd.Series, n_boot: int = 400, seed: int = 42) -> dict:
    """AUC 的 Bootstrap 置信区间。

    AUC = 0.53 到底算不算「有预测力」？单看数字没有意义——必须看它的抽样分布。
    这里对样本做 400 次有放回重抽，得到 AUC 的 95% 区间，
    并用「重抽中 AUC ≤ 0.5 的比例」近似单边 p 值。

    这是判断「模型是否只是运气好」最有说服力的一张图。
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if y.size < 30 or np.unique(y).size < 2:
        return {
            "lo": float("nan"),
            "hi": float("nan"),
            "median": float("nan"),
            "p_le_half": float("nan"),
            "n": 0,
            "samples": np.array([]),
        }

    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, y.size, y.size)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(float(roc_auc_score(yy, p[idx])))
    if not vals:
        return {
            "lo": float("nan"),
            "hi": float("nan"),
            "median": float("nan"),
            "p_le_half": float("nan"),
            "n": 0,
            "samples": np.array([]),
        }

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
    g = (
        df.groupby("bin", observed=True)
        .agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
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
        x = av[i - window + 1 : i + 1]
        y = bv[i - window + 1 : i + 1]
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
    for day, val in dd.items():
        if val < -1e-9:
            if not in_dd:
                in_dd, start, min_dd = True, day, val
                trough = day
            if val < min_dd:
                min_dd, trough = val, day
        elif in_dd:
            episodes.append(
                {
                    "开始": start,
                    "谷底": trough,
                    "恢复": day,
                    "最大回撤": min_dd,
                    "持续(交易日)": int(eq.index.get_loc(day) - eq.index.get_loc(start)),
                }
            )
            in_dd = False
    if in_dd:
        episodes.append(
            {
                "开始": start,
                "谷底": trough,
                "恢复": None,
                "最大回撤": min_dd,
                "持续(交易日)": int(len(eq) - 1 - eq.index.get_loc(start)),
            }
        )
    if not episodes:
        return pd.DataFrame()
    return pd.DataFrame(episodes).sort_values("最大回撤").head(top).reset_index(drop=True)


FEATURE_GROUPS: dict[str, list[str]] = {
    "价格 / 动量": PRICE_FEATURES,
    "市场环境": MARKET_FEATURES,
    "日历效应": CALENDAR_FEATURES,
}


# ================================================================= 侧边栏
PRESETS: dict[str, dict] = {
    "快速体验（逻辑回归 3 折）": dict(horizon=1, n_splits=3, model_type="logistic", init_ratio=0.5),
    "标准（LightGBM 5 折）": dict(horizon=1, n_splits=5, model_type="lightgbm", init_ratio=0.5),
    "稳健（四模型集成 6 折）": dict(horizon=1, n_splits=6, model_type="ensemble", init_ratio=0.6),
    "严格 · 深度（LightGBM 8 折 / 2 日）": dict(
        horizon=2, n_splits=8, model_type="lightgbm", init_ratio=0.6
    ),
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
    sb.pills(
        "快捷选择",
        list(QUICK_CODES),
        key="fs_quick_code",
        on_change=_apply_quick_code,
        label_visibility="collapsed",
    )
    code = sb.text_input(
        "基金代码",
        value="000001",
        key="fs_code",
        help="6 位公募基金代码。离线样例仅含 000001 / 161725 / 320007。",
    )
    bench_label = sb.selectbox(
        "基准指数",
        list(DEFAULT_BENCHMARKS.keys()),
        key="fs_bench",
        help="用于构造市场环境与相对强弱特征",
    )
    data_source = sb.radio(
        "数据来源",
        ["akshare", "local"],
        horizontal=True,
        key="fs_source",
        help="akshare = 联网实时取数（含盘中估值）；local = 仓库自带离线样例，断网也能跑",
    )

    sb.divider()

    # ---------------- ② 预测任务 ----------------
    sb.markdown("**② 预测任务**")
    horizon = sb.slider(
        "预测周期（交易日）", 1, 20, 1, key="fs_horizon", help="预测未来多少个交易日的涨跌方向"
    )
    init_ratio = sb.slider(
        "初始训练集占比",
        0.3,
        0.8,
        0.5,
        0.05,
        key="fs_init_ratio",
        help="滚动前向验证中，前多少比例的历史只用于训练、从不出现在预测集",
    )

    sb.divider()

    # ---------------- ③ 模型 ----------------
    sb.markdown("**③ 模型**")
    model_type = sb.selectbox(
        "模型类型",
        ["lightgbm", "logistic", "ensemble"],
        key="fs_model_type",
        help="lightgbm 梯度提升树；logistic 带标准化的逻辑回归基线；"
        "ensemble = LightGBM + 逻辑回归 + ExtraTrees + GBDT 四模型软投票（更慢更稳）",
    )
    n_splits = sb.slider("滚动验证折数", 2, 10, 5, key="fs_n_splits")
    calibrate = sb.toggle(
        "样本外概率校准",
        value=True,
        key="fs_calibrate",
        help="每折内部用训练集尾部 20% 拟合 isotonic 校准器，让输出概率具备频率含义。"
        "代价是可用训练样本少了两成，收益是概率可以直接当仓位用。",
    )
    embargo = sb.toggle(
        "净化间隔（防标签重叠泄漏）",
        value=True,
        key="fs_embargo",
        help="丢弃紧邻测试段的 horizon 个训练样本，切断「训练标签窗口」与"
        "「测试标签窗口」的重叠。属于严谨性开关，对结果影响很小。",
    )
    seed = sb.number_input("随机种子", value=42, step=1, key="fs_seed")

    sb.divider()

    # ---------------- ④ 成本与实时 ----------------
    sb.markdown("**④ 成本与实时**")
    cost_bps = sb.slider(
        "单边成本（基点）",
        0.0,
        100.0,
        15.0,
        1.0,
        key="fs_cost_bps",
        help="1 基点 = 0.01%。C 类基金免申购费但有销售服务费",
    )
    realtime = sb.toggle(
        "启用实时数据",
        value=True,
        key="fs_realtime",
        help="把净值序列补齐到最新交易日，并尝试抓取盘中估值",
    )
    auto_refresh = sb.toggle(
        "盘中估值自动刷新",
        value=False,
        key="fs_auto",
        help="开启后每 60 秒重新拉取一次实时估值（仅 akshare 数据源有效）",
    )

    sb.divider()

    # ---------------- ⑤ 界面 ----------------
    sb.markdown("**⑤ 界面**")
    particles = sb.toggle(
        "粒子背景",
        value=True,
        key="fs_particles",
        help="全页 canvas 粒子网络。低性能设备上可关闭。",
    )

    sb.divider()
    c1, c2 = sb.columns(2)
    run = c1.button("🚀 开始分析", type="primary", width="stretch", key="fs_run")
    refresh = c2.button(
        "🔄 刷新数据", width="stretch", key="fs_refresh", help="绕过本地缓存，强制重新向数据源拉取"
    )

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
        calibrate=calibrate,
        embargo=embargo,
        particles=particles,
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
        calibrate=bool(params["calibrate"]),
        embargo=bool(params["embargo"]),
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
        calibrate=p.get("calibrate", True),
        embargo=p.get("embargo", True),
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
        [("walk-forward 验证", "hot"), ("T+1 执行延迟", ""), ("含交易成本", ""), ("红涨绿跌", "")],
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        kpi(
            "严格时序验证",
            "walk-forward",
            "训练数据永远早于预测样本",
            sub="滚动前向推进，杜绝信息穿越",
            tone="flat",
        )
    with c2:
        kpi(
            "可交易收益",
            "T+1 执行",
            "买入价 = T+1 日净值",
            sub="不假设「看完净值还能按净值成交」",
            tone="flat",
        )
    with c3:
        kpi(
            "诚实评估",
            "对基线",
            "多数类基线 + AUC 置信区间",
            sub="直接告诉你模型到底有没有优势",
            tone="flat",
        )
    with c4:
        kpi("成本内建", "双边计费", "默认 15 基点单边", sub="忽略成本的回测没有意义", tone="flat")

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
        note(
            "<b>① 选标的</b><br>点快捷按钮，或直接填 6 位基金代码。"
            "断网时把「数据来源」切成 <code>local</code> 用离线样例。",
            "info",
        )
    with b:
        note(
            "<b>② 选预设</b><br>想要立刻看到东西就选「快速体验」；做正式结论选「严格 · 深度」。",
            "info",
        )
    with c:
        note(
            "<b>③ 点分析</b><br>首次联网取数需要几十秒，之后同参数秒开。"
            "结果会缓存 8 组，方便来回对比。",
            "info",
        )


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
            kpi(
                "数据新鲜度",
                f"{last.date()}",
                "已是最新交易日",
                sub="净值于每交易日约 20:00 后公布",
                tone="up",
            )
        elif age <= 3:
            kpi("数据新鲜度", f"{last.date()}", f"{int(age)} 天前", sub="仍在合理范围", tone="flat")
        else:
            kpi(
                "数据新鲜度",
                f"{last.date()}",
                f"滞后 {int(age)} 天",
                sub="建议点「🔄 刷新数据」",
                tone="down",
            )

    def realtime_cards(payload: dict) -> None:
        items = []
        if payload:
            growth = payload.get("est_growth")
            items.append(
                dict(
                    label="盘中实时估值",
                    value=f"{payload['est_nav']:.4f}",
                    delta=(f"{growth:+.2%}" if growth is not None else None),
                    tone=tone_of(growth),
                    sub=f"估值日 {pd.Timestamp(payload['date']).date()}　"
                    "第三方按持仓拟合，与公布净值有偏差",
                )
            )
        else:
            items.append(
                dict(
                    label="盘中实时估值",
                    value="—",
                    sub="数据源未覆盖该基金，或当前非交易时段",
                )
            )
        items.append(
            dict(
                label="申购状态",
                value=meta.get("purchase_status", "—"),
                sub=f"赎回状态 {meta.get('redeem_status', '—')}",
            )
        )
        items.append(
            dict(
                label="申购费率", value=meta.get("fee", "—"), sub="C 类通常免申购费、另收销售服务费"
            )
        )
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
        {
            "判据": "AUC 显著高于 0.5",
            "数值": f"{fnum(res.auc)}　95%CI [{fnum(boot['lo'], 3)}, {fnum(boot['hi'], 3)}]",
            "判定": verdict(boot["lo"] > 0.5, boot["hi"] > 0.5),
            "说明": "置信区间下界高于 0.5，才能说「不是运气」",
        },
        {
            "判据": "超额准确率 > 2%",
            "数值": fnum(res.wf.overall_metrics.get("acc_edge"), 4),
            "判定": verdict(float(res.wf.overall_metrics.get("acc_edge") or 0) > 0.02),
            "说明": "相对「永远猜多数类」基线的增量",
        },
        {
            "判据": "折间稳定性（各折 AUC > 0.5 占比）",
            "数值": fpct(fold_hit, 0),
            "判定": verdict(fold_hit >= 0.6, fold_hit >= 0.4),
            "说明": "若只有个别折有效，多半是过拟合",
        },
        {
            "判据": "折间 AUC 波动",
            "数值": fnum(fold_std, 4),
            "判定": verdict(fold_std < 0.05, fold_std < 0.08),
            "说明": "标准差越小，结论越可复现",
        },
        {
            "判据": "|IC| ≥ 0.03",
            "数值": fnum(ic, 4),
            "判定": verdict(abs(ic) >= 0.03, abs(ic) >= 0.015),
            "说明": "不依赖阈值的单调关系；低于 0.03 基本等同噪音",
        },
        {
            "判据": "概率校准 ECE < 0.05",
            "数值": fnum(ece, 4),
            "判定": verdict(ece < 0.05, ece < 0.08),
            "说明": "概率能否当概率用，而不只是排序",
        },
        {
            "判据": "概率分化度（输出标准差）",
            "数值": fnum(prob_spread, 4),
            "判定": verdict(prob_spread > 0.05, prob_spread > 0.03),
            "说明": "全都挤在 0.5 附近 = 模型不敢表态",
        },
        {
            "判据": "阈值稳健（扫描中跑赢持有的比例）",
            "数值": fpct(beat_ratio, 0),
            "判定": verdict(beat_ratio >= 0.6, beat_ratio >= 0.4),
            "说明": "只在某一阈值上亮眼，就是过拟合的信号",
        },
        {
            "判据": "样本量 ≥ 350 行",
            "数值": fint(len(frame)),
            "判定": verdict(len(frame) >= 350, len(frame) >= 200),
            "说明": "小样本下 AUC 的抽样噪声极大",
        },
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
        + (
            ""
            if pass_n >= 6
            else " <b>在这种情况下不要相信回测曲线</b>——"
            "策略赚钱往往来自市场本身的 beta，而不是模型的 alpha。"
        ),
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
    pct = pd.Series({name: float((hist[name] < row[name]).mean()) for name in names}, dtype=float)

    tbl = pd.DataFrame(
        {
            "特征": z.index,
            "当前值": row[z.index].to_numpy(),
            "历史均值": mean[z.index].to_numpy(),
            "z 分数": z.to_numpy(),
            "历史分位": pct[z.index].to_numpy(),
        }
    )
    tbl["历史分位"] = tbl["历史分位"].map(lambda v: fpct(v, 0))
    tbl = tbl.reindex(tbl["z 分数"].abs().sort_values(ascending=False).index)

    top = tbl.head(8)
    fig = go.Figure(
        go.Bar(
            x=top["z 分数"][::-1],
            y=top["特征"][::-1],
            orientation="h",
            marker_color=[
                theme()["up"] if v >= 0 else theme()["down"] for v in top["z 分数"][::-1]
            ],
            text=[
                f"{v:+.2f}σ · 分位 {p}"
                for v, p in zip(top["z 分数"][::-1], top["历史分位"][::-1], strict=True)
            ],
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
                        axis=dict(
                            range=[0, 100],
                            tickvals=[0, 25, 50, 75, 100],
                            tickfont=dict(size=10, color=theme()["muted"]),
                        ),
                        bar=dict(
                            color=theme()["up"] if prob >= thr else theme()["down"], thickness=0.34
                        ),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 35], color=theme()["band"]),
                            dict(range=[35, 50], color=theme()["mid"]),
                            dict(range=[50, 65], color=theme()["band"]),
                            dict(range=[65, 100], color=theme()["mid"]),
                        ],
                        threshold=dict(
                            line=dict(color=theme()["accent"], width=3),
                            thickness=0.86,
                            value=thr * 100,
                        ),
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
            dict(
                label="样本外 AUC",
                value=fnum(m["auc"]),
                delta=f"相对 0.5 超额 {fnum((m['auc'] or np.nan) - 0.5, 4)}",
                tone=tone_of((m["auc"] or np.nan) - 0.5, zero_is_flat=False),
                sub="0.5 = 与抛硬币无异",
            ),
            dict(
                label="准确率 vs 基线",
                value=fpct(m["accuracy"], 1),
                delta=f"超额 {fnum(m['acc_edge'], 4)}",
                tone=tone_of(m["acc_edge"]),
                sub=f"多数类基线 {fpct(m['majority_acc'], 1)}",
            ),
            dict(
                label="策略累计收益",
                value=fpct(m["strategy_return"], 1),
                delta=f"买入持有 {fpct(m['benchmark_return'], 1)}",
                tone=tone_of(m["strategy_return"]),
                sub=f"超额 {fpct(m['excess_return'], 1)}",
            ),
            dict(
                label="夏普比率",
                value=fnum(m["sharpe"], 2),
                tone=tone_of(m["sharpe"]),
                sub="无风险利率按 0 处理",
            ),
            dict(
                label="最大回撤",
                value=fpct(m["max_drawdown"], 1),
                tone="down" if (m["max_drawdown"] or 0) < -0.15 else "flat",
                sub="越大越考验持有耐心",
            ),
            dict(
                label="持仓时间占比",
                value=fpct(m["exposure"], 1),
                tone="flat",
                sub=f"样本外 {fint(bt.strategy_metrics.get('n_days'))} 个交易日",
            ),
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
    fig.add_trace(
        go.Scatter(
            x=navi["date"],
            y=navi["nav"],
            mode="lines",
            name="单位净值",
            line=dict(color=theme()["info"], width=2),
            fill="tozeroy",
            fillcolor="rgba(91,127,166,0.10)",
            hovertemplate="%{x|%Y-%m-%d}<br>净值 %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=navi["date"],
            y=navi["ma60"],
            mode="lines",
            name="60日均线",
            line=dict(color=theme()["accent"], width=1.4, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>MA60 %{y:.4f}<extra></extra>",
        )
    )
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
    fig.add_trace(
        go.Scatter(
            x=prob.index,
            y=prob.values,
            mode="lines",
            name="上涨概率",
            line=dict(color=t["accent"], width=1.6),
            hovertemplate="%{x|%Y-%m-%d}<br>概率 %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[prob.index.min(), prob.index.max()],
            y=[0.5, 0.5],
            mode="lines",
            name="0.5 中性线",
            line=dict(color=t["neutral"], dash="dot", width=1),
            hoverinfo="skip",
        )
    )
    fig.add_hline(
        y=res.config.threshold,
        line_dash="dash",
        line_color=t["info"],
        annotation_text=f"阈值 {res.config.threshold:.2f}",
        annotation_position="top left",
    )
    style_fig(fig, "", 380, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="上涨概率", range=[0, 1])
    time_selector(fig)
    chart(fig)

    c1, c2 = st.columns(2, gap="small")

    with c1:
        fig = go.Figure(
            go.Histogram(
                x=prob.values,
                nbinsx=40,
                marker_color=t["accent"],
                opacity=0.85,
                hovertemplate="概率 %{x:.3f}<br>%{y} 次<extra></extra>",
            )
        )
        fig.add_vline(x=0.5, line_dash="dot", line_color=t["neutral"])
        style_fig(fig, "概率分布越集中，模型越「不敢表态」", 320)
        fig.update_xaxes(title="上涨概率")
        fig.update_yaxes(title="样本数", rangemode="tozero")
        chart(fig)
        spread = float(prob.std(ddof=1))
        st.caption(
            f"概率标准差 {fnum(spread, 4)}。"
            + (
                "分布明显张开，模型愿意给出差异化判断。"
                if spread > 0.05
                else "分布几乎贴着 0.5，说明模型的判断与常数无异——这与弱信号是一致的。"
            )
        )

    with c2:
        grp, _ = calibration_table(prob, y_true, n_bins=10)
        if grp.empty:
            note("样本不足，无法计算校准曲线。", "warn")
        else:
            grp = grp.assign(count=grp["n"])
            fig = go.Figure()
            mx = float(max(grp["pred"].max(), grp["obs"].max())) * 1.08
            fig.add_trace(
                go.Scatter(
                    x=[0, mx],
                    y=[0, mx],
                    mode="lines",
                    name="完美校准",
                    line=dict(color=t["neutral"], dash="dash", width=1.2),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=grp["pred"],
                    y=grp["obs"],
                    mode="lines+markers",
                    name="实际频率",
                    line=dict(color=t["accent"], width=2),
                    marker=dict(size=8, color=t["accent"]),
                    customdata=grp["n"].to_numpy(),
                    hovertemplate="预测 %{x:.3f}<br>实际 %{y:.3f}<br>样本 %{customdata}<extra></extra>",
                )
            )
            style_fig(fig, "校准曲线：模型说的 60%，是否真有 60% 会涨？", 320, legend=True)
            fig.update_xaxes(title="分档平均预测概率")
            fig.update_yaxes(title="分档实际上涨频率")
            chart(fig)
            _, ece = calibration_table(prob, y_true, 10)
            st.caption(
                f"期望校准误差 ECE = {fnum(ece, 4)}。"
                + (
                    "概率读数基本可信。"
                    if ece < 0.05
                    else "偏离对角线较多，概率只能当排序看，不能直接当胜率用。"
                )
            )

    # ---------------- 概率分档的实际表现 ----------------
    section("概率分档 → 实际涨跌", "若模型有效，各档上涨率应单调上升")
    n_bins = st.segmented_control("分档数", [5, 10], default=10, key="fs_pred_bins")
    n_bins = int(n_bins or 10)
    try:
        bins = pd.qcut(prob, n_bins, labels=False, duplicates="drop")
        grp = (
            pd.DataFrame({"bin": bins, "y": y_true.to_numpy(), "r": fwd.to_numpy()})
            .groupby("bin", observed=True)
            .agg(up_rate=("y", "mean"), mean_ret=("r", "mean"), n=("y", "size"))
        )
        grp.index = [f"Q{i + 1}" for i in range(len(grp))]
        base = float(y_true.mean())

        c1, c2 = st.columns(2, gap="small")
        with c1:
            fig = go.Figure(
                go.Bar(
                    x=grp.index,
                    y=grp["up_rate"],
                    marker_color=[t["up"] if v >= base else t["down"] for v in grp["up_rate"]],
                    text=[
                        f"{v:.1%}<br>n={int(n)}"
                        for v, n in zip(grp["up_rate"], grp["n"], strict=True)
                    ],
                    textposition="outside",
                    hovertemplate="%{x}<br>上涨率 %{y:.1%}<extra></extra>",
                )
            )
            fig.add_hline(
                y=base,
                line_dash="dash",
                line_color=t["neutral"],
                annotation_text=f"整体上涨率 {base:.1%}",
            )
            style_fig(fig, "各档实际上涨频率", 340)
            fig.update_yaxes(title="上涨频率", tickformat=".0%", rangemode="tozero")
            chart(fig)

        with c2:
            fig = go.Figure(
                go.Bar(
                    x=grp.index,
                    y=grp["mean_ret"],
                    marker_color=[t["up"] if v >= 0 else t["down"] for v in grp["mean_ret"]],
                    text=[fpct(v, 2, sign=True) for v in grp["mean_ret"]],
                    textposition="outside",
                    hovertemplate="%{x}<br>平均收益 %{y:.3%}<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
            style_fig(fig, "各档平均可交易收益", 340)
            fig.update_yaxes(title="平均未来收益", tickformat=".2%")
            chart(fig)

        mono = float(
            pd.Series(grp["up_rate"].to_numpy()).corr(
                pd.Series(np.arange(len(grp))), method="spearman"
            )
        )
        note(
            f"分档上涨率与档位序号的秩相关为 <b>{fnum(mono, 3)}</b>。"
            + (
                "越接近 1，说明概率越高、真实上涨频率也越高，模型的排序能力成立。"
                if mono > 0.7
                else "排序关系不单调——高概率档并没有对应更高的真实上涨频率，"
                "说明模型输出的「概率」排序能力有限。"
            ),
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
        fig.add_trace(
            go.Scatter(
                x=roll.index,
                y=roll,
                mode="lines",
                name="滚动 IC",
                line=dict(color=t["accent"], width=1.6),
                hovertemplate="%{x|%Y-%m-%d}<br>IC %{y:.3f}<extra></extra>",
            )
        )
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
            + (
                "IC 长期在 0 附近来回穿越——这是「信号极弱」的典型形态，"
                "意味着任何单次交易的结果都主要由噪声决定。"
                if abs(valid.mean()) < 0.03
                else "IC 存在一定正向偏移，但仍需扣除交易成本后再判断是否有净收益。"
            ),
            "info",
        )

    # ---------------- 仓位时间轴 ----------------
    section("信号触发的持仓区间", "把「什么时候在场」摊开看")
    daily = res.backtest.daily
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["position"],
            mode="lines",
            name="仓位",
            line=dict(color=t["up"], width=1.2, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(214,69,69,0.22)",
            hovertemplate="%{x|%Y-%m-%d}<br>仓位 %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index,
            y=(daily["prob"] - 0.5) * 2,
            mode="lines",
            name="概率（映射到 ±1）",
            line=dict(color=t["info"], width=1.1, dash="dot"),
            opacity=0.75,
            hovertemplate="%{x|%Y-%m-%d}<br>偏移 %{y:.2f}<extra></extra>",
        )
    )
    style_fig(fig, "红色的「在场」区间与概率偏移是否同步？", 300, legend=True)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="仓位 / 概率偏移", range=[-1.15, 1.35])
    time_selector(fig)
    chart(fig)

    ic = information_coefficient(prob, fwd)
    kpi_row(
        [
            dict(
                label="信息系数 IC（全样本）",
                value=fnum(ic, 4),
                delta="不依赖阈值的诚实指标",
                tone=tone_of(ic, zero_is_flat=False),
                sub="|IC| < 0.03 基本等同于噪音",
            ),
            dict(
                label="持仓时间占比",
                value=fpct(res.backtest.exposure, 1),
                tone="flat",
                sub="阈值越高，在场时间越短",
            ),
            dict(
                label="样本外交易日数",
                value=fint(len(prob)),
                tone="flat",
                sub=f"占全样本 {fpct(len(prob) / max(len(frame), 1), 0)}",
            ),
            dict(
                label="概率标准差",
                value=fnum(float(prob.std(ddof=1)), 4),
                tone="flat",
                sub="越大说明模型越愿意表态",
            ),
        ],
        cols=4,
    )


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
            fig.add_trace(
                go.Scatter(
                    x=[
                        pd.Timestamp(row["train_end"])
                        - pd.Timedelta(days=int(row["n_train"]) * 365 / 244),
                        pd.Timestamp(row["train_end"]),
                    ],
                    y=[label, label],
                    mode="lines",
                    line=dict(color=t["bad"], width=13),
                    name="训练窗口",
                    showlegend=(i == 0),
                    hovertemplate=f"{label}<br>训练至 {row['train_end']}"
                    f"<br>{int(row['n_train'])} 行<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(row["test_start"]), pd.Timestamp(row["test_end"])],
                    y=[label, label],
                    mode="lines",
                    line=dict(color=t["up"], width=13),
                    name="测试窗口",
                    showlegend=(i == 0),
                    hovertemplate=f"{label}<br>{row['test_start']} ~ {row['test_end']}"
                    f"<br>{int(row['n_test'])} 行<extra></extra>",
                )
            )
        style_fig(fig, "训练窗口随折数推进而变长，测试段依次向前滚动", 300, legend=True)
        fig.update_yaxes(title="", autorange="reversed")
        fig.update_xaxes(title="时间")
        chart(fig)

    c1, c2 = st.columns([1.15, 1], gap="small")

    with c1:
        if not folds.empty:
            aucs = folds["auc"].to_numpy(dtype=float)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=folds["fold"].astype(str),
                    y=aucs,
                    mode="lines+markers",
                    name="AUC",
                    line=dict(color=t["neutral"], width=1.2, dash="dot"),
                    marker=dict(size=11, color=[t["up"] if a >= 0.5 else t["down"] for a in aucs]),
                    text=[f"{a:.3f}" for a in aucs],
                    textposition="top center",
                    hovertemplate="折 %{x}<br>AUC %{y:.4f}<extra></extra>",
                )
            )
            fig.add_hline(
                y=0.5, line_dash="dash", line_color=t["neutral"], annotation_text="0.5 = 无预测能力"
            )
            lo = min(0.46, float(np.nanmin(aucs)) - 0.03)
            hi = max(0.58, float(np.nanmax(aucs)) + 0.05)
            style_fig(fig, "各折 AUC：点越分散，说明结论越不稳", 340)
            fig.update_yaxes(title="AUC", range=[lo, hi])
            chart(fig)

    with c2:
        boot = bootstrap_auc(y_true, prob, seed=res.config.random_state)
        if boot["n"]:
            fig = go.Figure(
                go.Histogram(
                    x=boot["samples"],
                    nbinsx=36,
                    marker_color=t["info"],
                    opacity=0.85,
                    hovertemplate="AUC %{x:.3f}<br>%{y} 次<extra></extra>",
                )
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color=t["neutral"], annotation_text="0.5")
            fig.add_vline(
                x=res.auc,
                line_color=t["accent"],
                line_width=2.5,
                annotation_text=f"实测 {res.auc:.3f}",
            )
            style_fig(fig, f"Bootstrap AUC 分布（{boot['n']} 次重抽）", 340)
            fig.update_xaxes(title="AUC")
            fig.update_yaxes(title="出现次数", rangemode="tozero")
            chart(fig)
            sig = "显著" if boot["lo"] > 0.5 else "不显著"
            note(
                f"AUC 的 95% 置信区间为 <b>[{fnum(boot['lo'], 3)}, {fnum(boot['hi'], 3)}]</b>，"
                f"重抽中 {fpct(boot['p_le_half'], 1)} 次不高于 0.5。"
                f"结论：<b>{sig}</b>。"
                + (
                    ""
                    if boot["lo"] > 0.5
                    else " 区间跨越 0.5 意味着「这点预测力可能纯属抽样噪声」——"
                    "这是判断模型价值最有说服力的一张图。"
                ),
                "ok" if boot["lo"] > 0.5 else "warn",
            )
        else:
            note("样本不足，无法做 Bootstrap 置信区间。", "warn")

    if not folds.empty:
        section("各折明细")
        cols = [
            "fold",
            "train_end",
            "test_start",
            "test_end",
            "n_train",
            "n_test",
            "auc",
            "accuracy",
            "majority_acc",
            "acc_edge",
            "brier",
        ]
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
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="随机猜测",
                    line=dict(color=t["neutral"], dash="dash", width=1.2),
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"AUC = {res.auc:.4f}",
                    line=dict(color=t["accent"], width=2.6),
                    fill="tozeroy",
                    fillcolor="rgba(201,134,26,0.10)",
                    hovertemplate="假正率 %{x:.3f}<br>真正率 %{y:.3f}<extra></extra>",
                )
            )
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
            fig.add_hline(
                y=base,
                line_dash="dash",
                line_color=t["neutral"],
                annotation_text=f"基准率 {base:.1%}",
            )
            fig.add_trace(
                go.Scatter(
                    x=rec,
                    y=pre,
                    mode="lines",
                    name=f"AP = {ap:.4f}",
                    line=dict(color=t["info"], width=2.6),
                    fill="tozeroy",
                    fillcolor="rgba(62,124,177,0.10)",
                    hovertemplate="召回 %{x:.3f}<br>精确 %{y:.3f}<extra></extra>",
                )
            )
            style_fig(fig, "PR 曲线（不平衡数据下比 ROC 更敏感）", 360, legend=True)
            fig.update_xaxes(title="召回率", range=[0, 1])
            fig.update_yaxes(title="精确率", range=[0, 1.02])
            chart(fig)
            st.caption(
                f"平均精确率 AP = {fnum(ap, 4)}，上涨基准率 {fpct(base, 1)}。"
                + (
                    "AP 高于基准率，看多信号略优于随机抽样。"
                    if ap > base
                    else "AP 不高于基准率，说明「看多」信号并没有筛出更好的样本。"
                )
            )
        except ValueError as err:
            st.info(f"无法绘制 PR：{err}")

    section("特征重要性", "按贡献占比归一化，可跨模型比较")
    if len(res.importance):
        c1, c2 = st.columns([1.4, 1], gap="small")
        with c1:
            topn = st.slider("显示前 N 个特征", 5, min(30, len(res.importance)), 15, key="fs_topn")
            top = res.importance.head(int(topn))[::-1]
            fig = go.Figure(
                go.Bar(
                    x=top.values,
                    y=top.index,
                    orientation="h",
                    marker_color=[
                        t["accent"] if i < 3 else t["info"] for i in range(len(top) - 1, -1, -1)
                    ],
                    text=[f"{v:.1%}" for v in top.values],
                    textposition="outside",
                    hovertemplate="%{y}<br>贡献占比 %{x:.2%}<extra></extra>",
                )
            )
            style_fig(
                fig, f"贡献最大的 {int(topn)} 个特征（前 3 名高亮）", max(340, 24 * int(topn))
            )
            fig.update_xaxes(title="贡献占比", tickformat=".0%")
            chart(fig)

        with c2:
            grp = pd.Series(
                {
                    name: float(res.importance.reindex(cols).fillna(0.0).sum())
                    for name, cols in FEATURE_GROUPS.items()
                }
            ).sort_values(ascending=False)
            grp = grp[grp > 0]
            fig = go.Figure(
                go.Bar(
                    x=grp.values,
                    y=grp.index,
                    orientation="h",
                    marker_color=t["info"],
                    text=[f"{v:.1%}" for v in grp.values],
                    textposition="outside",
                    hovertemplate="%{y}<br>合计占比 %{x:.2%}<extra></extra>",
                )
            )
            style_fig(fig, "特征分组的合计贡献", 320)
            fig.update_xaxes(title="合计贡献占比", tickformat=".0%")
            chart(fig)
            dominant = grp.index[0]
            note(
                f"贡献最大的分组是 <b>{dominant}</b>（占 "
                f"{fpct(float(grp.iloc[0]), 1)}）。"
                + (
                    "模型主要依赖价格自身的动量与波动，市场环境信息只是配角。"
                    if dominant.startswith("价格")
                    else "模型较依赖外部市场环境，需注意基准指数取数失败时信号会退化。"
                ),
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
        fig = go.Figure(
            go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=["预测跌", "预测涨"],
                y=["实际跌", "实际涨"],
                text=[
                    [
                        f"TN<br>{tn}<br>{fpct(tn / total, 1)}",
                        f"FP<br>{fp}<br>{fpct(fp / total, 1)}",
                    ],
                    [
                        f"FN<br>{fn}<br>{fpct(fn / total, 1)}",
                        f"TP<br>{tp}<br>{fpct(tp / total, 1)}",
                    ],
                ],
                texttemplate="%{text}",
                textfont=dict(size=12),
                colorscale="Blues",
                showscale=False,
                hovertemplate="%{y} / %{x}<br>%{z} 次<extra></extra>",
            )
        )
        style_fig(fig, "", 320)
        chart(fig)
    with c2:
        kpi_row(
            [
                dict(label="精确率 Precision", value=fpct(prec, 1), sub="看多信号里真正上涨的比例"),
                dict(label="召回率 Recall", value=fpct(rec, 1), sub="真实上涨里被抓住的比例"),
                dict(label="F1", value=fnum(f1, 4), sub="精确与召回的调和平均"),
                dict(
                    label="预测看多占比",
                    value=fpct((tp + fp) / total, 1),
                    sub=f"实际上涨占比 {fpct((tp + fn) / total, 1)}",
                ),
            ],
            cols=2,
        )


# ================================================================= 回测
def render_backtest(res) -> None:
    frame = res.dataset.frame
    prob = res.wf.predictions
    fwd = frame.loc[prob.index, "fwd_ret"]
    t = theme()

    section("交互回测", "以下滑块不会重新训练模型，只重算回测，可实时观察敏感度")
    c1, c2, c3 = st.columns(3, gap="small")
    thr = c1.slider("看多阈值", 0.30, 0.70, 0.50, 0.01, key="bt_thr")
    cost = c2.slider("单边成本（基点）", 0.0, 100.0, float(res.config.cost_bps), 1.0, key="bt_cost")
    lag = c3.slider(
        "额外执行延迟（交易日）",
        0,
        5,
        0,
        1,
        key="bt_lag",
        help="0 = 信号次日按净值成交（已内含在收益定义中）；调大做压力测试",
    )

    bt = backtest_threshold(
        prob, fwd, threshold=float(thr), cost_bps=float(cost), execution_lag=int(lag)
    )

    kpi_row(
        [
            dict(
                label="策略累计收益",
                value=fpct(bt.strategy_metrics["total_return"], 1),
                tone=tone_of(bt.strategy_metrics["total_return"]),
                sub=f"年化 {fpct(bt.strategy_metrics['annual_return'], 1)}",
            ),
            dict(
                label="买入持有",
                value=fpct(bt.benchmark_metrics["total_return"], 1),
                tone=tone_of(bt.benchmark_metrics["total_return"]),
                sub=f"年化 {fpct(bt.benchmark_metrics['annual_return'], 1)}",
            ),
            dict(
                label="超额收益",
                value=fpct(bt.excess_return, 1),
                delta="跑赢持有" if bt.excess_return > 0 else "跑输持有",
                tone=tone_of(bt.excess_return, zero_is_flat=False),
                sub="扣费后的净差额",
            ),
            dict(
                label="持仓日胜率",
                value=fpct(bt.win_rate_active, 1),
                tone=tone_of(bt.win_rate_active - 0.5, zero_is_flat=False),
                sub="只统计有仓位的日子，比全样本胜率更反映择时",
            ),
            dict(
                label="最大回撤",
                value=fpct(bt.strategy_metrics["max_drawdown"], 1),
                tone="down" if bt.strategy_metrics["max_drawdown"] < -0.15 else "flat",
                sub=f"持有买入回撤 {fpct(bt.benchmark_metrics['max_drawdown'], 1)}",
            ),
            dict(
                label="持仓占比",
                value=fpct(bt.exposure, 1),
                tone="flat",
                sub=f"交易 {fint(bt.strategy_metrics['n_trades'])} 次",
            ),
        ],
        cols=3,
    )

    # ---------------- 净值 + 回撤 ----------------
    eq = pd.DataFrame(
        {
            "策略": (1 + bt.daily["strategy_ret"]).cumprod(),
            "买入持有": (1 + bt.daily["benchmark_ret"]).cumprod(),
        }
    )
    dd = eq["策略"] / eq["策略"].cummax() - 1

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.07,
        subplot_titles=("净值曲线（起点归一）", "策略回撤（水下曲线）"),
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq["策略"],
            name="策略",
            line=dict(color=t["up"], width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>策略 %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq["买入持有"],
            name="买入持有",
            line=dict(color=t["info"], width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>持有 %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="回撤",
            line=dict(color=t["down"], width=1.4),
            fill="tozeroy",
            fillcolor="rgba(47,158,95,0.20)",
            hovertemplate="%{x|%Y-%m-%d}<br>回撤 %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
    )
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
        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["毛收益（不计成本）", "交易成本拖累", "净收益"],
                y=[gross_ret, -drag, net_ret],
                text=[
                    fpct(gross_ret, 2, sign=True),
                    fpct(-drag, 2, sign=True),
                    fpct(net_ret, 2, sign=True),
                ],
                textposition="outside",
                connector=dict(line=dict(color=t["neutral"], width=1, dash="dot")),
                increasing=dict(marker=dict(color=t["up"])),
                decreasing=dict(marker=dict(color=t["down"])),
                totals=dict(marker=dict(color=t["accent"])),
                hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
        style_fig(fig, "成本吃掉多少收益？", 360)
        fig.update_yaxes(title="累计收益", tickformat=".0%")
        chart(fig)
    with c2:
        cover = (gross_ret / drag) if drag > 0 else float("inf")
        kpi_row(
            [
                dict(
                    label="累计换手（仓位变动量）",
                    value=f"{turnover_sum:.1f}",
                    sub=f"平均每次变动 1.0，共 {fint(bt.strategy_metrics['n_trades'])} 次交易",
                ),
                dict(
                    label="累计成本",
                    value=fpct(-cost_sum, 2, sign=True),
                    tone="down" if cost_sum > 0 else "flat",
                    sub=f"单边 {cost:.0f} 基点 × {turnover_sum:.1f} 换手",
                ),
                dict(
                    label="成本拖累占毛收益",
                    value=fpct(drag / gross_ret, 1) if gross_ret else "—",
                    tone="down" if drag > 0 else "flat",
                    sub="越高说明策略越「手忙脚乱」",
                ),
                dict(
                    label="毛收益对成本的覆盖倍数",
                    value=(f"{cover:.2f}×" if np.isfinite(cover) else "∞"),
                    tone=tone_of(cover - 1 if np.isfinite(cover) else 1, zero_is_flat=False),
                    sub="< 1 表示成本直接吃穿收益",
                ),
            ],
            cols=2,
        )
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
            fig.add_trace(
                go.Scatter(
                    x=sw["threshold"],
                    y=sw["total_return"],
                    mode="lines+markers",
                    name="策略累计收益",
                    line=dict(color=t["up"], width=2.4),
                    marker=dict(size=7),
                    hovertemplate="阈值 %{x:.2f}<br>累计收益 %{y:.2%}<extra></extra>",
                )
            )
            fig.add_hline(
                y=float(bt.benchmark_metrics["total_return"]),
                line_dash="dash",
                line_color=t["info"],
                annotation_text="买入持有",
            )
            fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
            fig.add_vline(
                x=float(thr), line_color=t["accent"], line_width=1.6, annotation_text="当前"
            )
            style_fig(fig, "不同阈值下的策略累计收益", 340)
            fig.update_xaxes(title="看多阈值")
            fig.update_yaxes(title="累计收益", tickformat=".0%")
            chart(fig)

        with c2:
            fig = go.Figure(
                go.Scatter(
                    x=sw["exposure"],
                    y=sw["total_return"],
                    mode="markers+lines",
                    marker=dict(
                        size=11,
                        color=sw["threshold"],
                        colorscale=[[0, t["info"]], [1, t["accent"]]],
                        showscale=True,
                        colorbar=dict(title="阈值", thickness=12, len=0.7),
                    ),
                    line=dict(color=t["neutral"], width=1, dash="dot"),
                    name="阈值路径",
                    text=[f"{v:.2f}" for v in sw["threshold"]],
                    hovertemplate="阈值 %{text}<br>持仓占比 %{x:.1%}"
                    "<br>累计收益 %{y:.2%}<extra></extra>",
                )
            )
            fig.add_hline(
                y=float(bt.benchmark_metrics["total_return"]),
                line_dash="dash",
                line_color=t["info"],
            )
            style_fig(fig, "阈值前沿：持仓越久收益越高，还是越少越稳？", 340)
            fig.update_xaxes(title="持仓时间占比", tickformat=".0%")
            fig.update_yaxes(title="累计收益", tickformat=".0%")
            chart(fig)

        st.dataframe(
            sw.assign(持仓占比=sw["exposure"] * 100).drop(columns=["exposure"]),
            width="stretch",
            hide_index=True,
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
            piv = pd.DataFrame(
                {"y": monthly.index.year, "m": monthly.index.month, "r": monthly.values}
            ).pivot(index="y", columns="m", values="r")
            fig = go.Figure(
                go.Heatmap(
                    z=piv.values,
                    x=[f"{c}月" for c in piv.columns],
                    y=piv.index.astype(str),
                    colorscale=[[0, t["down"]], [0.5, t["mid"]], [1, t["up"]]],
                    zmid=0,
                    colorbar=dict(title="收益", thickness=12),
                    hovertemplate="%{y}年%{x}<br>%{z:.2%}<extra></extra>",
                )
            )
            style_fig(fig, "月度收益热力图（红涨绿跌）", 340)
            chart(fig)
        else:
            st.info("样本区间过短，无法做月度聚合。")

    with c2:
        yr = pd.DataFrame(
            {
                "策略": (1 + bt.daily["strategy_ret"]).resample("YE").prod() - 1,
                "买入持有": (1 + bt.daily["benchmark_ret"]).resample("YE").prod() - 1,
            }
        )
        yr.index = [str(i.year) for i in yr.index]
        if len(yr) >= 1:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=yr.index,
                    y=yr["策略"],
                    name="策略",
                    marker_color=t["up"],
                    text=[fpct(v, 1, sign=True) for v in yr["策略"]],
                    textposition="outside",
                    hovertemplate="%{x} 策略 %{y:.2%}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=yr.index,
                    y=yr["买入持有"],
                    name="买入持有",
                    marker_color=t["info"],
                    text=[fpct(v, 1, sign=True) for v in yr["买入持有"]],
                    textposition="outside",
                    hovertemplate="%{x} 持有 %{y:.2%}<extra></extra>",
                )
            )
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
            lambda v: "尚未恢复" if v is None or pd.isna(v) else str(pd.Timestamp(v).date())
        )
        st.dataframe(eps_disp, width="stretch", hide_index=True)
        longest = eps.loc[eps["持续(交易日)"].idxmax()]
        end_raw = longest["恢复"]
        end_txt = "尚未恢复" if pd.isna(end_raw) else str(pd.Timestamp(end_raw).date())
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


# ================================================================= 风控与仓位
@st.cache_data(show_spinner=False, ttl=21600)
def _cached_complexity_path(
    code: str, horizon: int, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """复杂度扫描（含 5 次滚动前向训练），结果缓存 6 小时。"""
    return audit.complexity_path(X, y, n_splits=n_splits, embargo=max(1, horizon))


@st.cache_data(show_spinner=False, ttl=21600)
def _cached_leakage_audit(
    code: str, horizon: int, frame: pd.DataFrame, feature_names: tuple[str, ...]
) -> pd.DataFrame:
    """泄漏审计（三组对照各跑一次滚动前向），结果缓存 6 小时。"""
    return audit.leakage_audit(frame, list(feature_names), horizon=horizon)


def _position_curve(prob: np.ndarray, lo: float, hi: float, cap: float) -> np.ndarray:
    """分段线性仓位映射：``p<=lo`` 空仓，``p>=hi`` 满仓，中间线性过渡。

    为什么不用纯 Kelly？因为 Kelly 对概率误差极其敏感——概率高估 5 个百分点，
    Kelly 仓位会翻好几倍，而我们的概率本身只有 ±0.04 的校准误差。
    所以实务上统一用「有死区的线性映射 + 仓位上限」，再拿 Kelly 当参照系看有没有超标。
    """
    p = np.asarray(prob, dtype=float)
    out = np.zeros_like(p)
    mid = (p > lo) & (p < hi)
    out[mid] = cap * (p[mid] - lo) / max(hi - lo, 1e-9)
    out[p >= hi] = cap
    return np.clip(out, 0.0, 1.0)


def render_risk(res) -> None:
    """把校准后的概率翻译成「买多少」，并做压力测试。"""
    frame = res.dataset.frame
    prob = res.wf.predictions
    y_true = frame.loc[prob.index, "label"].astype(int)
    fwd = frame.loc[prob.index, "fwd_ret"]

    section("从概率到仓位", "校准让概率具备了频率含义，因此它可以被当作仓位使用")
    note(
        "未校准的概率只能排序、不能定量——模型说 0.8 并不代表 80% 会涨。"
        "本页用的是 <b>样本外 isotonic 校准</b>之后概率，"
        "所以「概率 0.60 → 仓位 30%」是有意义的映射。"
        "<br>但请注意：<span class='hi'>校准能修正概率的刻度，不能提高模型的排序能力</span>。"
        "如果 AUC 只有 0.52，那么把仓位做得再精细，也赚不回成本。",
        "info",
    )

    cur_prob = float(res.latest_prob) if not np.isnan(res.latest_prob) else float("nan")

    c1, c2, c3 = st.columns([1.25, 1, 1], gap="small")
    with c1:
        lo = st.slider(
            "空仓阈值（概率低于此值不持有）",
            0.30,
            0.55,
            0.52,
            0.01,
            key="fs_pos_lo",
            help="概率低于该值时完全空仓，避免在噪声区反复交易",
        )
    with c2:
        hi = st.slider(
            "满仓阈值", 0.55, 0.90, 0.62, 0.01, key="fs_pos_hi", help="概率高于该值时持有到仓位上限"
        )
    with c3:
        cap = st.slider(
            "仓位上限",
            0.1,
            1.0,
            0.6,
            0.05,
            key="fs_pos_cap",
            help="单标的仓位上限。永远不要满仓单一标的",
        )

    if hi <= lo:
        note("满仓阈值必须大于空仓阈值，已按默认值回退。", "warn")
        lo, hi = 0.52, 0.62

    pos_now = float(_position_curve(np.array([cur_prob]), lo, hi, cap)[0])
    # 纯 Kelly：b=1 时 f* = 2p-1；再取 1/4 Kelly 作为保守参照
    kelly_full = max(0.0, 2.0 * cur_prob - 1.0) if not np.isnan(cur_prob) else float("nan")
    kelly_quarter = kelly_full / 4.0 if not np.isnan(kelly_full) else float("nan")

    conf_now = abs(cur_prob - 0.5) if not np.isnan(cur_prob) else float("nan")
    conf_hist = (prob - 0.5).abs()
    pct_rank = (
        float((conf_hist < conf_now).mean() * 100) if not np.isnan(conf_now) else float("nan")
    )

    kpi_row(
        [
            dict(
                label="最新上涨概率（已校准）",
                value=fpct(cur_prob, 1) if cur_prob == cur_prob else "—",
                tone=tone_of(cur_prob - 0.5) if cur_prob == cur_prob else "flat",
                sub=f"净值日 {res.last_nav_date.date() if res.last_nav_date is not None else '—'}",
            ),
            dict(
                label="建议仓位（当前映射）",
                value=fpct(pos_now, 1),
                tone="accent",
                sub=f"空仓<{lo:.2f} → 满仓>{hi:.2f}，上限 {cap:.0%}",
            ),
            dict(
                label="纯 Kelly 参照",
                value=fpct(kelly_full, 1) if kelly_full == kelly_full else "—",
                tone="info",
                sub=f"1/4 Kelly = {fpct(kelly_quarter, 1) if kelly_quarter == kelly_quarter else '—'}",
            ),
            dict(
                label="模型自信度分位",
                value=f"{pct_rank:.0f}%" if pct_rank == pct_rank else "—",
                tone="flat",
                sub="在全部样本外预测中，今天这个「|p−0.5|」排在什么位置",
            ),
        ],
        cols=4,
    )

    # ---------------------------------------------------------- 映射曲线
    section("仓位映射曲线", "横轴是模型给出的概率，纵轴是按当前参数应持有的仓位")
    grid = np.linspace(0.0, 1.0, 201)
    curve = _position_curve(grid, lo, hi, cap)
    kelly_curve = np.clip(2.0 * grid - 1.0, 0, 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=curve,
            mode="lines",
            name="当前映射",
            line=dict(color=theme()["accent"], width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=kelly_curve,
            mode="lines",
            name="纯 Kelly（参照）",
            line=dict(color=theme()["neutral"], width=1.4, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=kelly_curve / 4.0,
            mode="lines",
            name="1/4 Kelly",
            line=dict(color=theme()["info"], width=1.4, dash="dash"),
        )
    )
    if cur_prob == cur_prob:
        fig.add_trace(
            go.Scatter(
                x=[cur_prob],
                y=[pos_now],
                mode="markers+text",
                marker=dict(
                    size=13,
                    color=theme()["up"],
                    symbol="diamond",
                    line=dict(color=theme()["card"], width=1.5),
                ),
                text=[f" 今天 p={cur_prob:.3f} → {pos_now:.0%}"],
                textposition="top right",
                textfont=dict(size=11.5),
                name="当前信号",
                showlegend=False,
            )
        )
    fig.add_vline(x=0.5, line=dict(color=theme()["neutral"], width=1, dash="dot"))
    style_fig(
        fig,
        "概率略高于 0.5 时，理性仓位只有个位数百分比 —— 这就是「模型有信号」的真实含义",
        height=380,
        legend=True,
    )
    fig.update_xaxes(title_text="预测上涨概率", tickformat=".0%")
    fig.update_yaxes(title_text="仓位", tickformat=".0%", range=[-0.03, 1.05])
    chart(fig, key="risk_map")

    # ---------------------------------------------------------- 置信度分层
    section("置信度分层与选择性下注", "用「不确定就不说话」换准确率 —— 唯一诚实的提准确率手段")
    t1, t2 = st.columns([1, 1.15], gap="medium")
    conf_tbl = audit.confidence_subset_table(prob, y_true, n_buckets=5)
    sel_tbl = audit.selective_accuracy_table(prob, y_true)

    with t1:
        st.markdown("**信心分组（等频分桶，Q5 最有把握）**")
        if conf_tbl.empty:
            note("样本不足，无法分层。", "warn")
        else:
            st.dataframe(
                conf_tbl,
                width="stretch",
                hide_index=True,
                column_config={
                    "覆盖率": st.column_config.NumberColumn(format="percent"),
                    "方向准确率": st.column_config.NumberColumn(format="%.3f"),
                    "层内基线": st.column_config.NumberColumn(format="%.3f"),
                    "超额": st.column_config.NumberColumn(format="%+.3f"),
                },
            )
    with t2:
        st.markdown("**只对最自信的前 k% 下注**")
        if sel_tbl.empty:
            note("样本不足，无法计算。", "warn")
        else:
            st.dataframe(
                sel_tbl,
                width="stretch",
                hide_index=True,
                column_config={
                    "下注比例": st.column_config.NumberColumn(format="percent"),
                    "方向准确率": st.column_config.NumberColumn(format="%.3f"),
                    "该子集基线": st.column_config.NumberColumn(format="%.3f"),
                    "超额": st.column_config.NumberColumn(format="%+.3f"),
                    "平均置信度": st.column_config.NumberColumn(format="%.3f"),
                },
            )
    note(
        "看这张表要**同时看两列**：<code>方向准确率</code> 与 <code>超额</code>。"
        "如果缩到 10% 覆盖率时准确率涨到 65%、但超额只有 0.5%，"
        "那说明这 65% 主要是「这段时间本来就在涨」，不是模型的本事。"
        "<b>只有超额为正、且随覆盖率下降而单调上升，才说明模型的排序能力是真的。</b>",
        "warn",
    )

    # ---------------------------------------------------------- 概率分布
    section("概率分布：模型有多「敢说话」", "校准后的概率会被压向 0.5 —— 这才是诚实的样子")
    raw = res.wf.raw_predictions if res.wf.raw_predictions is not None else prob
    t = theme()
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.09, subplot_titles=("校准前", "校准后")
    )
    for col, (tag, series, color) in enumerate(
        [("校准前", raw, t["info"]), ("校准后", prob, t["accent"])], start=1
    ):
        fig.add_trace(
            go.Histogram(
                x=series, nbinsx=40, marker_color=color, opacity=0.82, name=tag, showlegend=False
            ),
            row=1,
            col=col,
        )
        fig.add_vline(x=0.5, line=dict(color=t["neutral"], width=1, dash="dot"), row=1, col=col)
        rng = float(series.max() - series.min())
        fig.add_annotation(
            x=0.5,
            y=1.14,
            xref=f"x{'' if col == 1 else col} domain",
            yref=f"y{'' if col == 1 else col} domain",
            showarrow=False,
            text=f"区间宽度 {rng:.3f}｜标准差 {float(series.std()):.4f}",
            font=dict(size=11, color=t["muted"]),
        )
    style_fig(fig, "校准把「假装很自信」压回了真实的不确定性", height=330)
    fig.update_xaxes(title_text="预测概率", tickformat=".2f")
    fig.update_yaxes(title_text="样本数")
    chart(fig, key="risk_prob_dist")
    note(
        f"校准后概率集中在 <b>0.5 ± {float((prob - 0.5).abs().quantile(0.95)):.3f}</b> 之间，"
        f"95% 的预测都落在这么窄的一条带里。这不是 bug，而是<b>数据的真实信息量</b>："
        "如果一只基金明天涨跌真的接近随机，那么任何诚实的模型都只能给出接近 50% 的概率。"
        "反过来，任何敢给出 0.9 的模型，要么在做梦，要么在作弊。",
        "bad",
    )

    # ---------------------------------------------------------- 波动率与压力
    section("波动率状态与压力测试", "同样的仓位，在不同波动环境下是完全不同的风险")
    nav = res.nav.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    daily = nav.sort_values("date").set_index("date")["nav"].pct_change()
    vol20 = daily.rolling(20, min_periods=10).std() * np.sqrt(244)
    cur_vol = float(vol20.iloc[-1]) if len(vol20.dropna()) else float("nan")

    vol_hist = vol20.dropna()
    vol_rank = (
        float((vol_hist < cur_vol).mean() * 100)
        if vol_hist.size and cur_vol == cur_vol
        else float("nan")
    )

    if vol_hist.size >= 60 and cur_vol == cur_vol:
        try:
            buckets = pd.qcut(vol_hist, 5, labels=False, duplicates="drop")
            cur_bucket = int(buckets.iloc[-1])
            same_idx = buckets[buckets == cur_bucket].index
            same_fwd = fwd.reindex(fwd.index.intersection(same_idx)).dropna()
        except ValueError:
            same_fwd = pd.Series(dtype=float)
    else:
        same_fwd = pd.Series(dtype=float)

    kpi_row(
        [
            dict(
                label="当前年化波动率（20日）",
                value=fpct(cur_vol, 1) if cur_vol == cur_vol else "—",
                tone="flat",
                sub=f"处于历史第 {vol_rank:.0f} 分位" if vol_rank == vol_rank else "",
            ),
            dict(
                label="同波动率档位样本",
                value=fint(len(same_fwd)),
                tone="flat",
                sub="用于估条件收益分布的历史天数",
            ),
            dict(
                label="该档位下未来收益中位数",
                value=fpct(float(same_fwd.median()), 2, sign=True) if len(same_fwd) else "—",
                tone=tone_of(float(same_fwd.median())) if len(same_fwd) else "flat",
                sub=f"持有 {res.config.horizon} 个交易日",
            ),
            dict(
                label="该档位下跌概率",
                value=fpct(float((same_fwd <= 0).mean()), 1) if len(same_fwd) else "—",
                tone="down" if len(same_fwd) and (same_fwd <= 0).mean() > 0.5 else "up",
                sub="条件于当前波动环境",
            ),
        ],
        cols=4,
    )

    if len(same_fwd) >= 20:
        qs = [0.05, 0.25, 0.5, 0.75, 0.95]
        vals = [float(same_fwd.quantile(q)) for q in qs]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["P5", "P25", "中位", "P75", "P95"],
                y=vals,
                marker_color=[theme()["down"] if v < 0 else theme()["up"] for v in vals],
                text=[f"{v * 100:+.1f}%" for v in vals],
                textposition="outside",
                textfont=dict(size=11.5),
                showlegend=False,
            )
        )
        style_fig(
            fig,
            f"历史「同档波动率」环境下，未来 {res.config.horizon} 个交易日收益的分位分布",
            height=340,
        )
        fig.update_yaxes(title_text="区间收益", tickformat=".1%")
        chart(fig, key="risk_cond")
        note(
            "把「未来收益的整体分布」按当前波动率档位**条件化**之后，"
            "就能看到下行尾部到底有多长（P5）。"
            "如果一个策略在最差的 5% 情形里会亏掉两位数百分比，"
            "那么它的仓位上限就不该超过 50%——不管你有多相信自己的模型。",
            "info",
        )

    st.markdown("**压力测试：当前仓位下的直接冲击**")
    stress = pd.DataFrame(
        {
            "情景": ["温和回调", "明显回调", "深度回调", "极端下跌", "系统性风险"],
            "标的下行": [-0.03, -0.05, -0.08, -0.10, -0.20],
        }
    )
    # 转成百分数数值（×100），printf 格式才能带正负号
    stress["当前仓位 " + f"{pos_now:.0%} 的损失"] = stress["标的下行"] * pos_now * 100
    st.dataframe(
        stress,
        width="stretch",
        hide_index=True,
        column_config={
            "标的下行": st.column_config.NumberColumn(format="percent"),
            stress.columns[2]: st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )
    note(
        f"仓位 <b>{pos_now:.0%}</b> 意味着标的跌 10% 时你亏 "
        f"<b>{10 * pos_now:.1f}%</b>。<br>"
        "这一步才是风险管理真正的内容：<b>先决定你能承受多大亏损，再倒推仓位</b>，"
        "而不是先算出「该持多少」，再去承受结果。"
        "本工具给的是概率，仓位和风险承受度必须由你自己决定。",
        "warn",
    )


# ================================================================= 精度审计
def render_audit(res) -> None:
    """正面回答「准确率为什么做不到 90%」，并把证据摊开。"""
    frame = res.dataset.frame
    prob = res.wf.predictions
    y_true = frame.loc[prob.index, "label"].astype(int)

    section("这份模型能做到多准？", "先用一句话给出答案，再给出全部证据")
    capa = res.capability or audit.capability_report(prob, y_true)
    need_auc = capa.get("所需AUC", audit.accuracy_to_auc(0.90))
    auc = capa.get("auc", float("nan"))
    edge = capa.get("acc_edge", float("nan"))
    upper = capa.get("理论准确率上界", float("nan"))

    note(
        f"<b>结论：在公募基金日频净值上，诚实的准确率上限就在 50%~55%，"
        f"做不到 90%。</b><br>"
        f"本次实测 AUC = <span class='hi'>{auc:.3f}</span>，"
        f"对应理论最高准确率 <span class='hi'>{upper:.3f}</span>；"
        f"而准确率 <b>90% 需要 AUC ≈ {need_auc:.3f}</b>，"
        f"缺口 <b>{capa.get('AUC缺口', float('nan')):.3f}</b>。<br>"
        "这不是调参问题，是<b>信息问题</b>——日频价格里根本不存在这么多可提取的信息。"
        "下面四项证据全部可复现、可自己验证。",
        "bad",
    )

    kpi_row(
        [
            dict(
                label="实测方向准确率",
                value=fpct(capa.get("accuracy", float("nan")), 2),
                tone="flat",
                sub="滚动前向验证的样本外结果",
            ),
            dict(
                label="多数类基线",
                value=fpct(capa.get("majority_acc", float("nan")), 2),
                tone="flat",
                sub="「永远猜多数类」什么模型都不用",
            ),
            dict(
                label="超额（准确率 − 基线）",
                value=fpct(edge, 2, sign=True),
                tone="up" if edge == edge and edge > 0.02 else "down",
                sub="这才是模型真实贡献；>+2% 才算有边缘",
            ),
            dict(
                label="实测 AUC",
                value=fnum(auc, 3),
                tone="flat",
                sub="0.5 = 无排序能力；1.0 = 完美",
            ),
            dict(
                label="该 AUC 的理论准确率上界",
                value=fpct(upper, 2),
                tone="accent",
                sub="即使阈值调到最优也超不过这个值",
            ),
            dict(
                label="达到 90% 所需的 AUC",
                value=fnum(need_auc, 3),
                tone="accent",
                sub="换算关系见下方曲线",
            ),
        ],
        cols=3,
    )

    # ---------------------------------------------------------- 理论换算曲线
    section("证据一：AUC → 准确率 的换算关系", "准确率对 AUC 极其敏感，它是个「门槛型」指标")
    curve_df = audit.auc_accuracy_curve(0.50, 0.995, 120)
    t = theme()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve_df["auc"],
            y=curve_df["accuracy"],
            mode="lines",
            line=dict(color=t["info"], width=2.6),
            name="理论准确率上界",
        )
    )
    fig.add_hline(
        y=0.90,
        line=dict(color=t["up"], width=1.6, dash="dash"),
        annotation_text="90% 目标线",
        annotation_position="bottom right",
        annotation_font=dict(size=11.5, color=t["up"]),
    )
    if need_auc < 0.995:
        fig.add_vline(
            x=need_auc,
            line=dict(color=t["accent"], width=1.4, dash="dot"),
            annotation_text=f"需 AUC {need_auc:.3f}",
            annotation_position="top left",
            annotation_font=dict(size=11.5, color=t["accent"]),
        )
    if auc == auc:
        fig.add_trace(
            go.Scatter(
                x=[auc],
                y=[upper],
                mode="markers+text",
                marker=dict(
                    size=13, color=t["up"], symbol="x", line=dict(color=t["card"], width=1.6)
                ),
                text=[f" 本次实测 AUC={auc:.3f} → 上界 {upper:.3f}"],
                textposition="top left",
                textfont=dict(size=11.5),
                name="本次实测",
            )
        )
    style_fig(
        fig,
        "把 AUC 从 0.52 提到 0.60，准确率只从 51% 涨到 57%；90% 那道线需要 AUC 0.965",
        height=400,
        legend=True,
    )
    fig.update_xaxes(title_text="AUC", tickformat=".2f")
    fig.update_yaxes(title_text="最优判定准确率", tickformat=".0%", range=[0.45, 1.0])
    chart(fig, key="audit_curve")
    note(
        "推导：设正负类得分分别为 <code>N(m,1)</code> 与 <code>N(0,1)</code>，"
        "则 <code>AUC = Φ(m/√2)</code>，最优阈值在 <code>m/2</code>，"
        "此时准确率 <code>= Φ(Φ⁻¹(AUC)/√2)</code>。"
        "<br><b>所以「准确率 90%」等价于「AUC 0.965」，"
        "意思是市场必须几乎确定性地可预测</b>——这显然不成立，"
        "否则所有量化基金都会是印钞机。",
        "info",
    )

    # ---------------------------------------------------------- 复杂度地形
    section("证据二：过拟合与欠拟合的完整地形图", "把「准确率刷到 90%」的代价现场演示一遍")
    X = frame[res.dataset.feature_names]
    y = frame["label"].astype(int)
    code = res.config.fund_code
    with st.status("正在扫描 5 档模型复杂度（每档 5 次滚动前向训练）…", expanded=False) as status:
        path = _cached_complexity_path(
            code, int(res.config.horizon), X, y, int(res.config.n_splits)
        )
        status.update(label="复杂度扫描完成", state="complete")

    if path.empty:
        note("复杂度扫描失败，已跳过。", "warn")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=path["配置"],
                y=path["过拟合差距"],
                name="过拟合差距（训练−样本外）",
                marker_color=t["neutral"],
                opacity=0.55,
                text=[f"{v:+.2f}" for v in path["过拟合差距"]],
                textposition="outside",
                textfont=dict(size=11),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path["配置"],
                y=path["训练准确率"],
                mode="lines+markers",
                name="训练集准确率",
                line=dict(color=t["up"], width=2.6),
                marker=dict(size=9),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path["配置"],
                y=path["样本外准确率"],
                mode="lines+markers",
                name="样本外准确率",
                line=dict(color=t["accent"], width=2.8),
                marker=dict(size=9),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path["配置"],
                y=path["多数类基线"],
                mode="lines",
                name="多数类基线",
                line=dict(color=t["down"], width=1.5, dash="dot"),
            )
        )
        style_fig(
            fig,
            "训练准确率一路冲到 100%，样本外却始终趴在 50% —— 这就是过拟合的全貌",
            height=420,
            legend=True,
        )
        fig.update_yaxes(title_text="准确率", tickformat=".0%", range=[0.45, 1.06])
        chart(fig, key="audit_overfit")
        st.dataframe(
            path,
            width="stretch",
            hide_index=True,
            column_config={
                "训练准确率": st.column_config.NumberColumn(format="%.3f"),
                "样本外准确率": st.column_config.NumberColumn(format="%.3f"),
                "过拟合差距": st.column_config.NumberColumn(format="%+.3f"),
                "样本外AUC": st.column_config.NumberColumn(format="%.3f"),
                "多数类基线": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        gap_max = float(path["过拟合差距"].max())
        note(
            f"<b>怎么读这张图：</b><br>"
            f"① <b>欠拟合区</b>（左侧）：训练 55%、样本外 52% —— 两者都低，"
            f"说明模型太弱、连训练集都学不动。这时候应该<b>加特征、加复杂度</b>。<br>"
            f"② <b>过拟合区</b>（右侧）：训练 <b>{float(path['训练准确率'].max()) * 100:.1f}%</b>、"
            f"样本外 <b>{float(path['样本外准确率'].iloc[-1]) * 100:.1f}%</b>，"
            f"差距高达 <b>{gap_max * 100:.1f} 个百分点</b>。模型把噪声全背下来了，"
            f"实盘等于随机。<br>"
            f"③ <b>关键事实</b>：无论怎么加复杂度，"
            f"样本外准确率始终在 <b>{path['样本外准确率'].min() * 100:.1f}%~"
            f"{path['样本外准确率'].max() * 100:.1f}%</b> 之间摆动，"
            f"样本外 AUC 也只有 <b>{path['样本外AUC'].min():.3f}~{path['样本外AUC'].max():.3f}</b>。"
            f"<span class='hi'>要让「准确率 ≥ 90%」这个指标变绿，唯一的路就是把训练集准确率"
            f"误当成最终指标——而那正是过拟合的定义。</span>",
            "warn",
        )

    # ---------------------------------------------------------- 泄漏审计
    section("证据三：泄漏审计", "高准确率无法证明模型有效，只有独立的泄漏审计才能")
    with st.status("正在运行泄漏审计（三组对照各一次滚动前向）…", expanded=False) as status:
        lk = _cached_leakage_audit(
            code, int(res.config.horizon), frame, tuple(res.dataset.feature_names)
        )
        status.update(label="泄漏审计完成", state="complete")

    if lk.empty:
        note("泄漏审计失败，已跳过。", "warn")
    else:
        st.dataframe(
            lk,
            width="stretch",
            hide_index=True,
            column_config={
                "准确率": st.column_config.NumberColumn(format="%.3f"),
                "AUC": st.column_config.NumberColumn(format="%.3f"),
                "多数类基线": st.column_config.NumberColumn(format="%.3f"),
                "超额": st.column_config.NumberColumn(format="%+.3f"),
            },
        )
        leak_row = lk[lk["变体"].str.contains("答案入特征")]
        leak_acc = float(leak_row["准确率"].iloc[0]) if len(leak_row) else float("nan")
        note(
            f"② 「答案入特征」的准确率是 <b>{leak_acc:.3f}</b> —— "
            "注意这是在**严格滚动前向验证**下跑出来的，时间没打乱、训练集严格早于测试集。"
            "所以<b>「我做了时序验证，准确率 95%」完全不能证明模型有效</b>："
            "只要特征里混进了未来信息，任何验证协议都拦不住。<br>"
            "③ 「未来函数标签」的准确率与 ① 几乎一样 —— "
            "<b>准确率这个指标根本发现不了未来函数</b>，"
            "只有回测收益曲线会虚高。这就是本项目为什么把「标签时点」写成独立测试"
            "（<code>test_no_lookahead_bias</code>）锁死，而不是靠人肉检查。",
            "bad",
        )

    # ---------------------------------------------------------- 高准确率的手段
    section("那「准确率 90%」都是怎么来的？", "五种常见做法，以及每一种的代价")
    tricks = pd.DataFrame(
        [
            {
                "做法": "把训练集准确率当作模型准确率",
                "能到多少": "99%~100%",
                "代价": "自欺。样本外回到 50%，实盘必亏",
            },
            {
                "做法": "打乱时间做随机 K 折交叉验证",
                "能到多少": "55%~75%",
                "代价": "未来信息泄漏。金融时序强自相关，洗牌=用未来训过去",
            },
            {
                "做法": "把未来信息混进特征（最隐蔽）",
                "能到多少": "接近 100%",
                "代价": "任何验证协议都拦不住，只有泄漏审计能发现",
            },
            {
                "做法": "标签或执行时点算错（未来函数）",
                "能到多少": "几乎不变",
                "代价": "准确率看不出来，但回测收益虚高几十个百分点",
            },
            {
                "做法": "把标签改成罕见事件 + 恒猜多数类",
                "能到多少": "80%~95%",
                "代价": "模型什么都没做，超额 = 0。准确率高只是因为事件不平衡",
            },
            {
                "做法": "选择性预测：只在极少数高置信样本上下注",
                "能到多少": "60%~70%（覆盖率仅 5%~10%）",
                "代价": "诚实但覆盖率低；且必须同时看「超额」，否则可能只是行情好",
            },
        ]
    )
    st.dataframe(tricks, width="stretch", hide_index=True)
    note(
        "表里前五行都是**幻觉**，只有最后一行是正当做法。"
        "所以一个诚实的量化项目应该报告的不是「准确率」，而是一组指标："
        "<code>超额(acc_edge)</code>、<code>AUC + 置信区间</code>、"
        "<code>IC</code>、<code>Brier / ECE</code>、以及<b>扣费后</b>的收益。"
        "本项目六项全给。",
        "info",
    )

    # ---------------------------------------------------------- 选择性预测
    section(
        "如果一定要高准确率：选择性预测", "把「什么时候开口」也纳入模型 —— 覆盖率的代价一目了然"
    )
    sel = audit.selective_accuracy_table(prob, y_true)
    if sel.empty:
        note("样本不足，无法计算选择性预测。", "warn")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sel["下注比例"],
                y=sel["方向准确率"],
                mode="lines+markers",
                name="方向准确率",
                line=dict(color=t["accent"], width=2.8),
                marker=dict(size=9),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sel["下注比例"],
                y=sel["该子集基线"],
                mode="lines+markers",
                name="该子集多数类基线",
                line=dict(color=t["down"], width=2, dash="dot"),
                marker=dict(size=7),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sel["下注比例"],
                y=sel["超额"],
                mode="lines+markers",
                name="超额（准确率−基线）",
                line=dict(color=t["info"], width=2),
                marker=dict(size=7),
                yaxis="y2",
            )
        )
        style_fig(
            fig,
            "覆盖率越低、准确率越高 —— 但超额是否同步上升，才是判断真假的唯一标准",
            height=380,
            legend=True,
        )
        fig.update_xaxes(title_text="下注比例（占全部交易日）", tickformat=".0%")
        fig.update_yaxes(title_text="准确率", tickformat=".0%", range=[0.45, 0.75])
        fig.update_layout(
            yaxis2=dict(
                title="超额",
                overlaying="y",
                side="right",
                tickformat="+.1%",
                gridcolor=t["grid"],
                showgrid=False,
            )
        )
        chart(fig, key="audit_selective")
        st.dataframe(
            sel,
            width="stretch",
            hide_index=True,
            column_config={
                "下注比例": st.column_config.NumberColumn(format="percent"),
                "方向准确率": st.column_config.NumberColumn(format="%.3f"),
                "该子集基线": st.column_config.NumberColumn(format="%.3f"),
                "超额": st.column_config.NumberColumn(format="%+.3f"),
                "平均置信度": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    # ---------------------------------------------------------- 结论
    section("本项目的立场", "为什么我们不把准确率做到 90%")
    note(
        "<b>因为那只有两种实现方式：作弊，或者过拟合。</b><br>"
        "把 90% 写进需求，等于要求模型发现一个<b>统计上不存在</b>的规律。"
        "面对这种需求，一个负责任的做法不是去满足它，而是：<br>"
        "① 用可复现的实验说清楚为什么做不到（就是上面三条证据）；<br>"
        "② 把真正能提升的部分做到极致 —— "
        "<b>概率校准</b>（让概率可当仓位用）、"
        "<b>净化间隔</b>（切断标签重叠泄漏）、"
        "<b>多模型集成</b>（降低方差）、"
        "<b>成本建模</b>（把真实摩擦算进去）；<br>"
        "③ 给出一组<b>诚实的指标</b>，让使用者自己判断这个信号值不值得用。",
        "ok",
    )
    note(
        f"顺带一提：本次运行里，模型准确率 "
        f"<code>{capa.get('accuracy', float('nan')):.3f}</code> 对多数类基线 "
        f"<code>{capa.get('majority_acc', float('nan')):.3f}</code>，"
        f"超额 <code>{edge:+.3f}</code>。"
        + (
            "<b>超额不到 +2%，按本项目的判据，这个信号不具备可用边缘。</b>"
            if edge == edge and edge <= 0.02
            else "<b>超额超过 +2%，但请务必结合 Bootstrap 置信区间一起看。</b>"
        )
        + " 这不是坏消息 —— 在弱式有效市场里，这本来就是最常见的结果。",
        "info",
    )


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
        fig = go.Figure(
            go.Histogram(
                x=ret,
                nbinsx=80,
                marker_color=t["info"],
                opacity=0.85,
                hovertemplate="%{x:.2%}<br>%{y} 天<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color=t["neutral"], line_width=1)
        style_fig(fig, "日收益率分布：是否对称、尾部有多厚", 330)
        fig.update_xaxes(title="日收益率", tickformat=".1%")
        fig.update_yaxes(title="天数", rangemode="tozero")
        chart(fig)
    with c2:
        roll = df.set_index("date")["日收益率"].rolling(60).std() * np.sqrt(244)
        fig = go.Figure(
            go.Scatter(
                x=roll.index,
                y=roll,
                mode="lines",
                line=dict(color=t["accent"], width=1.6),
                fill="tozeroy",
                fillcolor="rgba(201,134,26,0.14)",
                hovertemplate="%{x|%Y-%m-%d}<br>年化波动 %{y:.1%}<extra></extra>",
            )
        )
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
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.64, 0.36],
            vertical_spacing=0.08,
            subplot_titles=("净值归一对比", "相对强弱（基金 / 基准）"),
        )
        fig.add_trace(
            go.Scatter(
                x=fund_lvl.index,
                y=fund_lvl,
                name="基金",
                line=dict(color=t["up"], width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>基金 %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bench_lvl.index,
                y=bench_lvl,
                name="基准指数",
                line=dict(color=t["info"], width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>基准 %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rel.index,
                y=rel,
                name="相对强弱",
                line=dict(color=t["accent"], width=1.6),
                fill="tozeroy",
                fillcolor="rgba(201,134,26,0.12)",
                hovertemplate="%{x|%Y-%m-%d}<br>比值 %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
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
            g = sub.groupby("bin", observed=True).agg(
                ret=("fwd_ret", "mean"), n=("fwd_ret", "size")
            )
            labels = [f"Q{i + 1}" for i in range(len(g))]
            fig = go.Figure(
                go.Bar(
                    x=labels,
                    y=g["ret"],
                    marker_color=[t["up"] if v >= 0 else t["down"] for v in g["ret"]],
                    text=[
                        f"{fpct(v, 2, sign=True)}<br>n={int(n)}"
                        for v, n in zip(g["ret"], g["n"], strict=True)
                    ],
                    textposition="outside",
                    hovertemplate="%{x}<br>平均未来收益 %{y:.3%}<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color=t["neutral"], line_width=1)
            fig.add_hline(
                y=float(sub["fwd_ret"].mean()),
                line_dash="dash",
                line_color=t["neutral"],
                annotation_text="整体均值",
            )
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
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="相关系数", thickness=12),
            hovertemplate="%{y} ↔ %{x}<br>相关 %{z:.2f}<extra></extra>",
        )
    )
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
        mime="text/csv",
        width="stretch",
    )
    wf_tbl = res.wf.predictions.to_frame("prob").join(frame[["label", "fwd_ret", "nav"]])
    c2.download_button(
        "⬇️ 样本外预测 (CSV)",
        data=wf_tbl.to_csv().encode("utf-8-sig"),
        file_name=f"fund_signal_{res.config.fund_code}_oos.csv",
        mime="text/csv",
        width="stretch",
    )
    c3.download_button(
        "⬇️ 各折指标 (CSV)",
        data=pd.DataFrame(res.wf.fold_metrics).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"fund_signal_{res.config.fund_code}_folds.csv",
        mime="text/csv",
        width="stretch",
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
# ================================================================= 事件与风险
# 这一页的设计目标与别的页签不同：**先用一屏把结论说清楚，细节全部折叠**。
# 因为「加新闻/事件让预测更准」是最容易走偏的诉求，页面的首要任务是先把
# 实测结论摆在最上面，再给证据，最后才给原始数据。
@st.cache_data(ttl=600, show_spinner=False)
def _cached_news(limit: int) -> pd.DataFrame:
    return ev.fetch_news(limit=limit)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_econ_calendar(day: str) -> pd.DataFrame:
    return ev.fetch_econ_calendar(day)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_placebo(nav: pd.DataFrame, mode: str) -> dict:
    """安慰剂检验：把事件日随机化后重算，看真实值是否只是日历巧合。

    ``nav`` 直接参与缓存键，因此换基金、换区间会自动失效重算。
    """
    if nav is None or nav.empty:
        return {}
    n = nav.copy()
    n["date"] = pd.to_datetime(n["date"])
    first, last = n["date"].iloc[0].date(), n["date"].iloc[-1].date()
    cal = ev.scheduled_events(first, last, trade_dates=n["date"])
    return ev.placebo_test(
        n,
        cal,
        before=0,
        after=1,
        horizon=1,
        min_importance=3,
        n_draws=200,
        max_shift=20,
        mode=mode,
    )


def _importance_badge(level: int) -> str:
    label = {3: "高", 2: "中", 1: "低"}.get(int(level), "-")
    cls = {3: "hot", 2: "on", 1: ""}.get(int(level), "")
    return f'<span class="fs-badge {cls}">{label}</span>'


def render_events(res) -> None:
    nav = res.nav
    if nav is None or nav.empty:
        st.info("没有净值数据，无法做事件对齐分析。")
        return

    nav = nav.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    first, last = nav["date"].iloc[0].date(), nav["date"].iloc[-1].date()

    cal = ev.scheduled_events(first, last, trade_dates=nav["date"])
    imp = ev.event_impact(nav, cal, before=0, after=1, horizon=1, min_importance=3)

    # 安慰剂检验提前算——最终判断必须由它裁决（见 events.verdict_text）
    try:
        pl_shift = _cached_placebo(nav, "shift")
        pl_uni = _cached_placebo(nav, "uniform")
    except Exception as err:  # noqa: BLE001
        pl_shift, pl_uni = {}, {}
        st.caption(f"安慰剂检验失败：{err}")
    pl_best = pl_uni or pl_shift

    # ---------------------------------------------------------- 结论先行
    section("一句话结论", "先看这里，下面的细节都是它的证据")
    note(f"<b>{ev.verdict_text(imp, pl_best)}</b>", "warn")

    merged = imp[imp["事件"].str.startswith("★")] if not imp.empty else pd.DataFrame()
    cover = float(merged["覆盖率"].iloc[0]) if not merged.empty else float("nan")
    vol_ratio = float(merged["波动比"].iloc[0]) if not merged.empty else float("nan")
    win_diff = float(merged["胜率差"].iloc[0]) if not merged.empty else float("nan")
    vol_pct = float(pl_best.get("p_波动", float("nan")))

    up = ev.upcoming_events(days=14, min_importance=2, trade_dates=nav["date"])
    kpi_row(
        [
            dict(label="未来 14 天重要事件", value=fint(len(up)), sub="重要性 ≥ 中", tone="flat"),
            dict(
                label="事件窗口覆盖率",
                value=fpct(cover),
                sub="超过 50% 则对照失效",
                tone="flat",
                help_text="高重要性事件日及其次一交易日占全部交易日的比例",
            ),
            dict(
                label="事件窗口波动比",
                value=f"{vol_ratio:.2f}×",
                sub=f"随机化后分位 {vol_pct * 100:.0f}%（≈50% 即无区别）",
                tone="flat",
                help_text="与非事件期相比。< 1 表示波动更小，但要看安慰剂分位："
                "若随机日期也能得到同样读数，它就不是事件效应",
            ),
            dict(
                label="事件方向胜率差",
                value=f"{win_diff * 100:+.1f}pp",
                sub="≈0 说明方向无差别",
                tone=tone_of(win_diff),
            ),
        ]
    )

    # ---------------------------------------------------------- 未来有什么大事
    section("接下来有什么大事", "这些日期是**算出来的**，不依赖任何新闻源，所以可以提前知道")
    span = st.radio(
        "查看范围",
        [7, 14, 30],
        index=1,
        horizontal=True,
        format_func=lambda d: f"未来 {d} 天",
        key="fs_ev_span",
    )
    ups = ev.upcoming_events(days=span, min_importance=2, trade_dates=nav["date"])
    if ups.empty:
        st.caption("这个区间内没有重要性 ≥ 中的日程事件。")
    else:
        body = (
            "<table class='fs-tbl'><thead><tr><th>日期</th><th>事件</th><th>类别</th>"
            "<th>重要性</th><th>说明</th></tr></thead><tbody>"
        )
        for _, r in ups.iterrows():
            body += (
                f"<tr><td>{r['date'].strftime('%m-%d')} "
                f"<span style='opacity:.6'>{r['date'].strftime('%a')}</span></td>"
                f"<td><b>{r['name']}</b></td><td>{r['category']}</td>"
                f"<td>{_importance_badge(r['importance'])}</td>"
                f"<td style='opacity:.75'>{r['note']}</td></tr>"
            )
        st.markdown(_flat(body + "</tbody></table>"), unsafe_allow_html=True)
    note(
        "「日历事件」是<b>唯一可以合法进入模型</b>的一类事件——它的时间点事先已知，"
        "不依赖任何未来信息。而<b>当天看到的新闻不能进模型</b>：新闻的「重要性」"
        "只有事后才看得清，用当天快讯挑事件再回看收益，属于事后诸葛。<br>"
        "但「可以合法使用」不等于「有用」——它到底有没有用，由下面的实测与安慰剂检验裁决。",
        "info",
    )

    # ---------------------------------------------------------- 实时事件（折叠）
    with st.expander("📰 当下正在发生什么（实时快讯 / 经济日历，需联网）", expanded=False):
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown("**市场快讯**")
            try:
                news = _cached_news(25)
            except Exception as err:  # noqa: BLE001
                news = pd.DataFrame()
                st.caption(f"获取失败：{err}")
            if news.empty:
                st.caption("无数据（可能断网或数据源波动）。")
            else:
                for _, r in news.head(12).iterrows():
                    st.markdown(
                        _flat(
                            f"<div style='padding:6px 0;border-bottom:1px solid rgba(128,128,128,.18)'>"
                            f"<span style='opacity:.55;font-size:.78rem'>{r['time'][-8:]} · "
                            f"{r['source']}</span><br>{r['title']}</div>"
                        ),
                        unsafe_allow_html=True,
                    )
        with right:
            st.markdown("**经济日历（含市场预期值）**")
            try:
                ec = _cached_econ_calendar(date.today().strftime("%Y-%m-%d"))
            except Exception as err:  # noqa: BLE001
                ec = pd.DataFrame()
                st.caption(f"获取失败：{err}")
            if ec.empty:
                st.caption("当日无数据。")
            else:
                keep = ec[ec["重要性"] >= 2].head(14)
                if keep.empty:
                    st.caption("当日没有高重要性数据。")
                else:
                    st.dataframe(
                        keep[["时间", "地区", "事件", "公布", "预期", "前值"]],
                        width="stretch",
                        hide_index=True,
                    )
        note(
            "这两块是**解释与预警**用的——知道「今天为什么跌」和「明天有什么要公布」，"
            "但它不参与模型计算。原因见上面那条。",
            "info",
        )

    # ---------------------------------------------------------- 实测证据（折叠）
    with st.expander("📊 事件到底有没有用？看实测证据", expanded=False):
        if imp.empty:
            st.caption("样本不足，无法检验。")
        else:
            st.markdown(
                "把每个事件日及其**次一交易日**（即真正可成交的那一天）与非事件期对比。"
                "最重要的一列是**波动比**——它衡量风险，而不是方向。"
            )
            show = imp.copy()
            show["时间精度"] = (
                show["精度"].map({"exact": "发布日固定", "approx": "窗口（±2 天浮动）"}).fillna("—")
            )
            st.dataframe(
                show[
                    [
                        "事件",
                        "重要性",
                        "时间精度",
                        "样本数",
                        "覆盖率",
                        "窗口上涨概率",
                        "非事件期上涨概率",
                        "胜率差",
                        "窗口日波动",
                        "非事件期日波动",
                        "波动比",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "重要性": st.column_config.NumberColumn(format="%d"),
                    "覆盖率": st.column_config.NumberColumn(format="percent"),
                    "窗口上涨概率": st.column_config.NumberColumn(format="percent"),
                    "非事件期上涨概率": st.column_config.NumberColumn(format="percent"),
                    "胜率差": st.column_config.NumberColumn(format="percent"),
                    "窗口日波动": st.column_config.NumberColumn(format="%.4f"),
                    "非事件期日波动": st.column_config.NumberColumn(format="%.4f"),
                    "波动比": st.column_config.NumberColumn(format="%.2f"),
                },
            )

            # 把「看着像发现、其实是陷阱」的东西主动点出来
            traps = imp[
                (~imp["事件"].str.startswith("★"))
                & (imp["胜率差"].abs() >= 0.05)
                & (imp["样本数"] <= 200)
            ]
            if not traps.empty:
                names = "、".join(
                    f"{r['事件']}（{int(r['样本数'])} 天）" for _, r in traps.iterrows()
                )
                note(
                    f"⚠️ <b>警惕这几行</b>：{names} 的方向差异看着很大，但它们的日期在"
                    "日历上是固定的（政治局会议在 4/7/10/12 月、两会在 3 月、中央经济工作会议在 12 月），"
                    "所以**捕捉到的很可能是「季节性」，而不是「事件效应」**；样本只有几十天，"
                    "在十几个类别里挑出最极端的那个，本身就是多重比较下的过拟合。"
                    "这一类结论<b>不能当成信号使用</b>。",
                    "bad",
                )
            note(
                "纵向比较要非常小心：不同事件的窗口会重叠，而且「事件日」不是随机分配的。"
                "本表只适合读**波动比**这一个方向，不要拿它去构造交易规则。",
                "info",
            )

            # -------------------------------------------------- 安慰剂检验
            st.markdown("**安慰剂检验：把事件日随机化，还看得出来吗？**")
            st.markdown(
                "上面的「波动比 < 1」看着像一个发现。但事件日**不是随机分配的**——"
                "LPR 固定在每月 20 日、两会在 3 月、中央经济工作会议在 12 月。"
                "所以要把事件日随机化后重算一遍：如果随机日期也能得到同样的结果，"
                "那所谓「事件效应」就只是**日历位置**带来的假象。"
            )
            if pl_shift and pl_uni:
                prows = [
                    {
                        "随机化方式": "平移 ±20 个交易日（保留月内位置）",
                        "随机波动比中位数": pl_shift["随机波动比中位数"],
                        "5%~95% 区间": f"{pl_shift['随机波动比 P5']:.2f} ~ "
                        f"{pl_shift['随机波动比 P95']:.2f}",
                        "真实值": pl_shift["真实波动比"],
                        "真实值分位": pl_shift["p_波动"],
                    },
                    {
                        "随机化方式": "样本区间内均匀随机取日期（打破日历结构）",
                        "随机波动比中位数": pl_uni["随机波动比中位数"],
                        "5%~95% 区间": f"{pl_uni['随机波动比 P5']:.2f} ~ "
                        f"{pl_uni['随机波动比 P95']:.2f}",
                        "真实值": pl_uni["真实波动比"],
                        "真实值分位": pl_uni["p_波动"],
                    },
                ]
                st.dataframe(
                    pd.DataFrame(prows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "随机波动比中位数": st.column_config.NumberColumn(format="%.3f"),
                        "真实值": st.column_config.NumberColumn(format="%.3f"),
                        "真实值分位": st.column_config.NumberColumn(format="percent"),
                    },
                )
                in_middle = 0.05 < float(pl_uni["p_波动"]) < 0.95
                note(
                    ev.placebo_summary_text(pl_uni)
                    if in_middle
                    else ev.placebo_summary_text(pl_shift),
                    "bad" if in_middle else "warn",
                )
                note(
                    "注意这两种随机化的**随机波动比中位数本身也小于 1**——"
                    "说明「事件窗口波动比 < 1」这个读数里，本来就含有一部分来自"
                    "「少数日 vs 多数日」这种切分方式的偏差。真实值落在随机分布的"
                    "中部，因此它<b>不能</b>作为「事件改变了风险」的证据。",
                    "info",
                )

    # ---------------------------------------------------------- 怎么用
    section("所以，事件该怎么用", "这是本页唯一想让你记住的三件事")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        note(
            "<b>❌ 不要用来猜涨跌</b><br>实测三只基金的事件窗口方向差异全落在噪声里，"
            "而「拿当天新闻挑事件再回看收益」最容易靠未来函数刷出漂亮回测。",
            "bad",
        )
    with c2:
        note(
            "<b>❌ 也别指望它「降波动」</b><br>本页读到的「事件窗口波动比 &lt; 1」"
            "经不起安慰剂检验：把事件日随机化后，随机日期也能得到同样的读数。"
            "所以它不是事件效应，不能当成「事件期更安全」的理由。",
            "bad",
        )
    with c3:
        note(
            "<b>✅ 真正有用的是「日程可提前推算」</b><br>知道哪几天有数据公布、"
            "哪几天有会议，用来安排<b>什么时候下手</b>（避开数据公布前一次性重仓），"
            "以及事后解释「今天为什么动」。它提供的是信息，不是预测。",
            "ok",
        )


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
    particles_on = params.pop("particles", True)

    particle_background(enabled=bool(particles_on))

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

    tabs = st.tabs(
        [
            "📊 总览",
            "🎯 信号详情",
            "📅 事件与风险",
            "🧪 模型评估",
            "💰 回测",
            "🧭 风控与仓位",
            "🛡️ 精度审计",
            "🔍 数据探索",
            "📋 数据与导出",
        ]
    )
    with tabs[0]:
        render_overview(result)
    with tabs[1]:
        render_prediction(result)
    with tabs[2]:
        render_events(result)
    with tabs[3]:
        render_model(result)
    with tabs[4]:
        render_backtest(result)
    with tabs[5]:
        render_risk(result)
    with tabs[6]:
        render_audit(result)
    with tabs[7]:
        render_data(result)
    with tabs[8]:
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
