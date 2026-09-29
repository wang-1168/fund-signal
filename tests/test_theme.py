# -*- coding: utf-8 -*-
"""界面主题（亮 / 暗切换）的回归测试。

用 ``streamlit.testing.v1.AppTest`` 渲染真实的 ``app.py``，只渲染**欢迎页**
（不点「开始分析」），所以不联网、不训练，跑得很快。

守住的是这几条**容易悄悄坏掉**的约束：

1. 默认亮色，且注入的 CSS 变量来自 LIGHT 色板；
2. 偏好切成 dark 后，变量来自 DARK 色板，并且**同一轮**多注入了擦除覆盖层
   （晚一轮就等于动画卡帧）；
3. 没切换的那些轮次**不再**注入覆盖层（否则 DOM 里会堆垃圾、动画反复重播）；
4. 覆盖层的 ``animation-name`` 按切换次数奇偶交替 —— React 会复用这个 DOM 节点，
   只改属性不会重放动画，实测第 2 次之后的切换完全看不到擦除效果；
5. 侧边栏「外观」三项选择与真相源 ``fs_theme`` 双向同步；
6. 粒子画布的补间起点（``fs_theme_prev`` / ``fs_theme_shock``）**在同一轮内被消费掉**，
   所以那段颜色补间只会播一次。
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit", reason="界面测试需要 streamlit")
AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="需要 streamlit >= 1.28 的 AppTest"
).AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def _run(app, timeout: int = 180):
    app.run(timeout=timeout)
    return app


@pytest.fixture
def app():
    """一份全新的 AppTest（默认落在欢迎页）。"""
    at = AppTest.from_file(str(APP), default_timeout=180)
    return _run(at)


def _markdown(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def _css(at) -> str:
    """注入的全局样式（带 --bg: 的那一段）。"""
    return "\n".join(str(m.value) for m in at.markdown if "--bg:" in str(m.value))


def _wipe_count(at) -> int:
    """擦除覆盖层的**元素**数量。

    注意别写成 ``.fs-wipe`` —— ``_CSS`` 里本来就常驻着 ``.fs-wipe`` 的规则文本，
    要数的是 ``<div class="fs-wipe">`` 这个元素。
    """
    return _markdown(at).count('<div class="fs-wipe')


def _set_theme(at, pref: str):
    at.session_state["fs_theme"] = pref
    return _run(at)


def _radio_value(at, label: str = "外观"):
    for r in at.radio:
        if r.label == label:
            return r.value
    raise AssertionError(f"没找到「{label}」单选控件")


def _quick_button(at):
    for b in at.button:
        if b.key == "fs_theme_quick":
            return b
    raise AssertionError("没找到浮动主题按钮 fs_theme_quick")


# ------------------------------------------------------------------ 基础
def test_default_theme_is_light(app):
    css = _css(app)
    assert "--bg:#FFFFFF" in css
    assert "filter:none" in css  # 亮色不需要给数据网格加反色滤镜
    assert _wipe_count(app) == 0
    assert app.session_state["fs_theme_applied"] == "light"


def test_dark_palette_and_wipe_injected_same_round(app):
    _set_theme(app, "dark")
    css = _css(app)
    assert "--bg:#0D1117" in css, "暗色色板没生效"
    assert "invert(0.94)" in css, "暗色下数据网格的反色滤镜没生效"
    assert _wipe_count(app) == 1, "切换那一轮必须注入擦除覆盖层"
    assert "--fs-old:#FBFCFE" in _markdown(app), "覆盖层应该用**旧**色板打底"


def test_no_wipe_when_theme_unchanged(app):
    _set_theme(app, "dark")
    assert _wipe_count(app) == 1
    _run(app)  # 原样再跑一轮
    assert _wipe_count(app) == 0, "没切换就不该再注入覆盖层"
    assert _css(app).count("--bg:#0D1117") == 1


def test_wipe_animation_name_alternates(app):
    """连续切换必须交替 A / B 的 animation-name，否则 React 复用节点时动画不重播。"""
    seen = []
    for pref in ("dark", "light", "dark", "light"):
        _set_theme(app, pref)
        assert _wipe_count(app) == 1
        md = _markdown(app)
        seen.append("fs-wipe alt" in md)
    assert seen == [True, False, True, False] or seen == [False, True, False, True], seen


def test_tween_state_consumed_in_same_round(app):
    _set_theme(app, "dark")
    # sync_theme() 写入、particle_background() 弹出 —— 同一轮就被消费掉
    assert "fs_theme_prev" not in app.session_state
    assert "fs_theme_shock" not in app.session_state


def test_unknown_preference_falls_back_to_light(app):
    _set_theme(app, "霓虹粉")
    assert "--bg:#FFFFFF" in _css(app)
    assert app.session_state["fs_theme_applied"] == "light"


# ------------------------------------------------------------------ 控件联动
def test_sidebar_radio_synced_to_source_of_truth(app):
    assert _radio_value(app) == "light"
    _set_theme(app, "dark")
    assert _radio_value(app) == "dark", "真相源变了，侧边栏三项选择没跟上"


def test_quick_button_label_flips(app):
    assert _quick_button(app).label == "🌙", "亮色时按钮应提示「点了会变暗」"
    _set_theme(app, "dark")
    assert _quick_button(app).label == "☀️"


def test_auto_follows_streamlit_base(app):
    """auto 模式下按 Streamlit 自己的 base 走；本项目 config.toml 固定 light。"""
    _set_theme(app, "auto")
    assert "--bg:#FFFFFF" in _css(app)
    assert _radio_value(app) == "auto"
