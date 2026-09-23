---
name: Bug 报告
about: 报告一个可以复现的问题
title: "[Bug] "
labels: bug
assignees: ""
---

<!--
提交前请先确认：
1. 已经 `pip install -U akshare` 复现过（akshare 接口变动很频繁）；
2. 已经试过加 --no-cache 强制刷新数据；
3. 在 issue 列表里搜过同样的问题。
-->

## 问题描述

<!-- 一句话说清发生了什么。 -->

## 复现步骤

```bash
# 粘贴你实际执行的完整命令，例如：
python -m fund_signal --code 000001 --data akshare
```

1.
2.
3.

## 期望结果

<!-- 你认为应该发生什么。 -->

## 实际结果

<!-- 实际发生了什么。请粘贴完整报错堆栈，不要只贴最后一行。 -->

```text
（粘贴 traceback）
```

## 环境信息

<!-- 请完整填写，缺项会让排查变慢。 -->

| 项目 | 值 |
|---|---|
| 操作系统 | 例如 Windows 11 23H2 / Ubuntu 24.04 / macOS 15 |
| Python 版本 | 例如 3.13.12 |
| fund-signal 版本 | 例如 0.1.0（`pip show fund-signal`） |
| akshare 版本 | 例如 1.18.96 |
| 安装方式 | `pip install -e .` / `pip install -r requirements.txt` |
| 是否离线模式 | `--data local` 也复现吗？ |

## 相关标的（如适用）

- 基金代码：
- 数据源：akshare / local

## 补充信息

<!--
如果问题与「回测结果看起来不合理」有关，请额外说明：
- 你用的阈值、费率、horizon 分别是多少？
- 参考 CONTRIBUTING.md 的「未来函数」一节自查过了吗？

如果是数据接口报错，请运行以下命令并粘贴输出，会很有帮助：

    python -c "import akshare; print(akshare.__version__)"
-->
