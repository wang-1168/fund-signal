<div align="center">

# 📈 fund-signal

**公募基金量化信号分析框架**

从数据获取到样本外回测的完整链路，一套**方法论严谨**、**结果诚实**的开源工具。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-63%20passed-brightgreen.svg)](tests/)
[![CI](https://github.com/wang-1168/fund-signal/actions/workflows/ci.yml/badge.svg)](https://github.com/wang-1168/fund-signal/actions)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[快速开始](#-快速开始) · [核心特性](#-核心特性) · [方法论](#-方法论) · [常见问题](#-常见问题) · [免责声明](#-免责声明)

</div>

---

## ⚠️ 先说清楚：这个项目不承诺任何收益

市面上大量「基金预测」项目用随机划分训练集、忽略交易成本、不做样本外验证的方式，画出一条漂亮的回测曲线，然后暗示你可以据此赚钱。**本项目反其道而行**——它的默认输出长这样：

```
【样本外预测质量】（滚动前向验证，5 折）
  AUC         0.5286   （0.5 = 无预测能力）
  准确率      0.5149
  多数类基线  0.5018
  超额        0.0130   ✗ 未超过基线

【样本外回测】（阈值 0.5，成本 15bps）
               策略     买入持有
累计收益      -33.79%   26.14%
持仓日胜率     49.84%   ← 只看有仓位的日子
```

**持仓日胜率 49.84%，几乎是抛硬币。** 这是基金日频方向预测的真实水平，也是弱式有效市场假说下应有的结果。

那这个项目的价值在哪？

> 它演示了一套**完整的、经得起推敲的量化研究流程**，并且用可运行的代码证明了一件事：
> **「预测准确率略高于 50%」和「能赚钱」之间，隔着交易成本、风险暴露和统计显著性这三道墙。**

如果你在做量化方向的课题、课程设计、简历项目，或者单纯想搞清楚「为什么那么多基金预测模型都是骗人的」，这个项目就是你要的。

---

## ✨ 核心特性

| 特性 | 说明 |
|---|---|
| 🔒 **严格防未来函数** | 特征只用 T 日及之前的信息；`test_no_lookahead_bias` 用截断数据逐值比对，任何信息穿越都会让测试失败 |
| ⏱️ **真实的执行时序** | 标签以 **T+1 日净值**为买入价（基金 T 日晚才公布净值、T+1 日才能成交）。用 T 日净值当买入价会凭空多赚一天，是最常见的回测虚高来源 |
| 📊 **滚动前向验证** | walk-forward：每一折只用「该折之前」的数据训练。拒绝随机划分，拒绝 `TimeSeriesSplit` 以外的任何偷懒做法 |
| 💸 **计入交易成本** | 默认单边 15bps，仓位变动才计费。零成本回测毫无意义 |
| 🎯 **诚实的评估体系** | 同时输出 AUC、**多数类基线**、**持仓日胜率**、**信息系数 IC**——专门用来戳穿「看着还行」的幻觉 |
| 🔄 **实时数据能力** | 自动补齐最新交易日净值（含"当日净值未公布时回退到上一交易日"的处理）+ 盘中实时估值 |
| 🖥️ **交互式界面** | 6 个标签页、14 张可交互图表；阈值/成本/延迟滑块**实时重算回测**，无需重新训练 |
| 📦 **零配置可用** | 数据源 akshare 免费无需 token；仓库自带离线样例数据，**断网也能跑通全流程** |
| ✅ **63 个单元测试** | 全部离线运行，CI 秒级完成 |

---

## 🚀 快速开始

### 环境要求

- Python **3.10+**（已在 3.10 / 3.11 / 3.12 / 3.13 上验证）
- 无需注册任何账号，无需 API Key

### 三步跑起来

```bash
# 1. 获取代码
git clone https://github.com/wang-1168/fund-signal.git
cd fund-signal

# 2. 安装依赖
pip install -r requirements.txt
# 国内网络建议加镜像源，速度快很多：
# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 3. 启动交互界面
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`，在左侧输入基金代码（如 `000001`）点「🚀 开始分析」即可。

> **网络不通？** 把左侧「数据来源」切到 `local`，会用仓库自带的离线样例数据跑通全流程，不需要联网。

### 命令行用法

```bash
# 默认参数分析（沪深300 为基准，预测下一交易日方向）
python -m fund_signal --code 000001

# 预测未来 5 个交易日，换逻辑回归，调整阈值和成本
python -m fund_signal --code 110022 --horizon 5 --model logistic --threshold 0.55 --cost-bps 20

# 强制重新拉取最新数据（等同界面上的「刷新数据」）
python -m fund_signal --code 000001 --no-cache

# 离线运行，不需要网络
python -m fund_signal --code 000001 --data local

# 查看全部参数
python -m fund_signal --help
```

### 当库调用

```python
from fund_signal.config import Config
from fund_signal.pipeline import run_analysis

result = run_analysis(Config(fund_code="000001", horizon=1, n_splits=5))

print(result.summary_metrics())
print("最新上涨概率：", result.latest_prob)
print("持仓日胜率：", result.backtest.win_rate_active)
print("模型是否真有优势：", result.has_edge)
```

更多示例见 [`examples/quickstart.py`](examples/quickstart.py)。

---

## 🖥️ 界面预览

界面分 6 个标签页，全部图表可缩放、可悬停查看数值：

| 标签页 | 内容 |
|---|---|
| 📊 **概览** | 数据新鲜度 / 盘中估值 / 申赎状态 / 最新信号 / 净值走势 |
| 🎯 **预测详情** | 样本外概率序列、概率分布、**概率分档实际胜率**、信息系数 IC |
| 🧪 **模型评估** | 各折 AUC 柱状图、ROC 曲线、特征重要性、混淆矩阵 |
| 💰 **回测** | 净值曲线 + 回撤子图、**阈值/成本/延迟实时调节**、阈值敏感度、月度收益热力图 |
| 🔍 **数据探索** | 日收益率分布、滚动波动率、特征相关性热力图 |
| 📋 **原始数据** | 完整数据集与样本外预测结果，可一键导出 CSV |

配色遵循中国市场惯例：**红涨绿跌**。

---

## 🔬 方法论

### 为什么不用随机划分训练集？

金融时间序列有强自相关。随机划分会让「未来样本」进入训练集，模型相当于开卷考试，AUC 轻松刷到 0.7+，但实盘一文不值。本项目使用**滚动前向验证**：

```
折 1: [────────训练────────][测试]
折 2: [──────────训练──────────][测试]
折 3: [────────────训练────────────][测试]
                        ↑ 训练数据永远早于被预测样本
```

### 标签为什么以 T+1 日为起点？

公募基金的真实交易流程：

1. T 日收盘后（约 20:00）公布 T 日净值；
2. 投资者在 **T+1 日 15:00 前**提交申购，按 **T+1 日净值**成交。

所以 T 日晚上算出的信号，最早只能在 T+1 日成交。如果像很多项目那样用 `nav[T+1]/nav[T] - 1` 作为可交易收益，相当于假设「看完 T 日净值还能按 T 日净值买入」——**凭空多赚一天**。

本项目定义为：

```
buy_price  = nav[T + 1]              # T+1 日按净值成交
sell_price = nav[T + 1 + horizon]    # 持有 horizon 个交易日后卖出
tradeable_ret[T] = sell_price / buy_price - 1
```

### 特征清单（21 个）

| 类别 | 特征 |
|---|---|
| 动量 | `ret_1` `ret_5` `ret_10` `ret_20` |
| 波动 | `vol_5` `vol_20` |
| 均线偏离 | `ma_gap_5` `ma_gap_20` `ma_gap_60` |
| 技术指标 | `rsi_14` `macd_hist` `bias_20` |
| 风险 | `max_dd_20` `up_days_10` |
| 市场环境 | `idx_ret_1` `idx_ret_5` `idx_ret_20` `rel_strength_20` `corr_20` |
| 日历效应 | `dow` `month` |

所有特征均可在 T 日收盘后计算完成，不含任何未来信息。详见 [`docs/methodology.md`](docs/methodology.md)。

### 评估指标为什么这么设计

| 指标 | 作用 |
|---|---|
| **AUC** | 阈值无关的排序能力。0.5 = 无预测能力 |
| **多数类基线** | 「永远猜涨（或跌）」的准确率。**模型没超过它 = 什么都没学到** |
| **持仓日胜率** | 只统计有仓位的日子。全样本胜率会把空仓日（收益恰好为 0）算进分母，系统性低估 |
| **信息系数 IC** | 概率与未来收益的秩相关，不依赖阈值。\|IC\| < 0.03 基本是噪音 |
| **卡玛比率** | 年化收益 / 最大回撤，比夏普更贴近真实体感 |

---

## 📁 项目结构

```
fund-signal/
├── app.py                      # Streamlit 交互界面
├── fund_signal/
│   ├── config.py               # 参数中心
│   ├── data.py                 # 数据获取（含实时净值/盘中估值）+ 缓存
│   ├── features.py             # 特征工程（严格防未来函数）
│   ├── model.py                # 滚动前向验证 + LightGBM/逻辑回归
│   ├── backtest.py             # 回测（含执行延迟与交易成本）
│   ├── metrics.py              # 分类质量 + 策略绩效指标
│   ├── pipeline.py             # 端到端编排
│   └── __main__.py             # 命令行入口
├── examples/
│   ├── quickstart.py           # 当库调用的示例
│   └── sample_data/            # 离线样例数据（断网可用）
├── scripts/
│   └── build_sample_data.py    # 重新生成离线样例数据
├── tests/                      # 63 个单元测试（全部离线）
├── docs/methodology.md         # 方法论详解
├── requirements.txt
└── pyproject.toml
```

---

## ❓ 常见问题

<details>
<summary><b>为什么我的结果和「年化 20%」的基金预测项目差这么多？</b></summary>

因为那些项目多半犯了下面一个或多个错误：随机划分训练集、用了未来数据、忽略交易成本、只在单一样本上报告结果、或者干脆把「累计收益」和「年化收益」搞混。

本项目把这些问题都堵上了，所以结果看起来"不好看"——**但这个难看的结果才是真的**。
</details>

<details>
<summary><b>预测 A 股 ETF 行不行？</b></summary>

可以，但需要改造。ETF 有盘中连续交易和场内价格，`fund_open_fund_info_em` 拿不到。可以换用 `ak.fund_etf_hist_em`，并把执行时序假设从「T+1 按净值成交」改为「T 日收盘价成交」（ETF 日内可交易）。`backtest.py` 的 `execution_lag` 参数就是为这类调整预留的。
</details>

<details>
<summary><b>盘中估值接口为什么对某些基金没数据？</b></summary>

`fund_value_estimation_em` 只覆盖约 687 只基金（数据源按持仓可拟合度筛选），且仅在交易时段有值。取不到时返回 `None`，界面显示为 `—`。这是数据源覆盖范围决定的，不是 bug。
</details>

<details>
<summary><b>提示"样本量不足"怎么办？</b></summary>

时序验证需要足够长的历史。建议：换一只成立更早的基金、把 `horizon` 调小、或降低 `n_splits`。项目要求至少 150 行可用样本（推荐 350+）。
</details>

<details>
<summary><b>pip 安装很慢 / 卡住？</b></summary>

国内直连 PyPI 拉 akshare、streamlit 这种依赖多的包会非常慢。加上镜像源即可：
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```
另外，如果你的系统设置了 `HTTP_PROXY` 环境变量但代理已经失效，pip 会一直卡在超时上。可以临时清掉：
```bash
# Linux / macOS / Git Bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
# Windows PowerShell
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY -ErrorAction SilentlyContinue
```
</details>

<details>
<summary><b>可以用于实盘吗？</b></summary>

**不可以，也不建议。** 项目输出的是统计信号，默认结果已经显示它不具备可交易的优势。请把它当作学习与研究的工具。
</details>

---

## 🤝 参与贡献

欢迎提 Issue 和 PR！特别欢迎这几类贡献：

- 🐛 报告数据源接口变更（akshare 接口变动较频繁）
- 📊 补充新的特征或评估指标
- 🌍 适配 A 股 ETF / 港股基金 / 美股 ETF
- 📖 完善文档与示例

提交前请先跑通测试：

```bash
pytest tests/ -v
```

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## ⚖️ 免责声明

1. 本项目为**量化研究方法论演示与教学工具**，不构成任何投资建议。
2. 所有输出均为基于历史数据的统计结果，**历史表现不代表未来收益**。
3. 公募基金净值的短期走势接近随机，本项目已通过样本外验证展示这一点。
4. 数据来源于 akshare 聚合的公开接口，仅供学习研究使用，不保证准确性与及时性。
5. **投资有风险，决策需谨慎。因使用本项目产生的任何损失，作者不承担责任。**

---

## 📄 License

[MIT](LICENSE) © 2026 [wang-1168](https://github.com/wang-1168)

<div align="center">

**如果这个项目帮你搞懂了「为什么基金预测大多是骗人的」，欢迎点个 ⭐ Star**

</div>
