# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

_暂无_

---

## [0.2.0] - 2026-09-23

「能力边界」版本。v0.1.0 之后，本项目正面回答了那个一定会被问到的问题——**准确率能不能做到 90%**——并把它做成了可复现的实验、可交互的页面。

### 新增 Added

**能力边界审计（新模块 `fund_signal.audit`）**

- `auc_to_accuracy` / `accuracy_to_auc` / `auc_accuracy_curve`：AUC 与最优判定准确率的理论换算。推导给出 **准确率 90% 需要 AUC ≈ 0.965** 这条硬约束。
- `complexity_path`：5 档复杂度 × 滚动前向验证，同时记录训练准确率与样本外准确率，画出**过拟合/欠拟合的完整地形图**（实测训练 56%→100%，样本外始终 50%~52%）。
- `leakage_audit`：三组对照（正常 / 答案入特征 / 未来函数标签）证明「高准确率无法证明模型有效」——答案入特征时**严格滚动前向也会给出 100% 准确率**，而未来函数标签几乎不改变准确率。
- `confidence_subset_table` / `selective_accuracy_table`：等频置信度分层与选择性预测（用覆盖率换准确率），并把「层内多数类基线」并排摆出来，防止把行情好误认成模型强。
- `calibration_diagnostics` / `capability_report`：ECE / MCE / Brier 与能力边界汇总。

**建模层**

- 新增 `model_type="ensemble"`：LightGBM + 逻辑回归 + ExtraTrees + GBDT 四模型软投票，`feature_importance` 改为逐成员归一化后取平均。
- `walk_forward_validate` 新增 **`embargo` 净化间隔**（丢弃紧邻测试段的 `horizon` 个训练样本，切断标签窗口重叠泄漏）与 **`calibrate` 样本外概率校准**（每折内部用训练集尾部 20% 拟合 isotonic 校准器）。
- `WalkForwardResult` 新增 `raw_predictions` / `calibrate_method`，校准前后概率可对比。
- 新增 `fit_calibrator` / `apply_calibrator`，样本不足或单一类别时优雅回退。

**界面（6 → 8 页签）**

- 新增 **🧭 风控与仓位**：校准概率 → 分段线性仓位映射（附纯 Kelly 与 1/4 Kelly 参照）、置信度分层、选择性预测、校准前后概率分布、同波动率档位条件收益分布、压力测试。
- 新增 **🛡️ 精度审计**：AUC→准确率理论曲线（标注实测点与 90% 目标线）、复杂度地形、泄漏审计、多数类基线对照、五种「刷准确率」手段的代价表、选择性预测曲线。
- **canvas 粒子背景** + 玻璃拟态 KPI 卡片 + 渐变区块标题 + 入场动效，可在侧边栏一键关闭；深浅色自适应。
- 侧边栏新增模型集成 / 概率校准 / 净化间隔 / 粒子背景开关，并新增「稳健（四模型集成 6 折）」预设。

**其他**

- 依赖新增 `scipy`（能力边界审计的正态分位换算）。
- 测试从 63 项扩充到 **81 项**（新增 `tests/test_audit.py` 与模型层校准/净化/集成测试）。

**文档与展示**

- 新增 `docs/screenshots/` 界面画廊：8 个标签页各一张实拍截图（WebP，合计约 0.67 MB），
  并在 README 页首加入总览页主视觉图。此前「界面预览」章节只有文字表格与描述，
  读者无法直观判断界面形态——这是本项目第一次把「界面长什么样」摆到台面上。
- README「界面预览」章节改为图片画廊 + 可折叠的逐页签详细说明。
- 新增 [`scripts/verify_claims.py`](scripts/verify_claims.py)：把 README 与 `docs/`
  中引用的**全部实测数字**集中成一份可执行的声明清单（54 项），每次运行都重新实算并逐项比对，
  存在漂移时以非零退出码报错。CI 新增 `verify documented numbers` 任务，在每次 push / PR 时运行。

### 变更 Changed

- 界面 `st.components.v1.html` 迁移到 `st.iframe`（Streamlit 1.63+，旧版自动回退）。
- 百分比列统一改用 `NumberColumn(format="percent")`，修复覆盖率/仓位显示成 1% 而非 100% 的错误。
- 版本号由 0.1.0 提升至 0.2.0。

### 修复 Fixed

- **修正文档中的失效实测数字（重要）**。`calibrate`（样本外概率校准）在本版本由可选变为默认之后，
  README 与 `docs/methodology.md` 里引用的 AUC / 准确率 / 累计收益 / 持仓日胜率**全部失效**——
  它们来自未开启校准的旧默认值，实算与文档不符。具体修正：

  | 指标（000001，离线样例） | 文档原值（已失效） | 修正后 |
  |---|---|---|
  | AUC | 0.5286 | **0.4994** |
  | 准确率 | 0.5149 | **0.4952** |
  | 超额 acc_edge | +0.0130 | **−0.0067** |
  | 策略累计收益 | −33.79% | **−49.70%** |
  | 持仓日胜率 | 49.84% | **47.94%** |
  | 可用样本 | 5,281 | **5,981** |

  连带修正「3 基金 × 4 周期 × 2 模型」24 组区间（AUC 均值 0.516 → **0.511**）、
  复杂度地形、校准前后 ECE（0.09→0.04 改为 **0.071→0.042**）、`docs/methodology.md` 的
  三基金实测表（补入第三只 161725 与多数类基线列）。
- **修正一处过度乐观的结论**。文档原称「选择性预测：只对最自信的前 5% 下注，准确率能到 60%~70%」。
  按当前默认参数实测**并不成立**：前 5% 准确率仅 0.520、超额 **−0.013**（为负）；
  最优档位是覆盖率 20% 时准确率 0.533、超额 +0.022。README 与 `docs/methodology.md`
  已改为如实呈现，并补上「置信度更高 ≠ 更准」的解释。
- 补充说明「联网结果与离线样例为何有差异」：联网取数时沪深300 指数自 2005-01-04 起，
  与基金净值内连接后样本比离线样例少约 700 行（000001 为例：5,281 vs 5,981），
  因此两侧数字不会完全一致。此前文档未说明这一点，容易被误读为结果不可复现。

---

## [0.1.0] - 2026-09-23

首个公开版本。目标不是「预测准」，而是**把量化基金信号这件事做对、说清楚**。

### 新增 Added

**核心框架**

- `fund_signal` 包，拆分为 `config` / `data` / `features` / `model` / `metrics` / `backtest` / `pipeline` 七个模块。
- `Config` 数据类统一承载所有超参数，支持 `--code --horizon --model --benchmark --threshold --splits --cost-bps --seed` 等 CLI 参数覆盖。
- CLI 入口：`python -m fund_signal` 与安装后的 `fund-signal` 命令。

**数据层**

- 基于 akshare 的数据接入，全部接口均带重试与本地 CSV 缓存：
  - `fund_open_fund_info_em` —— 历史净值；
  - `fund_open_fund_daily_em` —— 开放式基金实时快照（约 24000 只）；
  - `fund_value_estimation_em` —— 盘中净值估算（约 687 只，覆盖范围有限）；
  - `index_zh_a_hist` —— 基准指数行情。
- 离线样本数据：`examples/sample_data/` 内置 3 只基金净值与沪深300指数，**无网络也能完整跑通全流程**。
- `--data local` 强制离线模式，测试与 CI 依赖它。

**特征工程**

- 价格类：多周期收益、波动率、均线偏离、RSI(14 归一化)、MACD 柱、量价结构等。
- 市场类：基准指数同频收益、相对强弱。
- 日历类：星期效应哑变量。
- 严格约束：所有特征仅使用 T 日及更早信息。

**建模**

- LightGBM（保守参数：`n_estimators=300, lr=0.03, num_leaves=15, max_depth=4`，刻意压低过拟合空间）。
- LogisticRegression + StandardScaler 作为线性基线，用于对照。
- 滚动前推验证（walk-forward），训练集严格早于预测集。
- 特征重要性归一化输出。

**评估（诚实优先）**

- AUC、准确率、精确率/召回率。
- **多数类基线准确率**（majority baseline）与 **acc_edge**：模型相对「无脑猜涨」的增量。
- **持仓日胜率**：只在策略实际持仓的交易日统计，区别于「全样本准确率」的粉饰口径。
- **信息系数 IC**：预测概率与未来收益的秩相关。
- `has_edge` 判定：综合 AUC 与 acc_edge，明确回答「这个模型到底有没有优势」。

**回测**

- 含成本的阈值策略回测（默认 15 bps，仅在仓位变化时计提）。
- 阈值扫描 `sweep_thresholds`，输出阈值-收益曲线。
- **执行时滞建模**：`execution_lag` 默认 0，配合标签里的 T+1 成交价，避免「白送一天」。
- 净值曲线对齐基准（买入持有）对照。

**Streamlit 交互界面**（`app.py`）

- 6 个页签：概览 / 预测详情 / 模型评估 / 回测 / 数据探索 / 原始数据。
- 14 张 Plotly 交互图表，遵循 A 股「红涨绿跌」配色。
- 实时数据新鲜度面板：数据最新日期、距今交易日数、盘中估算净值。
- 阈值 / 费率 / 执行时滞实时滑块，改动即时重算。
- `@st.cache_data(ttl=900)` 缓存 + 手动「🔄 刷新数据」按钮。

**工程化**

- `tests/` 下 63 个单元测试，离线运行，1.6 秒跑完。
- 含 `test_no_lookahead_bias` —— 用数据截断法锁死未来函数。
- GitHub Actions CI：Ubuntu + Windows × Python 3.10/3.11/3.12/3.13，外加 ruff lint 任务。
- `pyproject.toml`（setuptools + ruff + pytest 配置）、`requirements.txt`（锁定实测版本）、`.gitignore`、MIT `LICENSE`。

### 修复 Fixed

- **修复未来函数（重大）**：标签原先用 `nav.shift(-horizon) / nav - 1`，等于以 T 日净值买入，与基金实际 T+1 成交规则不符，导致回测虚高。改为以 `nav.shift(-1)` 为买入价、`nav.shift(-(1+horizon))` 为卖出价，回测执行时滞同步由 1 调为 0。实测影响：样本基金策略收益由 −66% 修正为 −33.79%（同期买入持有 +26.14%），持仓日胜率 49.84%。
- 修复 `fetch_latest_nav` 在当日净值尚未公布（列为 NaN）时返回 `None` 的问题，改为按日期列倒序查找最后一个非空值。
- 修复 `equity_curve` 文档字符串错误（首值应为 `start * (1 + r[0])`，而非 1.0）。
- 修复测试用例中误用不存在的 `Series.has_duplicates`，改用 `is_unique`。
- 批量替换已废弃的 Streamlit 参数 `use_container_width=True` → `width="stretch"`（官方将于 2025-12-31 后移除）。

### 已知限制 Known Limitations

- **净值方向的可预测性本身很弱**。样本标的实测 AUC 约 0.50–0.53，持仓日胜率约 49.8%——接近抛硬币。这是**结论**，不是 bug。
- 盘中估值接口 `fund_value_estimation_em` 仅覆盖约 687 只基金（占全市场不足 3%），未被覆盖的基金会显示「无盘中估算数据」。
- 历史净值不区分 A/C 份额的费用差异，回测用统一 bps 近似。
- 未考虑大额申赎限制、限购、QDII 的 T+2 到账等真实约束。

### 说明 Notes

- 本项目**不构成投资建议**。它演示的是「如何严谨地评估一个信号」，而不是「如何赚钱」。
- 数据来自 akshare 聚合的公开接口，仅供研究学习，商业使用请自行确认数据来源的授权条款。

---

[Unreleased]: https://github.com/wang-1168/fund-signal/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/wang-1168/fund-signal/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/wang-1168/fund-signal/releases/tag/v0.1.0
