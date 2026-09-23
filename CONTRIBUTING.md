# 贡献指南

感谢你对 `fund-signal` 的关注。这个项目的**第一原则是诚实**：宁可报告一个「模型没有预测能力」的结论，也不要制造一个漂亮的假回测。任何贡献都必须服从这条原则。

---

## 目录

- [我能贡献什么](#我能贡献什么)
- [开发环境](#开发环境)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [最容易踩的坑：未来函数](#最容易踩的坑未来函数)
- [提交 PR 的流程](#提交-pr-的流程)
- [数据源相关](#数据源相关)
- [免责声明](#免责声明)

---

## 我能贡献什么

按价值排序，欢迎以下几类贡献：

| 类型 | 说明 | 难度 |
|---|---|---|
| **Bug 报告** | 尤其是「回测结果明显不合理」「数据接口报错」 | ⭐ |
| **新的因子/特征** | 必须在 `docs/methodology.md` 里说明经济学逻辑 | ⭐⭐ |
| **新的数据源适配** | 比如天天基金以外的净值源、可转债、ETF | ⭐⭐⭐ |
| **更好的评估方法** | 交叉验证方案、统计显著性检验（如 Diebold-Mariano） | ⭐⭐⭐ |
| **文档与文档纠错** | 中文错别字、说明不清、示例跑不通 | ⭐ |

**不欢迎**的贡献：

- 承诺收益率的营销文案；
- 用「调参到回测好看」来提升指标（过拟合）；
- 偷偷引入未来数据的特征工程；
- 把项目包装成投资建议工具。

---

## 开发环境

```bash
git clone https://github.com/wang-1168/fund-signal.git
cd fund-signal

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

国内网络建议加镜像：

```bash
pip install -e ".[dev]" -i https://mirrors.aliyun.com/pypi/simple/
```

跑一遍测试确认环境正常：

```bash
pytest -q
```

**测试必须离线可跑**：`tests/` 下所有用例只允许读取 `examples/sample_data/` 里的样本 CSV，不允许访问网络。CI 环境没有稳定的外网，联网测试会直接失败。

---

## 代码规范

- 使用 `ruff` 做检查与格式化（配置见 `pyproject.toml`）：

  ```bash
  ruff check .
  ruff format .
  ```

- 行宽 100 字符。
- 公有函数/类必须有 **中文 docstring**，说明参数、返回值，以及**任何与金融语义相关的假设**。
- 命名用英文，注释和文档用中文。
- 不要在库里 `print()`；统一用 `fund_signal.utils.get_logger()`。

---

## 测试要求

- 新增功能必须附带测试，放在 `tests/test_<模块名>.py`。
- 涉及时间序列的改动，**必须**补一个「无未来函数」的回归测试，模式参考 `tests/test_features.py::test_no_lookahead_bias`：

  ```python
  # 用截断后的数据重建数据集，最后一行特征必须和全量数据下的同一行完全一致
  ```

- 修改回测逻辑时，必须同时更新 `tests/test_backtest.py`，并在 PR 描述里给出**修改前 / 修改后**的关键指标对比。

提交前本地至少跑通：

```bash
pytest -q
ruff check .
```

---

## 最容易踩的坑：未来函数

这是本项目最在意的技术细节，请务必读完。

### 1. 标签的买入价必须是 T+1 净值

公募基金 T 日的净值在 T 日 **20:00 之后**才公布。也就是说，当你在 T 日收盘后拿到 `nav[T]` 并且产生买卖决策时，**你只能以 T+1 日的净值成交**。

因此前瞻收益的正确算法是：

```python
buy_nav = nav.shift(-1)  # T+1 成交价
sell_nav = nav.shift(-(1 + horizon))  # T+1+horizon 卖出价
fwd_ret = sell_nav / buy_nav - 1.0
```

早期版本错误地写成 `nav.shift(-horizon) / nav - 1`，等于用 T 日净值买入——这是**白送一天收益**，会让回测虚高。实测影响：某标的策略收益从 −66% 变成 −34%，持仓胜率从 51%+ 掉到 49.84%。这类错误会让整个结论反向，所以 `test_features.py` 里锁死了这条规则。

### 2. 特征只能用到 T 日及更早

所有滚动窗口都要 `shift(1)` 或用 `rolling(...).apply` 作用于已实现数据。**严禁**出现 `.shift(-n)`（n>0）出现在特征里。

### 3. 滚动前推验证时训练集必须严格早于预测集

`model.walk_forward_validate` 已经强制了这一点，回测也只使用 `wf.predictions`（样本外预测）。不要在回测里混入样本内预测。

### 4. 交易成本必须按换手收取

费率只在**仓位发生变化**时计提，不能按天固定扣。默认 15 bps（0.15%），是场外基金申赎费的一个粗略近似。

---

## 提交 PR 的流程

1. Fork 本仓库，从 `main` 切出分支：`git checkout -b feat/your-feature`。
2. 提交信息用约定式前缀：`feat:` / `fix:` / `docs:` / `test:` / `refactor:` / `chore:`。
3. 确保 `pytest -q` 与 `ruff check .` 全绿。
4. 发起 PR，并在描述里填清楚模板要求的内容——尤其是**方法学影响**。
5. 如果改动会影响 README 里的示例输出（比如胜率、AUC 数字），请一并更新 README，保持文档与实际一致。

---

## 数据源相关

- 默认数据源是 [akshare](https://github.com/akfamily/akshare)，免费、无需 token。
- akshare 的接口会变动。遇到 `KeyError` / `AttributeError` / 空 DataFrame，先试：

  ```bash
  pip install -U akshare
  ```

- 如果确实是接口变更，请更新 `fund_signal/data.py` 里的对应函数，并把新的列名写进规范化逻辑，同时补一个测试。
- **不要**在代码里硬编码任何私有 token 或付费接口。

---

## 免责声明

本项目是**量化方法论的演示与教学工具**，不构成任何投资建议。

公募基金的短期净值方向在学术上接近弱式有效市场，**不要指望它稳定赚钱**。项目里刻意保留并展示「模型没有优势」的诚实结论（见 README 的实测数据），这是设计的一部分。

贡献者在提交代码时，即表示认同：**不夸大预测能力，不隐瞒失败结果**。
