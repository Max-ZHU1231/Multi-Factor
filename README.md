# Multi-Factor 多因子选股研究框架

A end-to-end quantitative research framework for Chinese A-share markets, covering data downloading, cleaning, quality checks, factor construction, IC analysis, and layer backtesting.

---

## 目录

- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [数据层](#数据层)
  - [数据下载](#1-数据下载--download_datapy)
  - [数据清洗](#2-数据清洗--data_cleanerpy)
  - [数据质检](#3-数据质检--data_qualitypy)
- [因子框架](#因子框架--factor_framework)
  - [算子库](#31-算子库--operatorspy)
  - [因子引擎](#32-因子引擎--factor_enginepy)
  - [因子库](#33-内置因子库--factor_zoopy)
  - [中性化](#34-因子中性化--neutralizepy)
  - [IC 分析](#35-ic-分析--ic_analysispy)
  - [分层回测](#36-分层回测--backtestpy)
  - [端到端 Pipeline](#37-端到端-pipeline--pipelinepy)
- [快速上手](#快速上手)
- [测试](#测试)
- [文件索引](#文件索引)

---

## 项目结构

```
Multi-Factor/
│
├── 数据文件
│   ├── 股票列表-stock_basic.csv        # 5490 只 A 股基本信息（ts_code, industry 等）
│   ├── 交易日历-trade_cal.csv          # 交易日历
│   ├── 上市公司基本信息-stock_company.csv
│   └── Stocks/                         # 单股日频 CSV，文件名格式：000001.SZ.csv
│
├── 数据层
│   ├── download_data.py                # AKShare 数据下载脚本
│   ├── data_cleaner.py                 # MAD Winsorize + 缺失值处理
│   └── data_quality.py                 # 价格连续性/成交量异常/财务一致性/时区对齐
│
├── 因子框架
│   └── factor_framework/
│       ├── __init__.py                 # 包入口
│       ├── operators.py                # 算子库（时间序列/横截面/数学/跨资产）
│       ├── factor_engine.py            # FactorEngine：注册、计算、面板构建
│       ├── factor_zoo.py               # 20 个内置预定义因子
│       ├── neutralize.py               # 因子中性化（回归法/行业 Z-Score/正交化）
│       ├── ic_analysis.py              # IC 分析（Rank IC/ICIR/衰减/Newey-West）
│       ├── backtest.py                 # 分层回测（夏普/最大回撤/Calmar/换手率）
│       └── pipeline.py                 # FactorPipeline 端到端流水线
│
└── 测试
    ├── test_data_cleaner.py            # 262 个测试
    ├── test_data_quality.py            # 183 个测试
    └── test_factor_framework.py        # 94 个测试（全部通过，共 539 个）
```

---

## 环境准备

```powershell
# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 安装依赖
pip install pandas numpy scipy tqdm akshare pytest
```

**Python 版本**：3.10+（已在 3.13 上验证）

---

## 数据层

### 1. 数据下载 — `download_data.py`

使用 [AKShare](https://akshare.akfan.cn/) 免费接口下载 A 股数据，无需 Token。

```powershell
python download_data.py
```

下载内容：
| 数据类型 | 接口 | 保存位置 |
|---|---|---|
| 日频量价（复权后） | `ak.stock_zh_a_hist` | `Stocks/{code}.{exchange}.csv` |
| 估值指标（PE/PB/市值） | 百度股票接口 | 并入 `Stocks/` CSV |
| 综合财务指标 | `ak.stock_financial_analysis_indicator` | `fundamentals/` |

**每只股票 CSV 的列结构：**

| 列名 | 说明 |
|---|---|
| `交易日` | YYYYMMDD 字符串 |
| `股票代码` | ts_code 格式（如 `000001.SZ`） |
| `收盘价` | 后复权收盘价 |
| `开盘价 / 最高价 / 最低价` | 后复权 OHLC |
| `成交量（手）` / `成交额（千元）` | 成交量 / 额 |
| `换手率（%）` | 当日换手率 |
| `总市值（万元）` / `流通市值（万元）` | 市值 |
| `市净率` / `市盈率（TTM，亏损为空）` / `市销率（TTM）` | 估值指标 |
| `复权因子` | 后复权因子 |

---

### 2. 数据清洗 — `data_cleaner.py`

对单股 DataFrame 执行**五步清洗流程**：

```
原始 CSV
  ↓  [1] MAD Winsorize（每列独立，阈值 ±3 MAD）
  ↓  [2] 价格/量能列：停牌 ffill（最多 5 个交易日）
  ↓  [3] 估值类列：ffill 不限长度（PIT 原则，严禁前向填充未来数据）
  ↓  [4] 随机缺失（< 5%）：横截面均值填充
  ↓  [5] 大量缺失（> 30%）：整列标记无效 / 新股（< 60 行）：整只剔除
已清洗 DataFrame
```

**API：**

```python
from data_cleaner import load_and_clean, mad_winsorize

# 加载并清洗单股数据
df = load_and_clean("Stocks/000001.SZ.csv")
# 返回 None 表示新股数据不足，返回 DataFrame 则已完成清洗

# 单独调用 MAD Winsorize
from data_cleaner import mad_winsorize
clean_series = mad_winsorize(series, k=3.0)
```

---

### 3. 数据质检 — `data_quality.py`

四类额外质量检查：

| 检查项 | 函数 | 说明 |
|---|---|---|
| 价格连续性 | `check_price_continuity(df)` | 复权因子不变时前收盘跳跃 > 5% → 异常 |
| 成交量异常 | `check_volume_anomaly(df)` | 成交量 = 0 的交易日标记为疑似停牌 |
| 财务一致性 | `check_financial_balance(df, fin_df)` | \|资产 - 负债 - 权益\| / 资产 > 1% → 异常 |
| 月频对齐 | `align_monthly_to_daily(macro_df, daily_df)` | 月频宏观/财务数据对齐到日频 |

```python
from data_quality import run_all_checks, align_monthly_to_daily

result = run_all_checks(df)
# result = {
#   "price_continuity": {"anomaly_count": 0, ...},
#   "volume_anomaly":   {"suspension_days": 12, ...},
#   "financial_balance": None,   # 若 fin_df 未传入则跳过
# }

# 月频财务数据对齐
daily_fin = align_monthly_to_daily(fin_df, price_df)
```

---

## 因子框架 — `factor_framework/`

完整的**定义因子 → 构建面板 → 中性化 → IC 检验 → 分层回测 → 输出报告**流水线。

```
自定义因子函数
    ↓  FactorEngine.register()
    ↓  FactorEngine.build_panel()      →  (日期 × 股票) 因子面板
    ↓  neutralize_regression()          →  市值 + 行业中性化
    ↓  FactorEngine.apply_cross_section(cs_rank)  →  截面标准化
    ↓  compute_ic() + ic_stats()        →  IC / ICIR / t 检验
    ↓  layer_backtest() + long_short_stats()  →  分层回测 + 多空收益
    ↓  FactorReport.print_summary()     →  终端报告
    ↓  FactorReport.save()              →  CSV 输出
```

---

### 3.1 算子库 — `operators.py`

所有函数均作用于 `pd.Series`，可自由组合搭建自定义因子。

#### 时间序列算子（输入：单股历史序列）

| 函数 | 说明 |
|---|---|
| `ts_sum(x, d)` | 滚动 d 天累加和 |
| `ts_mean(x, d)` | 滚动 d 天均值 |
| `ts_stddev(x, d)` | 滚动 d 天标准差 |
| `ts_corr(x, y, d)` | 滚动 d 天皮尔逊相关 |
| `delay(x, d)` | 取 d 天前的值 |
| `ts_max(x, d)` / `ts_min(x, d)` | 滚动最大/最小值 |
| `ts_rank(x, d)` | 过去 d 天内当日值的百分比排名 |
| `ts_delta(x, d)` | 当日值 - d 天前的值 |
| `ts_wma(x, d)` | 线性加权移动均值 |
| `ts_zscore(x, d)` | 滚动 Z-Score |
| `ts_skew(x, d)` | 滚动偏度 |
| `ts_autocorr(x, d, lag)` | 滚动自相关系数 |

#### 横截面算子（输入：某截面日所有股票的因子值）

| 函数 | 说明 |
|---|---|
| `cs_rank(x)` | 百分比排名（0~1） |
| `cs_zscore(x)` | 截面标准化 |
| `cs_demean(x)` | 截面去均值 |
| `cs_scale(x, a)` | 线性映射到 [0, a] |
| `cs_industry_neutral(x, group)` | 行业内去均值 |
| `cs_industry_zscore(x, group)` | 行业内 Z-Score |
| `cs_winsorize(x, n_std)` | 截面 MAD Winsorize |

#### 数学/逻辑算子

`log`, `sqrt`, `absx`, `sign`, `if_else`, `clip`, `power`

---

### 3.2 因子引擎 — `factor_engine.py`

`FactorEngine` 负责遍历 `Stocks/` 目录、批量计算因子、拼接面板。

**因子函数签名：**
```python
def my_factor(df: pd.DataFrame) -> pd.Series:
    # df：单只股票的日频 DataFrame（已清洗，按交易日升序）
    # 返回：与 df 等长的因子值 Series
    ...
```

**API：**
```python
from factor_framework.factor_engine import FactorEngine
from factor_framework.operators import ts_mean, log

engine = FactorEngine(
    stocks_dir  = "Stocks/",
    stock_basic = "股票列表-stock_basic.csv",
    min_rows    = 60,      # 新股最小行数
    verbose     = True,    # 显示进度条
)

# 注册因子
engine.register("log_mktcap", lambda df: log(df["总市值（万元）"]))
engine.register("mom_20",     lambda df: df["收盘价"].pct_change(20))

# 构建面板（日期 × 股票）
panel = engine.build_panel("mom_20", start="20200101", end="20231231")
# → pd.DataFrame，index=交易日，columns=ts_code

# 构建未来收益率面板（用于 IC / 回测）
ret_panel = engine.build_return_panel(forward=21, start="20200101", end="20231231")

# 横截面标准化
from factor_framework.operators import cs_rank
ranked = engine.apply_cross_section(panel, cs_rank)
```

---

### 3.3 内置因子库 — `factor_zoo.py`

20 个开箱即用的预定义因子，涵盖五大类：

| 类别 | 因子名 | 说明 |
|---|---|---|
| **动量** | `momentum_12_1` | 12-1 月动量（跳过最近 1 月） |
| | `momentum_6_1` | 6-1 月中期动量 |
| | `momentum_1m` | 1 月短期动量 |
| | `momentum_52w_high` | 价格 / 52 周高点 |
| **反转** | `reversal_1w` | 1 周短期反转 |
| | `reversal_1m` | 1 月反转 |
| **波动率** | `vol_20d` | 20 日历史波动率 |
| | `vol_60d` | 60 日历史波动率 |
| | `vol_skew` | 60 日收益率偏度 |
| | `downside_vol` | 下行波动率 |
| **估值** | `value_pb` | 1 / 市净率 |
| | `value_pe_ttm` | 1 / 市盈率(TTM) |
| | `value_ps_ttm` | 1 / 市销率(TTM) |
| **规模** | `size_log_mktcap` | −ln(总市值)（小市值因子） |
| | `size_log_free_cap` | −ln(流通市值) |
| **量价** | `amihud_illiquidity` | Amihud 非流动性 |
| | `turnover_rate` | 20 日平均换手率 |
| | `vol_price_corr` | 量价相关性 |
| | `vwap_deviation` | 价格偏离 VWAP |
| | `price_strength` | 收盘价强度（C / (H+L)/2） |

```python
from factor_framework.factor_zoo import register_all

# 一键注册全部内置因子
register_all(engine)

# 或通过 pipeline 选择注册
pipe.register_builtins(["momentum_12_1", "vol_20d", "value_pb"])
```

---

### 3.4 因子中性化 — `neutralize.py`

三种方法消除因子中的风险暴露：

#### 方法一：回归法（推荐）
$$f_i = \alpha + \beta \cdot \ln(\text{MktCap}_i) + \sum_k \gamma_k D_{ik} + \varepsilon_i$$
取残差 $\varepsilon_i$ 作为中性化后因子。

```python
from factor_framework.neutralize import neutralize_regression

neutral_panel = neutralize_regression(
    factor_panel,
    mktcap_panel,
    industry_map = engine.industry_map,   # ts_code → industry
)
```

#### 方法二：行业内 Z-Score
```python
from factor_framework.neutralize import neutralize_industry_zscore

neutral_panel = neutralize_industry_zscore(factor_panel, engine.industry_map)
```

#### 方法三：正交化（消除与已有因子的相关性）
```python
from factor_framework.neutralize import orthogonalize

orth_panel = orthogonalize(new_factor_panel, [existing_factor1, existing_factor2])
```

---

### 3.5 IC 分析 — `ic_analysis.py`

```python
from factor_framework.ic_analysis import compute_ic, ic_stats, ic_significance, ic_decay

# 逐期计算 Rank IC（推荐）或 Normal IC
ic = compute_ic(factor_panel, return_panel, method="rank")

# 核心统计指标
stats = ic_stats(ic, annualize_periods=252)
# 返回：mean_ic, std_ic, icir, win_rate, t_stat, p_value,
#        ic_positive, ic_negative, total_periods, annualized_icir

# Newey-West 修正 t 检验（处理 IC 自相关）
nw = ic_significance(ic, lags=4)
# 返回：nw_t_stat, nw_p_value

# IC 衰减分析（不同预测期的预测力）
decay_df = ic_decay(factor_panel, close_panel, forward_periods=[1, 5, 10, 21, 60])
# 返回 DataFrame，index=预测期，columns=[mean_ic, icir, win_rate, ...]

# 多因子截面相关性矩阵
from factor_framework.ic_analysis import cross_factor_correlation
corr_matrix = cross_factor_correlation({"mom": mom_panel, "vol": vol_panel})
```

**因子有效性参考标准：**

| 指标 | 合格线 | 优秀线 |
|---|---|---|
| \|Mean IC\| | > 0.02 | > 0.05 |
| ICIR | > 0.5 | > 1.0 |
| IC 胜率 | > 55% | > 60% |
| \|t 统计量\| | > 2.0 | > 3.0 |

---

### 3.6 分层回测 — `backtest.py`

```python
from factor_framework.backtest import layer_backtest, long_short_stats, turnover_analysis

# 将股票按因子值分为 5 层，等权持有，计算各层收益
layer_ret = layer_backtest(factor_panel, return_panel, n_groups=5, direction=1)
# 返回 DataFrame，columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'LS']
# LS = Q5 - Q1（多空收益）

# 多空组合绩效
stats = long_short_stats(layer_ret, periods_per_year=252, rf=0.0)
# 返回：
#   ls_annual_return, ls_sharpe, ls_max_drawdown, ls_calmar, ls_win_rate
#   monotone_score（Spearman 单调性，越接近 1 越好）
#   layer_annual_return（各层年化收益字典）
#   nav（净值 DataFrame）

# 换手率与交易成本
to = turnover_analysis(factor_panel, n_groups=5, cost_per_side=0.002)
# 返回：avg_turnover, avg_cost
```

---

### 3.7 端到端 Pipeline — `pipeline.py`

`FactorPipeline` 将以上所有步骤串联，一行代码完成完整因子检验。

```python
from factor_framework.pipeline import FactorPipeline

# ① 初始化
pipe = FactorPipeline(
    stocks_dir  = "Stocks/",
    stock_basic = "股票列表-stock_basic.csv",
    verbose     = True,
)

# ② 注册因子（选一种）
pipe.register_builtins(["momentum_12_1", "vol_20d"])          # 使用内置因子
pipe.register_factor("my_factor", lambda df: df["收盘价"].pct_change(20))  # 自定义因子

# ③ 运行单因子完整检验
report = pipe.run(
    factor_name      = "momentum_12_1",
    start            = "20200101",
    end              = "20231231",
    forward          = 21,           # 预测期：21 个交易日（约 1 月）
    n_groups         = 5,            # 分层数
    direction        = 1,            # +1：因子越大越好；-1：反转
    standardize      = "rank",       # 截面标准化方式：'rank' / 'zscore' / None
    neutralize       = False,        # 是否做市值+行业中性化
    winsorize        = True,         # 截面 MAD Winsorize
    ic_method        = "rank",       # IC 计算方式
    ic_forward_list  = [1, 5, 10, 21, 60],  # IC 衰减分析的预测期
    periods_per_year = 252,
    cost_per_side    = 0.002,        # 单边交易成本 0.2%
)

# ④ 查看结果
report.print_summary()          # 终端打印完整报告
report.save("output/")          # 保存所有 CSV 到 output/momentum_12_1/

# ⑤ 批量比较多个因子
comparison = pipe.run_batch(
    factor_names = ["momentum_12_1", "vol_20d", "value_pb"],
    start        = "20200101",
    end          = "20231231",
    forward      = 21,
    n_groups     = 5,
    standardize  = "rank",
    neutralize   = False,
    ic_forward_list = [1, 5, 10],
)
print(comparison[["mean_ic", "icir", "ls_sharpe", "ls_annual_return"]])
```

**`report.save()` 输出文件：**

```
output/momentum_12_1/
├── summary.csv          # 所有指标汇总（一行）
├── ic_series.csv        # 逐期 IC 时间序列
├── ic_decay.csv         # IC 衰减分析表
├── layer_returns.csv    # 各层逐期收益
├── nav.csv              # 净值曲线（Q1~Q5 + LS）
└── factor_panel.csv     # 因子面板
```

---

## 快速上手

### 场景一：检验单个内置因子

```python
from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline("Stocks/", "股票列表-stock_basic.csv")
pipe.register_builtins(["momentum_12_1"])

report = pipe.run(
    "momentum_12_1",
    start="20200101", end="20231231",
    forward=21, n_groups=5, standardize="rank",
)
report.print_summary()
report.save("output/")
```

### 场景二：用算子自定义因子

```python
from factor_framework.pipeline import FactorPipeline
from factor_framework.operators import ts_mean, ts_stddev, log

pipe = FactorPipeline("Stocks/", "股票列表-stock_basic.csv")

# 定义：量价背离因子（成交量 Z-Score 与收益率的差）
def vol_price_divergence(df):
    vol_z = (df["成交量（手）"] - ts_mean(df["成交量（手）"], 20)) / ts_stddev(df["成交量（手）"], 20)
    ret   = df["收盘价"].pct_change()
    return vol_z - ts_mean(ret, 20) / (ts_stddev(ret, 20) + 1e-9)

pipe.register_factor("vol_price_div", vol_price_divergence)
report = pipe.run("vol_price_div", start="20200101", end="20231231", forward=5)
report.print_summary()
```

### 场景三：中性化 + 多因子对比

```python
pipe = FactorPipeline("Stocks/", "股票列表-stock_basic.csv")
pipe.register_builtins(["momentum_12_1", "vol_20d", "value_pb", "amihud_illiquidity"])

# 中性化检验（消除市值 + 行业暴露）
report = pipe.run(
    "momentum_12_1",
    start="20180101", end="20231231",
    forward=21, neutralize=True, standardize="rank",
)

# 多因子对比
df = pipe.run_batch(
    ["momentum_12_1", "vol_20d", "value_pb", "amihud_illiquidity"],
    start="20180101", end="20231231", forward=21, neutralize=False,
)
print(df.sort_values("icir", ascending=False))
```

---

## 测试

```powershell
# 激活虚拟环境
.venv\Scripts\Activate.ps1

# 运行全部测试（539 个）
python -m pytest -v

# 只运行因子框架测试
python -m pytest test_factor_framework.py -v

# 只运行数据清洗测试
python -m pytest test_data_cleaner.py test_data_quality.py -v
```

**当前测试状态：**

| 测试文件 | 测试数 | 状态 |
|---|---|---|
| `test_data_cleaner.py` | 262 | ✅ 全部通过 |
| `test_data_quality.py` | 183 | ✅ 全部通过 |
| `test_factor_framework.py` | 94 | ✅ 全部通过 |
| **合计** | **539** | ✅ |

---

## 文件索引

| 文件 | 职责 |
|---|---|
| `download_data.py` | AKShare 数据下载（量价 + 估值 + 财务） |
| `data_cleaner.py` | MAD Winsorize + 5 规则缺失值处理 |
| `data_quality.py` | 价格连续性/停牌/财务等式/时区对齐检查 |
| `factor_framework/operators.py` | ~30 个时序/横截面/数学算子 |
| `factor_framework/factor_engine.py` | 因子注册 + 批量面板构建 |
| `factor_framework/factor_zoo.py` | 20 个内置预定义因子 |
| `factor_framework/neutralize.py` | 回归法/行业 Z-Score/正交化 |
| `factor_framework/ic_analysis.py` | IC / ICIR / Newey-West / IC 衰减 |
| `factor_framework/backtest.py` | 分层回测 + 夏普/回撤/Calmar + 换手率 |
| `factor_framework/pipeline.py` | `FactorPipeline` 端到端流水线 + `FactorReport` |
| `股票列表-stock_basic.csv` | 5490 只 A 股元数据（含行业字段） |
| `Stocks/` | 单股日频 CSV（格式：`000001.SZ.csv`） |
