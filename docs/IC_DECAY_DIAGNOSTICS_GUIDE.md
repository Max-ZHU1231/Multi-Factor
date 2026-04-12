# IC 衰减异常诊断使用指南

> **版本**: v1.0 · **更新**: 2026-04-12  
> **适用场景**: 当观察到 forward 越长、IC 越高（而非正常衰减）时，使用本框架定位根因。

---

## 目录

1. [背景与问题定义](#1-背景与问题定义)
2. [六模块框架概览](#2-六模块框架概览)
3. [快速上手](#3-快速上手)
   - 3.1 CLI 一键运行
   - 3.2 Python API 调用
   - 3.3 从已有 FactorReport 调用
4. [模块详解](#4-模块详解)
   - M1 时间对齐与前瞻偏差检查
   - M2 收益定义与累计窗口拆解
   - M3 市场/风格暴露剥离
   - M4 样本偏差检查
   - M5 因子属性验证（时效性）
   - M6 稳健性复核
5. [最终判定口径](#5-最终判定口径)
6. [输出文件规范](#6-输出文件规范)
7. [判断阈值参数表](#7-判断阈值参数表)
8. [常见根因与修复建议](#8-常见根因与修复建议)
9. [附录：value_pb 诊断示例](#9-附录value_pb-诊断示例)

---

## 1. 背景与问题定义

**正常的因子 IC 衰减形态**应如图所示（forward 越长，IC 越低）：

```
IC
0.06 |*
0.05 | *
0.04 |   *
0.03 |     *
0.02 |       * *
     +-----------→ forward（天）
       1  5  10 21 60
```

**异常形态**（触发诊断的场景）：

```
IC
0.09 |           *
0.06 |        *
0.04 |     *
0.03 |   *
0.00 | *
     +-----------→ forward（天）
       1  5  10 21 60
```

这种"反向衰减"可能来自以下原因（优先级从高到低）：

| 优先级 | 根因 | 风险等级 |
|-------|------|--------|
| 1 | 前瞻偏差 / 时间对齐错误 | HIGH |
| 2 | 累计收益率窗口的统计放大效应 | MEDIUM |
| 3 | 市值 / 行业结构暴露 | HIGH |
| 4 | 生存偏差（长 forward 样本量萎缩） | HIGH |
| 5 | 因子本身具备真实中期预测力 | LOW（正常） |

---

## 2. 六模块框架概览

```
┌──────────────────────────────────────────────────────┐
│                ICDecayDiagnostics                    │
│                                                      │
│  M1: 时间对齐   → 排查前瞻偏差（lag=0/1/2 对比）      │
│  M2: 累计/增量  → 排查统计放大效应                    │
│  M3: 暴露剥离   → 排查市值/行业驱动                   │
│  M4: 样本偏差   → 排查生存偏差/覆盖率下降              │
│  M5: 时效性     → 验证因子半衰期与 IC 形态的一致性     │
│  M6: 稳健性     → 子期/参数扰动/市场状态分析           │
└──────────────────────────────────────────────────────┘
```

每个模块输出一个 `DiagnosticResult`，包含：
- `passed`: `True` / `False` / `None`（数据不足）
- `risk_level`: `LOW` / `MEDIUM` / `HIGH` / `UNKNOWN`
- `evidence`: 定量证据（DataFrame 或 dict）
- `conclusion`: 文字结论

---

## 3. 快速上手

### 3.1 CLI 一键运行

```bash
# 单因子分析 + IC 衰减诊断
mf single --factor value_pb --ic-decay-diagnostics

# 指定时间范围
mf single --factor value_pb --ic-decay-diagnostics \
    --start 20200101 --end 20251231

# 使用动态股票池
mf single --factor value_pb --ic-decay-diagnostics \
    --universe-mode topn_mktcap_dynamic --universe-top-n 500
```

诊断结果自动保存到：
```
artifacts/factor_analysis/value_pb/
  ic_decay_diagnostics/
    diagnostic_overview.csv
    final_judgement.json
    module1_alignment_audit.csv
    module2_cumulative_ic.csv
    module2_incremental_ic.csv
    module3_neutralized_compare.csv
    module4_survivorship_audit.csv
    module5_factor_horizon_profile.csv
    module5_incr_ic_by_k.csv
    module6_split_period_ic.csv
    module6_winsor_sensitivity.csv
    module6_regime_ic.csv
```

### 3.2 Python API 调用

```python
from factor_framework.analytics.ic_decay_diagnostics import ICDecayDiagnostics

diag = ICDecayDiagnostics(
    factor_panel = factor_panel,      # (T×N) 日频，未经 T+1 shift
    price_panel  = price_panel,       # (T×N) 后复权收盘价
    forward_list = [1, 5, 10, 21, 60],
    industry_map = engine.industry_map,   # pd.Series: code → industry（可选）
    mktcap_panel = mktcap_panel,          # (T×N) 市值面板（可选）
    ic_method    = "rank",
    factor_name  = "value_pb",
)

# 运行全部 6 模块
report = diag.run_all(verbose=True)

# 只运行指定模块
report = diag.run_all(run_modules=[1, 2, 3], verbose=True)

# 打印完整报告
report.print_full()

# 保存结果
from pathlib import Path
from factor_framework.pipeline import _save_diag_report
_save_diag_report(report, Path("output/value_pb/ic_decay_diagnostics"))

# 序列化为 dict（可转 JSON）
import json
d = report.to_dict()
json.dumps(d, ensure_ascii=False, indent=2, default=str)
```

### 3.3 从已有 FactorReport 调用

```python
from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline(stocks_dir="stocks/stocks/")
pipe.register_builtins(["value_pb"])

# 方式一：在 run() 时一并触发诊断（推荐）
report = pipe.run(
    factor_name             = "value_pb",
    run_ic_decay_diagnostics = True,
)
# 诊断报告已缓存在 report._diag_report
# save() 时自动保存到 ic_decay_diagnostics/ 子目录
report.save("artifacts/factor_analysis")

# 方式二：先 run()，再手动诊断
report = pipe.run(factor_name="value_pb", run_ic_decay_diagnostics=True)
diag_report = report.run_ic_diagnostics(
    forward_list = [1, 5, 10, 21, 60],
    save_dir     = "artifacts/factor_analysis/value_pb/ic_decay_diagnostics",
)
```

> **注意**：`run_ic_decay_diagnostics=True` 时，`pipe.run()` 会额外构建
> `price_panel` 和 `mktcap_panel` 并附加到 `report` 对象中，供诊断模块使用。

---

## 4. 模块详解

### M1：时间对齐与前瞻偏差检查

**核心问题**：因子计算时是否使用了未来数据？

**检测逻辑**：

```
lag=0：factor[t] vs ret[t]         （因子含当日信息 → 前瞻）
lag=1：factor[t] vs ret[t+1:]      （标准 T+1 对齐 → 正确）
lag=2：factor[t-1] vs ret[t+1:]    （额外滞后 1 天 → 过于保守）
```

**判断标准**：

| 条件 | 结论 | 风险 |
|------|------|------|
| IC(lag=0) - IC(lag=1) > 0.010 | 疑似前瞻偏差 | `FAIL / HIGH` |
| \|IC(lag=1) - IC(lag=2)\| > 0.010 | lag=1 对齐可能不充分 | `FAIL / HIGH` |
| 两者差值均 ≤ 0.010 | 时间对齐正确 | `PASS / LOW` |

**典型输出**（`evidence` DataFrame，indexed by `[forward, lag]`）：

```
            mean_ic  icir   t_stat
forward lag
1       0   0.0312   0.452  3.12
        1   0.0298   0.431  2.98
        2   0.0291   0.420  2.91
60      0   0.1521   1.823  ...
        1   0.0884   1.228  ...
        2   0.0877   1.218  ...
```

---

### M2：收益定义与累计窗口拆解

**核心问题**：长 forward IC 较高，是因为因子真的能预测中期收益，
还是累计收益率天然包含更多信息（重叠窗口统计放大）？

**检测逻辑**：

- **累计 IC**：`IC(f_t, R_{t+1:t+h})`（与主 pipeline 一致）
- **增量 IC**：`IC(f_t, r_{t+k})`（第 k 天的单日收益）

若增量 IC 在 k=5 后趋近 0，但累计 IC 仍持续上升，说明是统计放大效应。

**判断标准**：

| 条件 | 结论 | 风险 |
|------|------|------|
| `incr_IC(k=max)` < 50% × `incr_IC(k=1)` 且中点 < 75% | 统计放大效应 | `FAIL / MEDIUM` |
| `incr_IC(k=max)` ≥ 50% × `incr_IC(k=1)` | 真实中期信号 | `PASS / LOW` |

**典型输出**（`evidence["incr_ic"]` Series）：

```
k
1     0.0312
2     0.0251
3     0.0198
5     0.0145
10    0.0089
21    0.0042
30    0.0031
60   -0.0033
Name: incr_ic
```

---

### M3：市场/风格暴露剥离

**核心问题**：IC 是否因为因子与市值/行业高度相关（而非独立信号）而虚高？

**四个版本 IC 对比**：

| 版本 | 处理方式 | 目的 |
|------|---------|------|
| V0 | 原始 IC | 基准 |
| V1 | 收益率截面去均值 | 排除市场 β |
| V2 | 收益率行业内超额 | 排除行业暴露 |
| V3 | 因子对 log(mktcap) 回归取残差 | 排除市值因子暴露 |
| V4 | 因子对 log(mktcap) + 行业哑变量回归取残差 | 最严格双重中性化 |

**判断标准**：

| 条件 | 结论 | 风险 |
|------|------|------|
| V4 IC 上升幅度 < 30% 原始 | 结构暴露驱动 | `FAIL / HIGH` |
| V4 IC 上升幅度 30~70% 原始 | 部分暴露 | `N/A / MEDIUM` |
| V4 IC 上升幅度 ≥ 70% 原始 | 独立因子信号 | `PASS / LOW` |

---

### M4：样本偏差检查

**核心问题**：forward=60 的样本是否因为股票退市/停牌而偏向"存活"股票？

**检测逻辑**：

- 计算各 forward 的联合有效覆盖率（factor 和 return 均非 NaN 的比例）
- 比较 forward=60 vs forward=21 的覆盖率差异

**判断标准**：

| 覆盖率差距 | 结论 | 风险 |
|-----------|------|------|
| > 10% | 生存偏差显著 | `FAIL / HIGH` |
| 5~10% | 中等下降，需关注 | `N/A / MEDIUM` |
| ≤ 5% | 覆盖率稳定 | `PASS / LOW` |

---

### M5：因子属性验证（时效性）

**核心问题**：因子的信号持续时间（半衰期）是否与 IC 形态匹配？

**检测逻辑**：

1. 计算因子月末截面排名的 lag=1..12 期自相关
2. 拟合指数衰减 ρ(k) = ρ₀·e^(-λk)，估算半衰期 τ = ln(2)/λ（月）
3. 计算增量 IC 的有效预测半衰期（IC 降至 50% 初始值时的 k 值）

**判断标准**：

| 条件 | 结论 | 风险 |
|------|------|------|
| 半衰期 ≥ 60 交易日 | 缓变因子，长 forward IC 高属于合理 | `PASS / LOW` |
| 半衰期 < 21 日 且 IC(k=max)/IC(k=mid) > 1.30 | 时效与 IC 形态不匹配 | `FAIL / MEDIUM` |
| 其他 | 中等时效，综合判断 | `N/A / MEDIUM` |

---

### M6：稳健性复核

**核心问题**：IC 随 forward 上升的现象是否在不同时间段、参数设置、市场状态下均成立？

**三个维度**：

1. **时间外推**：将样本均分为 3 个子期（默认），逐子期检查单调性
2. **Winsorize 扰动**：严格（1.5 MAD）/ 标准（3 MAD）/ 宽松（5 MAD）
3. **市场状态（Regime）**：牛市（年化 >10%）/ 熊市（< -10%）/ 震荡

**判断标准**：

| 单调子期比例 | 结论 | 风险 |
|------------|------|------|
| ≥ 2/3 | 结论稳健 | `PASS / LOW` |
| 1/3 ~ 2/3 | 部分稳健 | `N/A / MEDIUM` |
| ≤ 1/3 | 依赖特定时间段 | `FAIL / MEDIUM` |

---

## 5. 最终判定口径

只有在**以下所有条件均满足**时，才判定为"真实中期有效"：

```
✅ M1 PASS — 无前瞻/错位
✅ M2 PASS — 增量 IC 链条支持持续有效
✅ M3 PASS — 中性化后仍显著（独立因子信号）
✅ M4 PASS — 样本偏差审计通过（覆盖率稳定）
✅ M5 PASS — 因子半衰期与 IC 形态一致
✅ M6 PASS — 样本外与分市场状态不反转
```

否则，默认按照如下优先级判定：

| 失败模块 | 最终判定 | 建议行动 |
|---------|---------|---------|
| M1 或 M2 FAIL | **实现偏差**（HIGH） | 修复 T+1 对齐，使用增量 IC 验证 |
| M3 FAIL | **结构暴露驱动**（HIGH） | 使用双重中性化后的因子重新回测 |
| M4 FAIL | **生存偏差**（HIGH） | 检查退市/停牌的 NaN 填充策略 |
| M5 FAIL | **时效不匹配**（MEDIUM） | 缩短 ic_forward_list，专注短期 |
| M6 FAIL | **时间不稳定**（MEDIUM） | 按市场状态分别评估因子有效性 |
| ≥3 模块 FAIL | **多因素混合偏差**（HIGH） | 逐一修复后重测 |

---

## 6. 输出文件规范

```
ic_decay_diagnostics/
├── diagnostic_overview.csv         # 6 模块状态汇总表
│     columns: module_id, module_name, passed, risk_level, conclusion
├── final_judgement.json            # 最终判定（机器可读）
│     keys: factor_name, verdict, risk_level, pass_count, failed_modules
├── module1_alignment_audit.csv     # M1: lag=0/1/2 IC 对比
│     index: [forward, lag], columns: mean_ic, icir, t_stat, n
├── module2_cumulative_ic.csv       # M2: 累计 IC 统计
│     index: forward, columns: cumul_ic, cumul_icir, t_stat, nw_t_stat
├── module2_incremental_ic.csv      # M2: 增量 IC 明细
│     index: k, columns: incr_ic, incr_icir, t_stat
├── module3_neutralized_compare.csv # M3: 四版本 IC 对比
│     index: forward, columns: ic_raw, ic_mkt_excess, ic_ind_excess,
│                              ic_mktcap_neut_factor, ic_dual_neut_factor
├── module4_survivorship_audit.csv  # M4: 覆盖率审计
│     index: forward, columns: n_dates, n_stocks, factor_coverage,
│                              ret_coverage, joint_coverage, n_monthly_periods
├── module5_factor_horizon_profile.csv  # M5: 月度自相关
│     index: lag_months, columns: autocorr
├── module5_incr_ic_by_k.csv       # M5: 增量 IC 检查点
│     index: k, columns: incr_ic
├── module6_split_period_ic.csv    # M6: 子期 IC 衰减
│     index: split, columns: start, end, ic_fwd1, ..., ic_fwd60
├── module6_winsor_sensitivity.csv # M6: Winsorize 扰动
│     columns: winsor_mads, label, mean_ic, icir, t_stat
└── module6_regime_ic.csv          # M6: 市场状态 IC
      columns: regime, n_periods, ic_fwd1, ..., ic_fwd60
```

---

## 7. 判断阈值参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lag_list` | `[0, 1, 2]` | M1 的 lag 组 |
| M1 泄露阈值 | `0.010` | IC(lag=0)-IC(lag=1) 超过此值 → FAIL |
| M2 衰减阈值 | `50% / 75%` | incr_IC(kmax)/incr_IC(k1) 的双重门槛 |
| M3 暴露阈值 | `30% / 70%` | V4 IC 上升幅度与原始的比值 |
| M4 覆盖率阈值 | `5% / 10%` | forward=60 vs 21 的覆盖率差距 |
| M5 半衰期门槛 | `60 / 21` | 缓变/快变因子（交易日） |
| M5 IC比例门槛 | `1.30` | IC(kmax)/IC(kmid) 异常上升 |
| `n_splits` | `3` | M6 子期数量 |
| M6 单调性阈值 | `2/3 / 1/3` | 稳健/不稳健子期比例 |

---

## 8. 常见根因与修复建议

### 根因 1：T+1 未对齐（M1 FAIL）

**现象**：IC(lag=0) 远大于 IC(lag=1)（差值 > 0.01）

**根本原因**：因子使用了收盘价/当日数据，但收益率从同一日起算，
导致因子"看见了"当日涨跌信息。

**修复**：
```python
# 确认 build_return_panel 内含 T+1 shift
# 在 ICDecayDiagnostics 中：factor_panel 传入的是未经 T+1 的原始面板
# pipeline.py 中的 build_return_panel 已内置 T+1，无需手动 shift
```

### 根因 2：累计统计放大（M2 FAIL）

**现象**：增量 IC 在 k=5 后趋近 0，但累计 IC 仍上升

**根本原因**：60 日累计收益 = Σ(各日收益)，包含相关成分，
导致用累计收益计算的 IC 自然高于用单日收益计算的 IC。

**修复**：改用增量 IC 作为因子评价指标；
或在报告中注明"累计 IC 受重叠窗口放大，实际有效期约 k 天"。

### 根因 3：市值/行业暴露（M3 FAIL）

**现象**：V4 双重中性化后 IC 上升现象消失

**根本原因**：P/B 因子与市值高度相关，大市值股票在长期表现优异，
导致 P/B 的长 forward IC 虚高。

**修复**：使用 `neutralize=True` 参数，或在因子构建阶段做市值调整。

### 根因 4：生存偏差（M4 FAIL）

**现象**：forward=60 的覆盖率比 forward=21 低 >10%

**根本原因**：退市/ST 股票在 60 日后无收益数据，被排除出样本，
剩余的"存活股票"整体表现偏好。

**修复**：检查 NaN 填充策略；使用滚动建仓（避免在退市前统一计算 60 日后收益）。

---

## 9. 附录：value_pb 诊断示例

以下为在 `value_pb` 因子（动态股票池 top500，2020-2025）上的典型诊断结果：

**观察到的异常 IC 形态**：

```
forward  mean_ic   icir
1        0.000165  0.0020
5        0.028428  0.3277
10       0.040535  0.4531
21       0.057077  0.6388
60       0.088429  1.2275
```

**预期诊断结论**（供参考，实际以运行结果为准）：

- **M1**：由于 pipeline 已内置 T+1，预期 PASS
- **M2**：P/B 为月度重采样因子，信号持续性较长；
  若增量 IC 在 k>21 后仍有一定幅度，则部分属于真实中期信号
- **M3**：P/B 与市值高度负相关（价值股往往市值较小）；
  预期双重中性化后 IC 上升幅度有所收窄，但可能仍显著
- **M4**：60 日内退市率相对稳定，预期覆盖率下降 < 5%（PASS）
- **M5**：P/B 为月度因子，自相关半衰期通常 > 3 个月（PASS）
- **M6**：按子期分析，牛市期间 IC 较强，熊市期间可能减弱

**如何运行**：

```bash
mf single --factor value_pb \
    --ic-decay-diagnostics \
    --universe-mode topn_mktcap_dynamic \
    --universe-top-n 500 \
    --start 20200101 --end 20251231
```
