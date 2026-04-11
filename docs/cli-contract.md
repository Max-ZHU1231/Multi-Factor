# CLI Contract — `mf` Command (v4.0)

> **文档优先**：实现必须与本文档保持一致。任何行为变更须先修改此文档。

---

## 安装与调用

```bash
pip install -e .          # 注册 mf console_script
mf <command> [options]    # 主路径
python -m factor_framework.cli <command> [options]   # 兼容路径
```

---

## 退出码规范

| 退出码 | 含义 |
|--------|------|
| `0` | 成功 |
| `1` | 运行失败（数据错误、计算异常等） |
| `2` | 参数错误（缺少必填参数、非法值） |

> 退出码 `2` 由 `argparse` 自动处理（invalid argument → exit 2）。  
> 业务异常统一捕获后以退出码 `1` 退出，同时打印 `[ERROR]` 前缀的消息到 stderr。

---

## 子命令一览

| 命令 | 用途 | v4.0 状态 |
|------|------|-----------|
| `mf single` | 单因子 IC + 分层回测 | ✅ 已实现 |
| `mf batch`  | 全批量因子验证 | ✅ 已实现 |
| `mf validate` | look-ahead / 数据质量验证套件 | ✅ 已实现 |
| `mf cache`  | 缓存管理（info / clear / gc） | ✅ 已实现 |
| `mf composite` | 多因子组合优化 | 🔜 v4.1 |
| `mf report` | artifact 报告生成 | 🔜 Phase D |

---

## `mf single` — 单因子筛选

```
mf single --factor FACTOR [FACTOR ...] [options]
```

### 必填

| 参数 | 说明 |
|------|------|
| `--factor NAME` | 因子名称（可多个，用空格分隔） |

### 可选

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--start YYYYMMDD` | `20200101` | 回测起始日期 |
| `--end YYYYMMDD` | `20251231` | 回测截止日期 |
| `--forward N` | `21` | 预测期（交易日） |
| `--n-groups N` | `5` | 分层数 |
| `--config PATH` | — | 用户 YAML 配置（叠加在 default.yaml 之上） |
| `--output DIR` | `output/single/<factor>` | 输出目录 |
| `--no-cache` | off | 禁用 L2 Parquet 缓存 |
| `--show-config` | off | 打印有效配置后退出（退出码 0） |
| `--quiet` | off | 只输出 summary，不打印进度 |

### 输出

```
<output>/
  ic_series.csv         每期 IC 时间序列
  ic_summary.csv        IC 均值 / ICIR / t-stat
  layer_stats.csv       各层累计收益 / Sharpe
  run_manifest.json     运行配置快照（Phase D 完整实现）
```

### 退出码

| 情形 | 退出码 |
|------|--------|
| 正常完成 | `0` |
| 因子名不存在于 REGISTRY | `1` |
| 数据不足（股票数 < min_stocks） | `1` |
| 参数非法（如 `--forward -1`） | `2` |

### 示例

```bash
mf single --factor momentum_12_1
mf single --factor vwap_deviation price_strength --start 20210101
mf single --factor value_pb --forward 10 --no-cache --output /tmp/test_run
```

---

## `mf batch` — 全批量因子验证

```
mf batch [options]
```

### 可选

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--factors NAME [...]` | 全部 REGISTRY 因子 | 指定子集 |
| `--start YYYYMMDD` | `20200101` | |
| `--end YYYYMMDD` | `20251231` | |
| `--forward N` | `21` | |
| `--n-groups N` | `5` | |
| `--config PATH` | — | |
| `--output DIR` | `artifacts/batch_results` | |
| `--no-cache` | off | |
| `--parallel N` | `1` | 并行 worker 数（`-1` = CPU 核数） |
| `--show-config` | off | |
| `--quiet` | off | |

### 输出

```
<output>/
  factor_screening_summary.csv   所有因子的 IC / Sharpe 汇总表
  <factor>/                      各因子子目录（结构同 mf single）
  run_manifest.json
```

### 退出码

| 情形 | 退出码 |
|------|--------|
| 所有因子成功 | `0` |
| 部分因子失败（>0 个跳过） | `0`（失败因子写入 summary 的 error 列） |
| 全部因子失败 | `1` |
| 参数非法 | `2` |

---

## `mf validate` — 验证套件

```
mf validate [--suite {lookahead,quality,all}]
```

### 可选

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--suite` | `all` | 运行哪个套件 |
| `--verbose` | off | 输出详细断言信息 |

### 退出码

| 情形 | 退出码 |
|------|--------|
| 全部通过 | `0` |
| 有失败项 | `1` |

---

## `mf cache` — 缓存管理

```
mf cache {info,clear,gc} [--factor FACTOR] [--dir DIR]
```

| action | 说明 |
|--------|------|
| `info` | 打印缓存目录大小和条目数 |
| `clear` | 删除全部 L2 Parquet 缓存（可 `--factor` 限定） |
| `gc` | 删除超过 `--days N`（默认 30）天未访问的条目 |

### 退出码：`0` 成功，`1` 失败。

---

## 旧脚本兼容性（Shim 层）

| 旧命令 | 转发至 | 移除版本 |
|--------|--------|---------|
| `scripts/run_analysis.py` | `mf single` | v4.2 |
| `scripts/run_batch.py` | `mf batch` | v4.2 |
| `scripts/run_validation.py` | `mf validate` | v4.2 |
| `run-analysis`（console_script） | `mf single` | v4.2 |
| `run-batch`（console_script） | `mf batch` | v4.2 |

旧脚本在转发前会打印：
```
⚠️  scripts/run_batch.py 已弃用，将在 v4.2 移除。
   请改用: mf batch [同等参数]
```
