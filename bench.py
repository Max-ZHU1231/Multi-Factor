"""
bench.py  —— 四项优化的端到端性能基准测试
运行：.venv\\Scripts\\python.exe bench.py
"""
import time, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline("stocks/", "股票列表-stock_basic.csv", verbose=False)
pipe.register_builtins(["momentum_12_1", "value_pb", "vol_20d",
                        "reversal_1m", "size_log_mktcap"])

START, END = "20240101", "20241231"
FACTORS    = ["momentum_12_1", "value_pb", "vol_20d",
              "reversal_1m", "size_log_mktcap"]

# ── 测试 1：单因子（衡量缓存首次加载速度）────────────────────────────────
print("\n[测试1] 单因子首次运行（momentum_12_1）")
t0 = time.perf_counter()
r  = pipe.run("momentum_12_1", start=START, end=END, forward=21)
t1 = time.perf_counter()
print(f"  耗时: {t1-t0:.1f}s  |  IC均值={r.ic_stats_['mean_ic']:.4f}")

# ── 测试 2：同一因子第二次运行（测缓存命中速度）────────────────────────
print("\n[测试2] 同因子再次运行（缓存命中）")
t0 = time.perf_counter()
r  = pipe.run("momentum_12_1", start=START, end=END, forward=21)
t1 = time.perf_counter()
print(f"  耗时: {t1-t0:.1f}s  （应远快于测试1）")

# ── 测试 3：5 因子批量（测缓存跨因子复用）──────────────────────────────
print(f"\n[测试3] {len(FACTORS)} 因子批量 run_batch（第2个起命中缓存）")
t0  = time.perf_counter()
df  = pipe.run_batch(FACTORS, start=START, end=END, forward=21)
t1  = time.perf_counter()
print(f"  总耗时: {t1-t0:.1f}s  |  平均每因子: {(t1-t0)/len(FACTORS):.1f}s")
print(df[["mean_ic","icir"]].to_string())

# ── 测试 4：compute_ic 向量化速度（直接调用）────────────────────────────
import numpy as np, pandas as pd
from factor_framework.ic_analysis import compute_ic

rng  = np.random.default_rng(0)
T, N = 500, 3000
fp   = pd.DataFrame(rng.standard_normal((T, N)),
                    index=[f"d{i}" for i in range(T)],
                    columns=[f"S{i}" for i in range(N)])
rp   = pd.DataFrame(rng.standard_normal((T, N)),
                    index=fp.index, columns=fp.columns)

print(f"\n[测试4] compute_ic 向量化（T={T}, N={N}）")
t0 = time.perf_counter()
for _ in range(5):
    compute_ic(fp, rp, method="rank")
t1 = time.perf_counter()
print(f"  5次均值耗时: {(t1-t0)/5*1000:.1f}ms  （原循环版约需数秒）")

print("\n基准测试完成 ✓")
