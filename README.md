
---

## 三层编译引擎（v2.5）

jit_ops.py 实现了三层因子表达式加速策略。

**层级 1：Numba JIT**（compile_target='numba'）：ts_mean/ts_sum/ts_rank 等滚动算子，比 Pandas rolling 快 5~10x。

**层级 2：Numexpr**（compile_target='numexpr'）：log/sqrt/power 等逐元素数学算子，比 NumPy 快 2~4x。

**层级 3：Pandas/NumPy 兜底**（compile_target='pandas'/'numpy'）：ts_ema/ts_rsi/cs_rank 等，直接使用 Pandas 内置方法。

降级保护：Numba/Numexpr 不可用时所有算子自动退化到 Pandas 路径。

预热：
`python

from factor_framework.jit_ops import warmup

warmup()  # 程序启动时调用，消除首次 JIT 编译延迟
`
