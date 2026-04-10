"""
factor_framework
================
多因子选股框架，包含以下模块：

  operators.py      - 时间序列、横截面、数学逻辑、跨资产算子库
                      时序: ts_ema, ts_slope, ts_rsi, ts_drawdown, ts_beta,
                            ts_regression_residual, ts_decay_linear, ts_prod
                      横截面: cs_rank_by_group, cs_neutralize, cs_top_n, cs_quantile
  factor_engine.py  - 因子注册 / 计算 / 面板构建引擎（DataFrame 缓存，DAG 依赖管理）
  dag.py            - DAG 节点类、LRU 缓存、DAGExecutor、DepGraph、CSE 报告
  neutralize.py     - 因子中性化（市值、行业、波动率、Beta、动量、流动性；OLS / WLS）
  ic_analysis.py    - IC 分析、t 检验、IC 衰减（向量化实现）
  backtest.py       - 分层回测、多空组合、夏普/最大回撤/Calmar
  factor_zoo.py     - 内置预定义因子库（28 个，含流动性质量和技术分析因子）
  optimizer.py      - 因子组合与权重优化（等权、ICIR 加权）
  pipeline.py       - 端到端 Pipeline：加载→清洗→计算→检验→输出
  core/             - Phase 2 数据结构基础层（TimestampedPanel, ReturnPanel）
  engine/           - Phase 2 缓存层（CacheLayer L1+L2）
  factors/          - Phase 3 元数据层（FactorMeta, FactorRegistry, _CompatDict）

快速开始
--------
from factor_framework.pipeline import FactorPipeline
pipe = FactorPipeline(stocks_dir='Stocks/', stock_basic='股票列表-stock_basic.csv')
# 单因子检验
pipe.run(factor_name='momentum_12_1')
# 多因子合成
pipe.run_composite(['momentum_12_1', 'size_log_mktcap'], method='icir')

# Phase 3：查询因子元数据
from factor_framework.factors.registry import REGISTRY
meta = REGISTRY.get('momentum_12_1')
print(meta.display_name, meta.category, meta.warmup_days)
df = REGISTRY.summary_df()   # 所有 28 个因子的元数据 DataFrame
"""

from factor_framework.optimizer import equal_weight, icir_weight, print_weights
from factor_framework.dag import (
    Expr, DataNode, OpNode, BinOpNode, ConstNode, PctChangeNode,
    data, const, op, pct_change,
    DAGExecutor, DepGraph, LRUCache,
    cse_report,
)
# Phase 3 symbols -- imported lazily to avoid circular initialisation.
# Access via: from factor_framework.factors import FactorMeta, FactorRegistry
# or:         from factor_framework.factors.registry import REGISTRY

