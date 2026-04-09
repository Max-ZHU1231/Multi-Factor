"""
factor_framework
================
多因子选股框架，包含以下模块：

  operators.py      - 时间序列、横截面、数学逻辑、跨资产算子库
  factor_engine.py  - 因子注册 / 计算 / 面板构建引擎
  neutralize.py     - 因子中性化（市值、行业、波动率）
  ic_analysis.py    - IC 分析、t 检验、IC 衰减
  backtest.py       - 分层回测、多空组合、夏普/最大回撤/Calmar
  factor_zoo.py     - 内置预定义因子示例库
  optimizer.py      - 因子组合与权重优化（等权、ICIR 加权）
  pipeline.py       - 端到端 Pipeline：加载→清洗→计算→检验→输出

快速开始
--------
from factor_framework.pipeline import FactorPipeline
pipe = FactorPipeline(stocks_dir='Stocks/', stock_basic='股票列表-stock_basic.csv')
# 单因子检验
pipe.run(factor_name='momentum_12_1')
# 多因子合成
pipe.run_composite(['momentum_12_1', 'size_log_mktcap'], method='icir')
"""

from factor_framework.optimizer import equal_weight, icir_weight, print_weights
