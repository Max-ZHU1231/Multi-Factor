"""
factor_framework.factors
========================
Phase 3: 因子元数据层 + 按类别拆分的因子文件。

子模块
------
meta.py       : FactorMeta dataclass, FactorCategory 枚举（10 类）
registry.py   : FactorRegistry, _CompatDict, REGISTRY 全局实例
momentum.py   : 动量/反转因子函数（re-export from factor_zoo）
volatility.py : 波动率因子函数（re-export from factor_zoo）
value.py      : 估值/规模因子函数（re-export from factor_zoo）
volume.py     : 量价/流动性/技术分析因子函数（re-export from factor_zoo）
transform.py  : TransformPipeline —— 可组合的横截面变换管道
ic_analyzer.py     : ICAnalyzer  —— 结构化 IC 分析封装
layer_backtester.py: LayerBacktester —— 分层回测封装

使用方式
--------
    from factor_framework.factors.meta import FactorMeta, FactorCategory
    from factor_framework.factors.registry import REGISTRY

    meta = REGISTRY.get("momentum_12_1")

    from factor_framework.factors.transform import TransformPipeline
    from factor_framework.factors.ic_analyzer import ICAnalyzer
    from factor_framework.factors.layer_backtester import LayerBacktester

注意：此 __init__.py 在包初始化时不主动导入任何内容，
以避免与 factor_zoo.py 之间的循环初始化依赖。
factor_zoo.py 在自身初始化末尾才导入 factors.meta / factors.registry，
单向依赖，无循环。类别文件（momentum.py 等）按需导入 factor_zoo，
同样是单向依赖。

imports factor_framework.factors.meta/registry during its own module-level
execution, which would re-trigger this __init__ before it finishes.
"""
# No imports here -- see docstring above.


