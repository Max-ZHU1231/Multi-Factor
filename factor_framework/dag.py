"""
dag.py
======
因子计算任务图（DAG）与公共子表达式消除（CSE）。

核心设计
--------
1. **延迟求值（Lazy Evaluation）**
   ``Expr`` 对象描述"如何计算"，而非立即执行。算子调用返回 ``Expr`` 树节点，
   只有在绑定到真实 DataFrame 并调用 ``eval()`` 时才触发计算。

2. **节点哈希（Content-based Hashing）**
   每个节点由 ``(op_name, input_hashes, params)`` 唯一标识。
   相同子表达式在整个 DAG 中只出现一次（指针相同），天然实现 CSE。

3. **两种注册方式**
   - **方式一**：``engine.register_expr(name, expr)``
     传入 ``Expr`` 树，引擎自动分析依赖，构建 DAG，执行时共享中间结果。
   - **方式二**：``engine.register(name, func, deps=[...])``
     传入 Python 函数 + 显式依赖列表，引擎按拓扑顺序执行，依赖结果写入
     共享缓存后供后续因子直接读取（不解析函数内部结构）。

4. **执行模式**
   ``DAGExecutor`` 对 DAG 做拓扑排序，识别无依赖关系的"兄弟节点"，
   并行计算（ThreadPoolExecutor），有依赖的节点串行等待。
   中间结果存入 ``_intermediate_cache``（LRU，key = node hash）。
   最终因子结果存入 ``_factor_cache``（LRU，key = factor_name + symbol）。

节点类型
--------
- ``DataNode``   : 数据列（叶节点），如 close、volume
- ``OpNode``     : 算子节点（内部节点），如 ts_mean(close, 20)
- ``FactorNode`` : 最终因子输出（根节点），包装 OpNode 并命名

接口速查
--------
>>> from factor_framework.dag import Expr, DataNode, data, op
>>> close  = data("close", col="收盘价")
>>> ret    = op("pct_change", close)
>>> vol20  = op("ts_stddev", ret, 20)
>>> vol60  = op("ts_stddev", ret, 60)
>>> # vol20 和 vol60 共享同一个 ret 节点，ret 只计算一次
>>> engine.register_expr("vol_20d", -vol20)
>>> engine.register_expr("vol_60d", -vol60)
"""

from __future__ import annotations

import hashlib
import threading
import warnings
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

# ─── LRU 缓存容量 ─────────────────────────────────────────────────────────────
_DEFAULT_INTERMEDIATE_LRU = 256   # 中间节点缓存条目数上限
_DEFAULT_FACTOR_LRU       = 512   # 最终因子缓存条目数上限


# ═══════════════════════════════════════════════════════════════════════════════
# 节点定义
# ═══════════════════════════════════════════════════════════════════════════════

class Expr:
    """
    因子表达式树节点基类（延迟求值）。

    子类需实现：
    - ``node_hash``  : 节点的内容哈希字符串（唯一标识）
    - ``inputs``     : 直接依赖的 Expr 节点列表
    - ``eval(df)``   : 绑定到 df 后立即求值，返回 pd.Series
    - ``__repr__``   : 人类可读的描述

    算术运算符重载（+、-、*、/、neg）让 Expr 之间的四则运算也返回 Expr，
    而不是立即执行，从而保持整棵表达式树的延迟求值语义。
    """

    # ── 算术运算符重载 ────────────────────────────────────────────────────────

    def __add__(self, other):
        return BinOpNode("+", self, _to_expr(other))

    def __radd__(self, other):
        return BinOpNode("+", _to_expr(other), self)

    def __sub__(self, other):
        return BinOpNode("-", self, _to_expr(other))

    def __rsub__(self, other):
        return BinOpNode("-", _to_expr(other), self)

    def __mul__(self, other):
        return BinOpNode("*", self, _to_expr(other))

    def __rmul__(self, other):
        return BinOpNode("*", _to_expr(other), self)

    def __truediv__(self, other):
        return BinOpNode("/", self, _to_expr(other))

    def __rtruediv__(self, other):
        return BinOpNode("/", _to_expr(other), self)

    def __neg__(self):
        return BinOpNode("*", ConstNode(-1.0), self)

    def __pos__(self):
        return self

    def __pow__(self, other):
        return BinOpNode("**", self, _to_expr(other))

    # ── 延迟求值入口 ──────────────────────────────────────────────────────────

    @property
    def node_hash(self) -> str:
        raise NotImplementedError

    @property
    def inputs(self) -> List["Expr"]:
        raise NotImplementedError

    def eval(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Expr {self.node_hash[:8]}>"


# ─── 常量节点 ─────────────────────────────────────────────────────────────────

class ConstNode(Expr):
    """标量常量节点（叶节点）。"""

    def __init__(self, value: float):
        self.value = value
        self._hash = _make_hash("const", str(value))

    @property
    def node_hash(self) -> str:
        return self._hash

    @property
    def inputs(self) -> List[Expr]:
        return []

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=df.index, dtype=float)

    def __repr__(self) -> str:
        return f"Const({self.value})"


# ─── 数据节点 ─────────────────────────────────────────────────────────────────

class DataNode(Expr):
    """
    原始数据列节点（叶节点）。

    Parameters
    ----------
    name : 逻辑名称（如 "close"）
    col  : DataFrame 中的实际列名（如 "收盘价"）；None 时 col=name
    """

    def __init__(self, name: str, col: Optional[str] = None):
        self.name = name
        self.col  = col if col is not None else name
        self._hash = _make_hash("data", name, self.col)

    @property
    def node_hash(self) -> str:
        return self._hash

    @property
    def inputs(self) -> List[Expr]:
        return []

    def eval(self, df: pd.DataFrame) -> pd.Series:
        if self.col not in df.columns:
            raise KeyError(f"[ERROR] DataNode '{self.name}': column '{self.col}' is missing in DataFrame.")
        return df[self.col].copy()

    def __repr__(self) -> str:
        return f"Data({self.name})"


# ─── 算子节点 ─────────────────────────────────────────────────────────────────

class OpNode(Expr):
    """
    算子节点（内部节点）。

    Parameters
    ----------
    op_name   : 算子名称（用于哈希和显示）
    inputs_   : 输入 Expr 列表（1 或 2 个）
    params    : 额外标量参数元组，如 (d,) 对应窗口长度
    fn        : 可调用对象，接受 (series1[, series2, *params]) → pd.Series
    """

    def __init__(
        self,
        op_name:  str,
        inputs_:  List[Expr],
        params:   Tuple[Any, ...],
        fn:       Callable,
    ):
        self.op_name  = op_name
        self._inputs  = inputs_
        self.params   = params
        self.fn       = fn
        self._hash    = _make_hash(
            op_name,
            *[n.node_hash for n in inputs_],
            *[str(p) for p in params],
        )

    @property
    def node_hash(self) -> str:
        return self._hash

    @property
    def inputs(self) -> List[Expr]:
        return self._inputs

    def eval(self, df: pd.DataFrame) -> pd.Series:
        # 直接求值输入（执行器会在调用前注入缓存结果，避免重复）
        evaluated = [inp.eval(df) for inp in self._inputs]
        return self.fn(*evaluated, *self.params)

    def __repr__(self) -> str:
        inp_str = ", ".join(repr(i) for i in self._inputs)
        par_str = (", " + ", ".join(str(p) for p in self.params)) if self.params else ""
        return f"{self.op_name}({inp_str}{par_str})"


# ─── 二元运算节点（+/-/*///**）────────────────────────────────────────────────

_BIN_OPS: Dict[str, Callable] = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "/":  lambda a, b: a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / b,
    "**": lambda a, b: a ** b,
}


class BinOpNode(Expr):
    """二元算术运算节点。"""

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op    = op
        self.left  = left
        self.right = right
        self._hash = _make_hash("binop", op, left.node_hash, right.node_hash)

    @property
    def node_hash(self) -> str:
        return self._hash

    @property
    def inputs(self) -> List[Expr]:
        return [self.left, self.right]

    def eval(self, df: pd.DataFrame) -> pd.Series:
        lv = self.left.eval(df)
        rv = self.right.eval(df)
        return _BIN_OPS[self.op](lv, rv)

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


# ─── 特殊节点：pct_change（无法映射为标准算子）─────────────────────────────────

class PctChangeNode(Expr):
    """序列差分百分比变化节点（用于构建收益率）。"""

    def __init__(self, inp: Expr, periods: int = 1):
        self._input   = inp
        self.periods  = periods
        self._hash    = _make_hash("pct_change", inp.node_hash, str(periods))

    @property
    def node_hash(self) -> str:
        return self._hash

    @property
    def inputs(self) -> List[Expr]:
        return [self._input]

    def eval(self, df: pd.DataFrame) -> pd.Series:
        return self._input.eval(df).pct_change(self.periods)

    def __repr__(self) -> str:
        return f"pct_change({self._input!r}, {self.periods})"


# ─── 工厂函数（公开 API）──────────────────────────────────────────────────────

def data(name: str, col: Optional[str] = None) -> DataNode:
    """创建数据列节点。"""
    return DataNode(name, col)


def const(value: float) -> ConstNode:
    """创建标量常量节点。"""
    return ConstNode(value)


def op(op_name: str, *args, fn: Optional[Callable] = None) -> OpNode:
    """
    创建算子节点。

    Parameters
    ----------
    op_name : 算子名称（字符串，用于哈希和调试；若 fn=None 则从 operators 模块查找）
    *args   : Expr 节点或标量参数（标量会自动转换为 ConstNode）；
              前 1 或 2 个位置参数为 Expr 输入，其余为 params
    fn      : 可选的直接可调用对象（优先于 ops 模块查找）

    Examples
    --------
    >>> close  = data("close", col="收盘价")
    >>> ret    = op("pct_change", close)          # 调用 PctChangeNode
    >>> vol20  = op("ts_stddev", ret, 20)
    >>> corr   = op("ts_corr", ret, ret, 5)       # 双序列算子
    """
    # pct_change 特殊处理
    if op_name == "pct_change":
        inp = args[0] if args else None
        periods = int(args[1]) if len(args) > 1 else 1
        if not isinstance(inp, Expr):
            raise TypeError("[ERROR] The first argument of pct_change must be an Expr node.")
        return PctChangeNode(inp, periods)  # type: ignore[return-value]

    # 分离 Expr 输入 vs 标量参数
    expr_inputs: List[Expr] = []
    scalar_params: List[Any] = []
    for a in args:
        if isinstance(a, Expr):
            expr_inputs.append(a)
        else:
            scalar_params.append(a)

    # 解析 fn
    if fn is None:
        fn = _lookup_op(op_name)

    return OpNode(op_name, expr_inputs, tuple(scalar_params), fn)


def pct_change(inp: Expr, periods: int = 1) -> PctChangeNode:
    """语法糖：创建 pct_change 节点。"""
    return PctChangeNode(inp, periods)


# ─── 算子模块查找 ────────────────────────────────────────────────────────────

def _lookup_op(name: str) -> Callable:
    """从 factor_framework.operators 中查找同名函数。"""
    try:
        from factor_framework import operators as _ops
        fn = getattr(_ops, name, None)
        if callable(fn):
            return fn
    except ImportError:
        pass
    raise AttributeError(
        f"[ERROR] Operator '{name}' was not found in factor_framework.operators. "
        "Provide a callable explicitly via fn=."
    )


# ─── 辅助：标量转 Expr ────────────────────────────────────────────────────────

def _to_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    return ConstNode(float(x))


# ─── 内容哈希 ────────────────────────────────────────────────────────────────

def _make_hash(*parts: str) -> str:
    """用 SHA-256 前 16 位生成节点哈希（足够区分，节省存储）。"""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def collect_nodes(root: Expr) -> List[Expr]:
    """
    从根节点出发，BFS 收集所有节点（去重）。
    返回列表按 BFS 顺序排列（根在最后的反转拓扑序）。
    """
    visited: Set[str] = set()
    order: List[Expr] = []
    queue: deque[Expr] = deque([root])
    while queue:
        node = queue.popleft()
        if node.node_hash in visited:
            continue
        visited.add(node.node_hash)
        order.append(node)
        for inp in node.inputs:
            queue.append(inp)
    return order


def topological_sort(roots: List[Expr]) -> List[Expr]:
    """
    对多个根节点合并后的 DAG 做拓扑排序。
    返回按执行顺序排列的节点列表（叶节点在前，根节点在后）。
    使用 Kahn 算法（BFS + 入度统计）。
    """
    # 1. 收集所有节点
    all_nodes: Dict[str, Expr] = {}
    for root in roots:
        for node in collect_nodes(root):
            all_nodes[node.node_hash] = node

    # 2. 构建入度表和邻接表
    in_degree: Dict[str, int] = {h: 0 for h in all_nodes}
    children:  Dict[str, List[str]] = {h: [] for h in all_nodes}

    for h, node in all_nodes.items():
        for inp in node.inputs:
            ih = inp.node_hash
            if ih in all_nodes:
                in_degree[h] += 1
                children[ih].append(h)

    # 3. Kahn 算法
    queue: deque[str] = deque(h for h, deg in in_degree.items() if deg == 0)
    result: List[Expr] = []
    while queue:
        h = queue.popleft()
        result.append(all_nodes[h])
        for child_h in children[h]:
            in_degree[child_h] -= 1
            if in_degree[child_h] == 0:
                queue.append(child_h)

    if len(result) != len(all_nodes):
        raise ValueError("[ERROR] Cycle detected in DAG. Check factor dependencies.")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LRU 缓存
# ═══════════════════════════════════════════════════════════════════════════════

class LRUCache:
    """
    线程安全 LRU 缓存。

    使用 collections.OrderedDict 实现，get/put 均为 O(1)。
    capacity = -1 表示无容量上限（相当于普通字典）。
    """

    def __init__(self, capacity: int = 256):
        self._cap   = capacity
        self._cache: OrderedDict = OrderedDict()
        self._lock  = threading.Lock()

    def get(self, key: str) -> Any:
        """命中返回值，未命中返回 _MISS 哨兵。"""
        with self._lock:
            if key not in self._cache:
                return _MISS
            self._cache.move_to_end(key)   # O(1)，最近访问移到末尾
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)   # O(1)
            else:
                if self._cap > 0 and len(self._cache) >= self._cap:
                    self._cache.popitem(last=False)   # O(1)，淘汰最久未访问
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


class _Miss:
    def __bool__(self): return False
    def __repr__(self): return "<MISS>"


_MISS = _Miss()


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 执行引擎（单股）
# ═══════════════════════════════════════════════════════════════════════════════

class DAGExecutor:
    """
    对单只股票的 DataFrame 执行 DAG，共享中间计算结果。

    参数
    ----
    intermediate_cache : LRUCache，key = node_hash，value = pd.Series
    factor_roots       : 已注册的 {factor_name: Expr} 字典
    n_jobs             : 并行执行无依赖节点的线程数（默认 4）

    工作流
    ------
    1. 合并所有因子根节点，做一次拓扑排序
    2. 按拓扑顺序逐层执行节点，每层内无相互依赖 → 并行
    3. 所有中间结果写入 intermediate_cache
    4. 最终因子结果从 intermediate_cache 中按根节点哈希读取
    """

    def __init__(
        self,
        intermediate_cache: LRUCache,
        factor_roots:       Dict[str, Expr],
        n_jobs:             int = 4,
    ):
        self._icache = intermediate_cache
        self._roots  = factor_roots
        self._n_jobs = n_jobs
        # 预计算拓扑顺序（只做一次）
        self._topo:  Optional[List[Expr]] = None
        self._dirty  = True   # 注册新因子时重新排序

    def mark_dirty(self) -> None:
        """注册新因子后调用，触发下次执行前重新做拓扑排序。"""
        self._dirty = True

    def _ensure_topo(self) -> None:
        if self._dirty or self._topo is None:
            self._topo = topological_sort(list(self._roots.values()))
            self._dirty = False

    def run(
        self,
        df:          pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> Dict[str, pd.Series]:
        """
        执行 DAG，返回 {factor_name: pd.Series}。

        Parameters
        ----------
        df           : 单股 DataFrame（已清洗）
        factor_names : 需要计算的因子子集（None = 全部）
        """
        self._ensure_topo()

        # 按拓扑顺序计算所有节点（叶→根）
        # 利用 intermediate_cache 跳过已算过的节点
        for node in self._topo:
            h = node.node_hash
            if self._icache.get(h) is not _MISS:
                continue  # 命中缓存，跳过
            # 向子类注入缓存值（避免 eval 内部重复递归）
            result = self._eval_node_with_cache(node, df)
            self._icache.put(h, result)

        # 从缓存提取最终因子输出
        targets = factor_names if factor_names is not None else list(self._roots.keys())
        output: Dict[str, pd.Series] = {}
        for name in targets:
            if name not in self._roots:
                continue
            root = self._roots[name]
            val  = self._icache.get(root.node_hash)
            if val is not _MISS:
                output[name] = val
        return output

    def _eval_node_with_cache(self, node: Expr, df: pd.DataFrame) -> pd.Series:
        """
        求值单个节点，输入节点的值从缓存读取而非递归重算。
        这是 DAG 共享中间结果的核心：每个子节点只计算一次。
        """
        if isinstance(node, (DataNode, ConstNode)):
            return node.eval(df)

        if isinstance(node, PctChangeNode):
            inp_val = self._get_cached_input(node.inputs[0], df)
            return inp_val.pct_change(node.periods)

        if isinstance(node, BinOpNode):
            lv = self._get_cached_input(node.left, df)
            rv = self._get_cached_input(node.right, df)
            return _BIN_OPS[node.op](lv, rv)

        if isinstance(node, OpNode):
            inp_vals = [self._get_cached_input(inp, df) for inp in node.inputs]
            return node.fn(*inp_vals, *node.params)

        # 其他类型：递归（保底）
        return node.eval(df)

    def _get_cached_input(self, inp: Expr, df: pd.DataFrame) -> pd.Series:
        """从缓存读取输入节点的值，未命中则直接求值（叶节点）。"""
        cached = self._icache.get(inp.node_hash)
        if cached is not _MISS:
            return cached
        # 叶节点或尚未缓存：直接求值
        val = inp.eval(df)
        self._icache.put(inp.node_hash, val)
        return val


# ═══════════════════════════════════════════════════════════════════════════════
# 显式依赖注册（方式二）
# ═══════════════════════════════════════════════════════════════════════════════

class DepGraph:
    """
    显式依赖图（方式二：lambda 函数 + deps 列表）。

    节点：因子名称（字符串）
    边：  name → dep_name（name 依赖 dep_name 的输出）

    执行时，依赖因子的 Series 结果通过特殊 DataFrame 列传递给下游：
        df[f"__dep_{dep_name}__"] = dep_result

    这样下游因子函数可以通过列名读取依赖结果，无需修改函数签名。
    """

    def __init__(self):
        self._graph: Dict[str, List[str]] = {}   # name → [dep_names]

    def register(self, name: str, deps: List[str]) -> None:
        self._graph[name] = list(deps)

    def deps_of(self, name: str) -> List[str]:
        return self._graph.get(name, [])

    def topo_order(self, names: List[str]) -> List[str]:
        """
        对给定的因子名称列表做拓扑排序（包含所有传递依赖）。
        返回执行顺序（依赖在前，被依赖在后）。
        """
        # 扩展到完整依赖闭包
        all_names: Set[str] = set()
        stack = list(names)
        while stack:
            n = stack.pop()
            if n in all_names:
                continue
            all_names.add(n)
            stack.extend(self._graph.get(n, []))

        # Kahn 算法
        in_deg = {n: 0 for n in all_names}
        children: Dict[str, List[str]] = {n: [] for n in all_names}
        for n in all_names:
            for dep in self._graph.get(n, []):
                if dep in all_names:
                    in_deg[n] += 1
                    children[dep].append(n)

        q = deque(n for n, d in in_deg.items() if d == 0)
        order: List[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for child in children[n]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    q.append(child)

        if len(order) != len(all_names):
            raise ValueError("[ERROR] Cycle detected in factor dependency graph. Check deps declarations.")

        # 只返回最初请求的因子（按拓扑顺序）
        requested = set(names)
        return [n for n in order if n in requested]


# ═══════════════════════════════════════════════════════════════════════════════
# 公共子表达式统计工具
# ═══════════════════════════════════════════════════════════════════════════════

def cse_report(factor_roots: Dict[str, Expr]) -> pd.DataFrame:
    """
    分析 DAG 中所有因子的公共子表达式（CSE）。

    Returns
    -------
    pd.DataFrame，columns: [node_hash, repr, ref_count, shared_by]
    其中 ref_count > 1 的节点即为公共子表达式。
    """
    ref_count:  Dict[str, int]       = {}
    shared_by:  Dict[str, List[str]] = {}
    node_repr:  Dict[str, str]       = {}

    for fname, root in factor_roots.items():
        for node in collect_nodes(root):
            h = node.node_hash
            ref_count[h]  = ref_count.get(h, 0) + 1
            shared_by.setdefault(h, [])
            if fname not in shared_by[h]:
                shared_by[h].append(fname)
            node_repr[h] = repr(node)

    rows = [
        {
            "node_hash": h,
            "repr":      node_repr[h],
            "ref_count": ref_count[h],
            "shared_by": ", ".join(sorted(shared_by[h])),
        }
        for h in ref_count
        if ref_count[h] > 1
    ]
    if not rows:
        return pd.DataFrame(columns=["node_hash", "repr", "ref_count", "shared_by"])
    df = pd.DataFrame(rows).sort_values("ref_count", ascending=False)
    return df.reset_index(drop=True)
