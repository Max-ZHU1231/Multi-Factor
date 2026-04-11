"""
factor_framework.research_config
=================================
ResearchConfig — 单次 pipeline.run() 调用的标准化配置对象。

设计目标（Phase C）
-------------------
* 最小闭环：覆盖 run() 当前最常用参数，不追求一次全包
* 双入口兼容：run(config=rc) 或旧 run(factor_name=..., forward=..., ...)
  两种调用方式均支持，内部统一转换为 ResearchConfig
* 配置升级器：upgrade_config(dict) -> dict，先做 no-op + version guard，
  为后续字段迁移（如重命名、类型变更）预留机制
* 哈希稳定：to_stable_dict() 排序键 + 去掉瞬时字段（symbols 等大列表）
  供 manifest._config_hash() 直接调用，保证跨会话稳定性

字段清单（v1）
--------------
必选
  factor_name      str        已注册的因子名称
  start            str        "YYYYMMDD" 或 None
  end              str        "YYYYMMDD" 或 None

可选（均有默认值，与 run() 保持一致）
  forward          int=21     预测期（天）
  n_groups         int=5      分层数
  direction        int=1      因子方向（+1 / -1）
  standardize      str="rank" 横截面标准化（"rank"/"zscore"/None）
  neutralize       bool=False 市值+行业中性化
  winsorize        bool=True  截面 MAD Winsorize
  ic_method        str="rank" IC 计算方式（"rank"/"normal"）
  periods_per_year int=252    年化期数
  rf               float=0.0  无风险利率（年化）
  cost_per_side    float=0.002 单边交易成本
  resample_monthly bool=True  月度重采样

元数据（不参与哈希）
  schema_version   str        配置协议版本（当前 "1.0"）
  symbols          List[str]|None  指定股票（None=全部；不计入稳定哈希）
  ic_forward_list  tuple       IC 衰减分析的预测期（不计入稳定哈希）

示例
----
    from factor_framework.research_config import ResearchConfig

    rc = ResearchConfig(
        factor_name = "momentum_12_1",
        start       = "20200101",
        end         = "20251231",
        forward     = 21,
        n_groups    = 5,
    )
    report = pipe.run(config=rc)
    # 或者旧方式（内部会自动转换为 ResearchConfig）：
    report = pipe.run(factor_name="momentum_12_1", start="20200101", end="20251231")
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# 当前协议版本——每次字段语义发生 breaking change 时递增
CURRENT_SCHEMA_VERSION: str = "1.0"

# 最低支持的 schema_version（低于此版本触发 upgrade 警告）
_MIN_SUPPORTED_VERSION: str = "1.0"


# ═════════════════════════════════════════════════════════════════════════════
#  配置升级器
# ═════════════════════════════════════════════════════════════════════════════

def upgrade_config(raw: dict) -> dict:
    """
    将旧格式配置字典升级到当前协议版本。

    当前为 no-op（v1.0 无需迁移）；未来字段重命名/合并/分裂时在此添加分支。

    Parameters
    ----------
    raw : 从 JSON / YAML / 旧代码传入的原始配置字典

    Returns
    -------
    升级后的字典（已注入 schema_version = CURRENT_SCHEMA_VERSION）

    Raises
    ------
    ValueError
        若 schema_version 存在但版本号格式非法（非数字点分）
    """
    d = copy.deepcopy(raw)
    version = str(d.get("schema_version", "")).strip()

    if version == "":
        # 缺失 schema_version → 视为最老版本，静默升级
        warnings.warn(
            "[ResearchConfig] 配置缺少 schema_version，视为旧版并自动升级至"
            f" {CURRENT_SCHEMA_VERSION}。建议在配置中显式指定 schema_version。",
            UserWarning,
            stacklevel=3,
        )
        d["schema_version"] = CURRENT_SCHEMA_VERSION
        return d

    # 版本比较（仅支持 "major.minor" 格式）
    try:
        def _ver(s):
            parts = s.split(".")
            return tuple(int(x) for x in parts)
        v_current = _ver(CURRENT_SCHEMA_VERSION)
        v_raw     = _ver(version)
        v_min     = _ver(_MIN_SUPPORTED_VERSION)
    except ValueError:
        raise ValueError(
            f"[ResearchConfig] schema_version 格式非法: {version!r}，"
            f"期望如 '1.0'。"
        )

    if v_raw < v_min:
        warnings.warn(
            f"[ResearchConfig] schema_version={version!r} 低于最低支持版本"
            f" {_MIN_SUPPORTED_VERSION!r}，已自动升级至 {CURRENT_SCHEMA_VERSION}。",
            UserWarning,
            stacklevel=3,
        )
        # ── future migration hooks ──────────────────────────────────────
        # if v_raw < (1, 0):
        #     d["new_field"] = d.pop("old_field", default)
        # ───────────────────────────────────────────────────────────────
        d["schema_version"] = CURRENT_SCHEMA_VERSION

    elif v_raw > v_current:
        warnings.warn(
            f"[ResearchConfig] schema_version={version!r} 高于当前代码版本"
            f" {CURRENT_SCHEMA_VERSION!r}，将按现有字段解析，未知字段被忽略。",
            UserWarning,
            stacklevel=3,
        )
        d["schema_version"] = CURRENT_SCHEMA_VERSION

    return d


# ═════════════════════════════════════════════════════════════════════════════
#  ResearchConfig
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ResearchConfig:
    """
    单次 pipeline.run() 调用的标准化配置对象（Phase C v1.0）。

    所有字段均与 ``FactorPipeline.run()`` 的同名参数一一对应。
    """

    # ── 必选字段 ──────────────────────────────────────────────────────────
    factor_name: str = ""

    # ── 日期范围 ──────────────────────────────────────────────────────────
    start: Optional[str] = None
    end:   Optional[str] = None

    # ── 回测参数（可选，有默认值）────────────────────────────────────────
    forward:          int   = 21
    n_groups:         int   = 5
    direction:        int   = 1
    standardize:      Optional[str] = "rank"
    neutralize:       bool  = False
    winsorize:        bool  = True
    ic_method:        str   = "rank"
    periods_per_year: int   = 252
    rf:               float = 0.0
    cost_per_side:    float = 0.002
    resample_monthly: bool  = True

    # ── 元数据（不参与稳定哈希）─────────────────────────────────────────
    schema_version:  str                = CURRENT_SCHEMA_VERSION
    symbols:         Optional[List[str]] = None          # 大列表，不计入哈希
    ic_forward_list: Tuple[int, ...]     = (1, 5, 10, 21, 60)

    # ── 工厂方法 ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchConfig":
        """
        从字典构造 ResearchConfig，自动调用 upgrade_config()。

        未知字段会被静默忽略（向前兼容）。
        """
        upgraded = upgrade_config(d)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in upgraded.items() if k in known}
        # ic_forward_list 需转为 tuple
        if "ic_forward_list" in filtered and not isinstance(filtered["ic_forward_list"], tuple):
            filtered["ic_forward_list"] = tuple(filtered["ic_forward_list"])
        return cls(**filtered)

    @classmethod
    def from_kwargs(
        cls,
        factor_name:      str,
        start:            Optional[str] = None,
        end:              Optional[str] = None,
        forward:          int = 21,
        n_groups:         int = 5,
        direction:        int = 1,
        standardize:      Optional[str] = "rank",
        neutralize:       bool = False,
        winsorize:        bool = True,
        ic_method:        str = "rank",
        ic_forward_list:  tuple = (1, 5, 10, 21, 60),
        periods_per_year: int = 252,
        rf:               float = 0.0,
        cost_per_side:    float = 0.002,
        symbols:          Optional[List[str]] = None,
        resample_monthly: bool = True,
        **_ignored,
    ) -> "ResearchConfig":
        """
        从 run() 旧式 kwargs 构造 ResearchConfig（向后兼容入口）。

        额外的未知关键字参数被静默忽略。
        """
        return cls(
            factor_name      = factor_name,
            start            = start,
            end              = end,
            forward          = forward,
            n_groups         = n_groups,
            direction        = direction,
            standardize      = standardize,
            neutralize       = neutralize,
            winsorize        = winsorize,
            ic_method        = ic_method,
            ic_forward_list  = tuple(ic_forward_list),
            periods_per_year = periods_per_year,
            rf               = rf,
            cost_per_side    = cost_per_side,
            symbols          = symbols,
            resample_monthly = resample_monthly,
        )

    # ── 序列化 ────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """返回完整字段字典（包含 schema_version 和 symbols）。"""
        d = asdict(self)
        # tuple → list（JSON 友好）
        d["ic_forward_list"] = list(d["ic_forward_list"])
        return d

    def to_stable_dict(self) -> dict:
        """
        返回用于哈希的稳定字典：

        - 排序所有键（保证 JSON 序列化字节顺序确定）
        - 去掉瞬时字段（symbols、ic_forward_list）
        - 保留 schema_version（用于跨版本哈希隔离）
        """
        _TRANSIENT = {"symbols", "ic_forward_list"}
        d = {k: v for k, v in self.to_dict().items() if k not in _TRANSIENT}
        return dict(sorted(d.items()))

    # ── 验证 ──────────────────────────────────────────────────────────────

    def validate(self) -> "ResearchConfig":
        """
        就地验证必填字段和参数范围，返回 self（支持链式调用）。

        Raises
        ------
        ValueError : factor_name 为空，或参数超出合理范围
        """
        if not self.factor_name:
            raise ValueError("ResearchConfig.factor_name 不能为空。")
        if self.forward < 1:
            raise ValueError(f"forward={self.forward} 必须 ≥ 1。")
        if self.n_groups < 2:
            raise ValueError(f"n_groups={self.n_groups} 必须 ≥ 2。")
        if self.direction not in (1, -1):
            raise ValueError(f"direction={self.direction} 必须为 +1 或 -1。")
        if self.standardize not in ("rank", "zscore", None):
            raise ValueError(
                f"standardize={self.standardize!r} 不合法，"
                "请选择 'rank'、'zscore' 或 None。"
            )
        if self.ic_method not in ("rank", "normal"):
            raise ValueError(
                f"ic_method={self.ic_method!r} 不合法，"
                "请选择 'rank' 或 'normal'。"
            )
        return self

    def __repr__(self) -> str:
        return (
            f"ResearchConfig("
            f"factor_name={self.factor_name!r}, "
            f"start={self.start!r}, end={self.end!r}, "
            f"forward={self.forward}, n_groups={self.n_groups}, "
            f"schema_version={self.schema_version!r})"
        )
