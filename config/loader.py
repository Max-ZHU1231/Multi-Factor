"""
config/loader.py
================
配置加载器：将 YAML 配置与 CLI 参数合并，遵循三级优先级：

  CLI 参数 > 用户配置文件 > config/default.yaml

用法
----
from config.loader import load_config

# 不传参数 → 纯 default.yaml
cfg = load_config()

# 加载用户配置（覆盖 default）
cfg = load_config(user_config="my_config.yaml")

# 加载用户配置 + CLI 覆盖
cfg = load_config(user_config="my_config.yaml", overrides={"backtest.forward": 10})

print(cfg.data.stocks_dir)
print(cfg.backtest.forward)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# 默认配置文件路径
_DEFAULT_YAML = Path(__file__).parent / "default.yaml"


class ConfigNamespace:
    """
    将嵌套 dict 包装为可点号访问的命名空间对象。

    示例
    ----
    cfg = ConfigNamespace({"data": {"stocks_dir": "Stocks/"}})
    cfg.data.stocks_dir  # "Stocks/"
    cfg["data.stocks_dir"]  # "Stocks/"
    """

    def __init__(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                object.__setattr__(self, k, ConfigNamespace(v))
            else:
                object.__setattr__(self, k, v)
        object.__setattr__(self, "_raw", d)

    def __getitem__(self, dotted_key: str) -> Any:
        """支持 cfg['backtest.forward'] 点号访问。"""
        keys = dotted_key.split(".")
        obj = self
        for k in keys:
            obj = getattr(obj, k)
        return obj

    def __repr__(self) -> str:
        return f"ConfigNamespace({self._raw!r})"

    def to_dict(self) -> Dict[str, Any]:
        """递归转回普通 dict。"""
        result = {}
        for k, v in self._raw.items():
            if isinstance(v, dict):
                result[k] = ConfigNamespace(v).to_dict()
            else:
                result[k] = v
        return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    """加载单个 YAML 文件，返回 dict；文件不存在或 PyYAML 缺失时返回空 dict。"""
    if not _HAS_YAML:
        # 降级：返回空 dict（调用方使用 default 硬编码值）
        return {}
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归合并两个嵌套 dict，override 中的值覆盖 base 中的同名值。"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _apply_dotted_overrides(d: Dict, overrides: Dict[str, Any]) -> Dict:
    """
    将 {'backtest.forward': 10} 形式的 CLI 覆盖写入嵌套 dict。

    支持：
    - 'backtest.forward' → d['backtest']['forward'] = 10
    - 'data.stocks_dir'  → d['data']['stocks_dir'] = ...
    """
    result = copy.deepcopy(d)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        obj = result
        for k in keys[:-1]:
            obj = obj.setdefault(k, {})
        obj[keys[-1]] = value
    return result


def load_config(
    user_config: Optional[str | Path] = None,
    overrides:   Optional[Dict[str, Any]] = None,
) -> ConfigNamespace:
    """
    加载并合并配置，返回 ConfigNamespace。

    优先级（高 → 低）：
      overrides（CLI 参数）> user_config > default.yaml

    Parameters
    ----------
    user_config : 用户 YAML 配置文件路径（可选）
    overrides   : 点号键-值对（可选），例如 {"backtest.forward": 10}

    Returns
    -------
    ConfigNamespace — 可点号访问的配置对象
    """
    # 1. 加载 default
    merged = _load_yaml(_DEFAULT_YAML)

    # 2. 叠加 user_config
    if user_config is not None:
        user_path = Path(user_config)
        user_data = _load_yaml(user_path)
        merged = _deep_merge(merged, user_data)

    # 3. 叠加 CLI overrides
    if overrides:
        merged = _apply_dotted_overrides(merged, overrides)

    return ConfigNamespace(merged)


def print_config(cfg: ConfigNamespace, indent: int = 0) -> None:
    """递归打印配置内容（用于启动时参数回显）。"""
    for k, v in cfg._raw.items():
        prefix = "  " * indent
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_config(ConfigNamespace(v), indent + 1)
        else:
            print(f"{prefix}{k}: {v!r}")
