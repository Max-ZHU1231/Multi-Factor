#!/usr/bin/env python
"""
scripts/run_analysis.py
=======================
全量因子分析入口脚本（factor_analysis.py 的迁移版）。

用法
----
cd "d:\\OneDrive - HKUST Connect\\桌面\\Multi Factor"
.venv\\Scripts\\python.exe scripts/run_analysis.py

等价于在项目根目录运行 factor_analysis.py，但遵循 v3.0 目录规范。
所有分析逻辑保留在 factor_analysis.py（向后兼容），此脚本仅作入口转发。
"""
import sys
from pathlib import Path

# 确保项目根目录在路径中
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    """Entry-point for `run-analysis` console script."""
    # 运行根目录版本（保持向后兼容）
    exec(open(ROOT / "factor_analysis.py", encoding="utf-8").read(), {"__file__": str(ROOT / "factor_analysis.py")})


if __name__ == "__main__":
    main()
