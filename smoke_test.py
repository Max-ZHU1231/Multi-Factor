"""烟雾测试：验证 factor_analysis 全流程（3 因子，1 年数据）"""
import warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from pathlib import Path
import factor_analysis as fa
from factor_analysis import (
    CFG, _ensure_dirs, build_all_reports,
    plot_ic_timeseries, plot_cumulative_ic,
    build_ic_summary, plot_layer_backtest,
    plot_factor_correlation, plot_factor_clustering,
)

cfg = CFG.copy()
cfg["start"]      = "20240101"
cfg["end"]        = "20241231"
cfg["output_dir"] = Path("output/factor_analysis_smoke")

fa.ALL_FACTORS = ["momentum_12_1", "value_pb", "vol_20d"]

_ensure_dirs(cfg)
reports = build_all_reports(cfg)
print("reports:", list(reports.keys()))
assert len(reports) == 3, f"期望 3 个报告，实际 {len(reports)}"

plot_ic_timeseries(reports, cfg)
plot_cumulative_ic(reports, cfg)
ic_df = build_ic_summary(reports, cfg)
ls_df = plot_layer_backtest(reports, cfg)
corr  = plot_factor_correlation(reports, cfg)
plot_factor_clustering(corr, cfg)

print("\nSMOKE TEST PASSED ✓")
