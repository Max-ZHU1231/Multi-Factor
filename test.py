from factor_framework.pipeline import FactorPipeline

pipe = FactorPipeline("Stocks/", "股票列表-stock_basic.csv")
pipe.register_builtins(["momentum_12_1"])

# 全量运行（~5800只，并行 8 线程，约 1-2 分钟）
# 若只想快速验证，取消下面两行注释改用前 200 只股票（约 5 秒）：
# symbols = pipe.engine.all_symbols()[:200]
# report = pipe.run("momentum_12_1", start="20200101", end="20231231",
#                   forward=21, n_groups=5, standardize="rank", symbols=symbols)

report = pipe.run(
    "momentum_12_1",
    start="20200101", end="20231231",
    forward=21, n_groups=5, standardize="rank",
)
report.print_summary()
report.save("output/")
