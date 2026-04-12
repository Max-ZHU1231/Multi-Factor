"""
universes/builder.py
=====================
动态 Top-N 市值股票池快照生成器（防未来函数）。

核心设计原则
-----------
  防未来函数 (no look-ahead bias)
    - decision_date：做排名的日期（T 日收盘后数据可得）
    - effective_date = next_trading_day(decision_date, lag=lag_days)
    - 构建快照时 **只使用 decision_date 当天及之前的数据**
    - 禁止 lag=0 且不显式开启，否则抛 ValueError

  确定性 tie-break
    - 市值相同时按 symbol 字母序升序排列，保证结果稳定可复现

  可追溯
    - 每条快照含 decision_date / effective_date / rank / mktcap_value /
      universe_id / version

数据源要求
----------
股票 CSV 必须含以下列之一（按优先级）：
  1. 总市值（万元）    → total_mktcap
  2. 流通市值（万元）  → free_float_mktcap（近似）

  注：当前 CSV 列名为中文，本模块自动识别。

使用方式
--------
    from universes.builder import DynamicUniverseBuilder

    builder = DynamicUniverseBuilder(
        stocks_dir   = "stocks/",
        trade_cal    = "交易日历-trade_cal.csv",
        top_n        = 500,
        metric       = "total_mktcap",            # or "free_float_mktcap"
        rebalance_freq     = "semiannual",         # or "annual" / "quarterly"
        rebalance_months   = [6, 12],
        effective_lag_days = 1,
    )
    snapshots = builder.build(start="20200101", end="20251231")
    # snapshots: pd.DataFrame with columns:
    #   decision_date, effective_date, symbol, rank, mktcap_value,
    #   universe_id, version

    # 保存到缓存
    builder.save(snapshots, "cache/universe/top500_total_mktcap_sa.parquet")
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ── 版本号（每次更改快照生成逻辑时递增）────────────────────────────────────
SNAPSHOT_VERSION: str = "1.0"

# ── 列名映射（CSV 中文列名 → 内部英文名）────────────────────────────────────
_COL_MAP: dict[str, str] = {
    "总市值（万元）":    "total_mktcap",
    "流通市值（万元）":  "free_float_mktcap",
    "交易日":           "trade_date",
}

# ── metric 合法值 ────────────────────────────────────────────────────────────
_VALID_METRICS = {"total_mktcap", "free_float_mktcap"}

# ── rebalance_freq 定义 ──────────────────────────────────────────────────────
_FREQ_MONTHS: dict[str, List[int]] = {
    "annual":     [12],
    "semiannual": [6, 12],
    "quarterly":  [3, 6, 9, 12],
}


# ═════════════════════════════════════════════════════════════════════════════
#  交易日历工具
# ═════════════════════════════════════════════════════════════════════════════

def _load_trade_cal(trade_cal: str | Path) -> List[str]:
    """
    读取交易日历，返回升序排列的交易日列表（格式 'YYYYMMDD'）。

    CSV 格式：含 'cal_date' 或 '日期' 列，以及 'is_open' 或 '是否交易日' 列。
    若无 is_open 列，则返回全部日期。
    """
    df = pd.read_csv(trade_cal, dtype=str)

    # 找日期列
    date_col = None
    for c in df.columns:
        if "cal_date" in c or c.strip() in ("cal_date", "日期", "trade_date", "date"):
            date_col = c
            break
    if date_col is None:
        # 尝试第一列
        date_col = df.columns[0]

    # 找 is_open 列
    open_col = None
    for c in df.columns:
        if "is_open" in c or c.strip() in ("is_open", "是否交易日"):
            open_col = c
            break

    dates = df[date_col].str.strip()
    if open_col is not None:
        dates = dates[df[open_col].astype(str).str.strip() == "1"]

    cal = sorted(dates.dropna().tolist())
    return cal


def _next_trading_day(cal: List[str], date: str, lag: int = 1) -> str:
    """
    返回 date 之后第 lag 个交易日。

    Parameters
    ----------
    cal  : 有序交易日列表
    date : 'YYYYMMDD' 字符串（decision_date）
    lag  : 往后数几个交易日（默认 1）

    Returns
    -------
    effective_date : str 'YYYYMMDD'

    Raises
    ------
    ValueError : date 不在日历范围内，或后移后超出末尾
    """
    # 找到 date 在日历中的位置（不要求精确命中，找最近的交易日）
    import bisect
    idx = bisect.bisect_right(cal, date)
    # bisect_right 给出第一个 > date 的位置
    # 我们需要从 date 当日（若是交易日）或 date 之后的第一个交易日起数 lag 步
    # 为简单起见：decision_date 本身可能不在日历（节假日），找其后第一个交易日
    # 然后再往后 lag-1 步
    if idx >= len(cal):
        raise ValueError(f"decision_date={date!r} is out of trading-calendar range.")
    # idx 已是 date 之后第一个交易日，再加 lag-1
    eff_idx = idx + (lag - 1)
    if eff_idx >= len(cal):
        raise ValueError(
            f"effective_date exceeds trading-calendar end (decision_date={date!r}, lag={lag})."
        )
    return cal[eff_idx]


def _last_trading_day_of_month(cal: List[str], year: int, month: int) -> Optional[str]:
    """返回指定年月的最后一个交易日（'YYYYMMDD'），不存在则返回 None。"""
    prefix = f"{year:04d}{month:02d}"
    days = [d for d in cal if d.startswith(prefix)]
    return days[-1] if days else None


def _generate_decision_dates(
    cal:       List[str],
    start:     str,
    end:       str,
    months:    List[int],
) -> List[str]:
    """
    生成调仓决策日列表：每年 months 中各月的最后一个交易日，
    限定在 [start, end] 范围内。

    Parameters
    ----------
    cal    : 交易日列表
    start  : 开始日期 'YYYYMMDD'
    end    : 结束日期 'YYYYMMDD'
    months : 调仓月份列表（如 [6, 12]）

    Returns
    -------
    list[str] : 升序排列的 decision_date 列表
    """
    start_year = int(start[:4])
    end_year   = int(end[:4])

    dates: List[str] = []
    for year in range(start_year - 1, end_year + 1):  # 多查一年，保证第一期有基准
        for month in sorted(months):
            d = _last_trading_day_of_month(cal, year, month)
            if d is not None and start <= d <= end:
                dates.append(d)

    return sorted(set(dates))


# ═════════════════════════════════════════════════════════════════════════════
#  快照构建器
# ═════════════════════════════════════════════════════════════════════════════

class DynamicUniverseBuilder:
    """
    动态 Top-N 市值股票池快照生成器。

    Parameters
    ----------
    stocks_dir         : 股票 CSV 数据目录（每只股票一个 CSV）
    trade_cal          : 交易日历 CSV 路径
    top_n              : 入选股票数量（默认 500）
    metric             : 市值字段（'total_mktcap' 或 'free_float_mktcap'）
    rebalance_freq     : 调仓频率（'semiannual' / 'annual' / 'quarterly'）
    rebalance_months   : 调仓月份列表（覆盖 rebalance_freq 时使用）；
                         None = 根据 rebalance_freq 自动确定
    effective_lag_days : 生效延迟交易日数（默认 1；0 需显式 allow_lag_zero=True）
    allow_lag_zero     : 是否允许 lag=0（T 日决策 T 日生效）；默认 False
    """

    def __init__(
        self,
        stocks_dir:          str | Path = "stocks/",
        trade_cal:           str | Path = "交易日历-trade_cal.csv",
        top_n:               int = 500,
        metric:              str = "total_mktcap",
        rebalance_freq:      str = "semiannual",
        rebalance_months:    Optional[List[int]] = None,
        effective_lag_days:  int = 1,
        allow_lag_zero:      bool = False,
    ) -> None:
        if metric not in _VALID_METRICS:
            raise ValueError(
                f"metric={metric!r} is invalid, allowed: {sorted(_VALID_METRICS)}"
            )
        if effective_lag_days == 0 and not allow_lag_zero:
            raise ValueError(
                "effective_lag_days=0 means same-day decision/effective date and may introduce look-ahead bias.\n"
                "If intentional, set allow_lag_zero=True."
            )
        if rebalance_freq not in _FREQ_MONTHS and rebalance_months is None:
            raise ValueError(
                f"rebalance_freq={rebalance_freq!r} is invalid, "
                f"allowed: {list(_FREQ_MONTHS.keys())}, "
                f"or specify rebalance_months explicitly."
            )

        self.stocks_dir        = Path(stocks_dir)
        self.trade_cal_path    = Path(trade_cal)
        self.top_n             = top_n
        self.metric            = metric
        self.rebalance_freq    = rebalance_freq
        self.rebalance_months  = rebalance_months or _FREQ_MONTHS.get(rebalance_freq, [6, 12])
        self.effective_lag_days = effective_lag_days
        self.allow_lag_zero    = allow_lag_zero

        # 延迟加载
        self._cal: Optional[List[str]] = None

    # ── 属性 ─────────────────────────────────────────────────────────────────

    @property
    def universe_id(self) -> str:
        """唯一标识此股票池配置的字符串（用于缓存文件名）。"""
        freq_tag = self.rebalance_freq[:2]  # sa / an / qu
        return f"top{self.top_n}_{self.metric[:5]}_{freq_tag}"

    @property
    def config_dict(self) -> dict:
        """返回可序列化的配置字典（用于计算 config_hash）。"""
        return {
            "top_n":               self.top_n,
            "metric":              self.metric,
            "rebalance_freq":      self.rebalance_freq,
            "rebalance_months":    sorted(self.rebalance_months),
            "effective_lag_days":  self.effective_lag_days,
            "snapshot_version":    SNAPSHOT_VERSION,
        }

    @property
    def config_hash(self) -> str:
        """MD5(config_dict) 前 12 位，用于缓存键区分不同参数配置。"""
        raw = json.dumps(self.config_dict, sort_keys=True).encode()
        return hashlib.md5(raw).hexdigest()[:12]

    # ── 交易日历 ──────────────────────────────────────────────────────────────

    def _get_cal(self) -> List[str]:
        if self._cal is None:
            self._cal = _load_trade_cal(self.trade_cal_path)
        return self._cal

    # ── 数据加载 ──────────────────────────────────────────────────────────────

    def _load_mktcap_on_date(self, decision_date: str) -> pd.Series:
        """
        读取所有股票在 decision_date 当天的市值，返回 Series(index=symbol)。

        防未来函数保证：仅读取 decision_date 当天的数据行。
        若某只股票 decision_date 停牌（无数据），该股票被排除。

        Parameters
        ----------
        decision_date : 'YYYYMMDD' 字符串

        Returns
        -------
        pd.Series : index = ts_code，value = 市值（万元，float）
        """
        col_zh = "总市值（万元）" if self.metric == "total_mktcap" else "流通市值（万元）"

        records: dict[str, float] = {}

        for csv_path in sorted(self.stocks_dir.glob("*.csv")):
            symbol = csv_path.stem.replace(".", "_")  # 000001.SZ → 000001_SZ → ts_code 格式
            # 规范化 ts_code：000001_SZ → 000001.SZ
            symbol_dot = symbol.replace("_", ".")

            try:
                # 只读取需要的列以节省内存
                df = pd.read_csv(
                    csv_path,
                    dtype={"交易日": str},
                    usecols=lambda c: c in ("交易日", col_zh),
                )
                if "交易日" not in df.columns or col_zh not in df.columns:
                    continue

                # 只取 decision_date 当天的行
                row = df[df["交易日"].str.strip() == decision_date]
                if row.empty:
                    continue  # 该日停牌/无数据，排除

                val = row[col_zh].iloc[0]
                if pd.isna(val) or float(val) <= 0:
                    continue

                records[symbol_dot] = float(val)

            except Exception:
                continue

        return pd.Series(records, name="mktcap", dtype=float)

    # ── 核心方法 ──────────────────────────────────────────────────────────────

    def build_snapshot(self, decision_date: str) -> pd.DataFrame:
        """
        构建单个调仓点的快照表。

        Parameters
        ----------
        decision_date : 'YYYYMMDD'，做排名的日期

        Returns
        -------
        pd.DataFrame，列：
            decision_date, effective_date, symbol, rank,
            mktcap_value, universe_id, version
        """
        cal = self._get_cal()
        if self.effective_lag_days == 0:
            effective_date = decision_date
        else:
            effective_date = _next_trading_day(cal, decision_date, lag=self.effective_lag_days)

        mktcap = self._load_mktcap_on_date(decision_date)
        if mktcap.empty:
            warnings.warn(
                f"[DynamicUniverseBuilder] decision_date={decision_date!r} "
                f"has no valid market-cap data; snapshot is empty.",
                RuntimeWarning,
            )
            return pd.DataFrame(
                columns=["decision_date", "effective_date", "symbol",
                         "rank", "mktcap_value", "universe_id", "version"]
            )

        # 降序排列，市值相同时按 symbol 升序（确定性 tie-break）
        mktcap_sorted = (
            mktcap
            .reset_index()
            .rename(columns={"index": "symbol", "mktcap": "mktcap_value"})
            .sort_values(["mktcap_value", "symbol"], ascending=[False, True])
            .reset_index(drop=True)
        )

        # 截断到 top_n
        top = mktcap_sorted.head(self.top_n).copy()
        top["rank"]          = range(1, len(top) + 1)
        top["decision_date"] = decision_date
        top["effective_date"] = effective_date
        top["universe_id"]   = self.universe_id
        top["version"]       = SNAPSHOT_VERSION

        return top[["decision_date", "effective_date", "symbol",
                    "rank", "mktcap_value", "universe_id", "version"]]

    def build(
        self,
        start: str,
        end:   str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        批量构建整个回测区间内所有调仓点的快照，拼接为一个 DataFrame。

        Parameters
        ----------
        start   : 回测开始日期 'YYYYMMDD'（只构建 >= start 的 decision_date）
        end     : 回测结束日期 'YYYYMMDD'
        verbose : 是否打印进度

        Returns
        -------
        pd.DataFrame，按 (decision_date, rank) 升序排列
        """
        cal = self._get_cal()

        # 生成调仓决策日：从 start 前一期开始，确保第一期有有效成分
        # 实际只保留 effective_date <= end 的快照
        decision_dates = _generate_decision_dates(
            cal, start=start, end=end, months=self.rebalance_months
        )

        # 若无任何调仓点，使用 start 前最近的交易日作为唯一调仓点
        if not decision_dates:
            # 找 start 之前最近的一个交易日
            import bisect
            idx = bisect.bisect_left(cal, start)
            if idx > 0:
                decision_dates = [cal[idx - 1]]
            else:
                decision_dates = [start]
            warnings.warn(
                f"[DynamicUniverseBuilder] No rebalance date found in [{start}, {end}], "
                f"using {decision_dates[0]!r} as the only snapshot.",
                RuntimeWarning,
            )

        if verbose:
            print(f"[DynamicUniverseBuilder] Building {len(decision_dates)} snapshots "
                  f"(top={self.top_n}, metric={self.metric})")

        frames: list[pd.DataFrame] = []
        for i, dd in enumerate(decision_dates):
            if verbose:
                print(f"  [{i+1}/{len(decision_dates)}] decision_date={dd} ...", end=" ")
            snap = self.build_snapshot(dd)
            if verbose:
                print(f"{len(snap)} symbols")
            frames.append(snap)

        if not frames:
            return pd.DataFrame(
                columns=["decision_date", "effective_date", "symbol",
                         "rank", "mktcap_value", "universe_id", "version"]
            )

        result = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["decision_date", "rank"])
            .reset_index(drop=True)
        )
        return result

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(
        self,
        snapshots: pd.DataFrame,
        path: str | Path,
    ) -> Path:
        """
        将快照表保存为 Parquet 文件。

        Parameters
        ----------
        snapshots : build() 返回的 DataFrame
        path      : 目标文件路径

        Returns
        -------
        Path : 实际写入路径
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        snapshots.to_parquet(out, index=False)
        return out

    @staticmethod
    def load(path: str | Path) -> pd.DataFrame:
        """从 Parquet 文件加载快照表。"""
        return pd.read_parquet(Path(path))
