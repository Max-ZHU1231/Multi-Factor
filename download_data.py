"""
多因子选股模型 - AKShare 数据下载脚本
====================================
功能：
1. 更新每只股票的日频量价数据（换手率、振幅等，并入已有 CSV）
2. 下载每日估值指标（PE/PB/总市值/流通市值），通过百度股票接口
3. 下载基本面财务数据（综合财务指标 + 三大报表），保存到 fundamentals/

数据来源: AKShare (免费，无需 Token)
"""

import akshare as ak
import pandas as pd
import os
import time
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

# ─────────────────────── 配置 ───────────────────────
BASE_DIR       = r"d:\OneDrive - HKUST Connect\桌面\Multi Factor"
STOCKS_DIR     = os.path.join(BASE_DIR, "stocks", "stocks")
FUND_DIR       = os.path.join(BASE_DIR, "fundamentals")   # 新建：基本面数据
STOCK_LIST_CSV = os.path.join(BASE_DIR, "股票列表-stock_basic.csv")

# 下载历史区间
START_DATE = "20150101"
END_DATE   = datetime.today().strftime("%Y%m%d")

# 请求间隔（秒），避免触发反爬限速
SLEEP_BETWEEN = 0.4

# ─────────────────── 日志设置 ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "download.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

os.makedirs(FUND_DIR, exist_ok=True)

# ════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════

def load_stock_list() -> pd.DataFrame:
    """读取股票列表，返回所有在市股票"""
    df = pd.read_csv(STOCK_LIST_CSV, dtype=str)
    df = df[df["list_status"] == "L"].copy()
    log.info(f"在市股票共 {len(df)} 只")
    return df


def ts_to_ak_code(ts_code: str) -> str:
    """000001.SZ -> 000001"""
    return ts_code.split(".")[0]


def retry_call(func, *args, retries=3, wait=2.0, **kwargs):
    """带重试的函数调用，网络波动时自动重试"""
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise  # 用户中断不重试，直接透传
        except Exception as e:
            if i < retries - 1:
                log.debug(f"  retry {i+1}/{retries} for {func.__name__}: {e}")
                time.sleep(wait * (i + 1))
            else:
                raise  # 超过重试次数则向上抛出


def get_existing_latest_date(filepath: str):
    """读取已有 CSV 中最新的 trade_date，文件不存在返回 None"""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, dtype=str, usecols=["trade_date"])
        if df.empty:
            return None
        return df["trade_date"].max()
    except Exception:
        return None


def next_day_str(date_str: str) -> str:
    """YYYYMMDD -> 下一天 YYYYMMDD"""
    d = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)
    return d.strftime("%Y%m%d")


def append_to_csv(new_df: pd.DataFrame, path: str, key_cols: list):
    """将 new_df 追加到 path CSV，按 key_cols 去重保留最新"""
    if new_df is None or new_df.empty:
        return
    new_df = new_df.astype(str)
    if os.path.exists(path):
        old_df = pd.read_csv(path, dtype=str)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    # 只对存在的 key_cols 去重
    valid_keys = [c for c in key_cols if c in combined.columns]
    if valid_keys:
        combined = combined.drop_duplicates(subset=valid_keys, keep="last")
    combined.to_csv(path, index=False, encoding="utf-8-sig")


# ════════════════════════════════════════════════════
#  模块 1：日频量价数据（前复权，含换手率、振幅）
#  数据来源：ak.stock_zh_a_hist
# ════════════════════════════════════════════════════

PRICE_COL_MAP = {
    "日期":   "trade_date",
    "开盘":   "open",
    "最高":   "high",
    "最低":   "low",
    "收盘":   "close",
    "成交量": "vol",
    "成交额": "amount",
    "振幅":   "amplitude",     # 新增 ★
    "涨跌幅": "pct_chg",
    "涨跌额": "change",
    "换手率": "turnover_rate", # 新增 ★
}


def download_daily_price(symbol: str, start: str, end: str):
    """下载日线量价（前复权），含换手率"""
    try:
        df = retry_call(
            ak.stock_zh_a_hist,
            symbol=symbol, period="daily",
            start_date=start, end_date=end,
            adjust="qfq"
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns=PRICE_COL_MAP)
        df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
        keep = [c for c in PRICE_COL_MAP.values() if c in df.columns]
        return df[keep].copy()
    except Exception as e:
        log.debug(f"  price error {symbol}: {e}")
        return None  # 网络失败时返回 None，不阻断主流程


# ════════════════════════════════════════════════════
#  模块 2：每日估值指标（PE/PB/总市值/流通市值）
#  数据来源：ak.stock_zh_valuation_baidu
#  indicator 可选: 总市值 / 流通市值 / 市盈率(TTM) / 市净率 / 市销率(TTM) / 股息率(TTM)
# ════════════════════════════════════════════════════

VALUATION_INDICATORS = {
    "总市值":      "total_mv",
    "流通市值":    "circ_mv",
    "市盈率(TTM)": "pe_ttm",
    "市净率":      "pb",
    "市销率(TTM)": "ps_ttm",
    "股息率(TTM)": "dv_ttm",
}


def download_valuation(symbol: str):
    """下载个股历史估值指标，返回含 trade_date 列的 DataFrame"""
    frames = {}
    for indicator, col_name in VALUATION_INDICATORS.items():
        try:
            df = ak.stock_zh_valuation_baidu(symbol=symbol, indicator=indicator)
            if df is not None and not df.empty:
                df["date"] = df["date"].astype(str).str.replace("-", "")
                df = df.rename(columns={"date": "trade_date", "value": col_name})
                df = df.set_index("trade_date")
                frames[col_name] = df[col_name]
        except Exception as e:
            log.debug(f"  valuation {indicator} skip {symbol}: {e}")
        time.sleep(0.15)
    if not frames:
        return None
    return pd.DataFrame(frames).reset_index()


# ════════════════════════════════════════════════════
#  整合：更新单只股票的日频 CSV
# ════════════════════════════════════════════════════

def update_stock_daily(ts_code: str):
    """拉取增量量价 + 全量估值，合并写入个股 CSV"""
    symbol   = ts_to_ak_code(ts_code)
    exchange = ts_code.split(".")[1]
    filepath = os.path.join(STOCKS_DIR, f"{symbol}_{exchange}.csv")

    # 判断增量起始日
    latest = get_existing_latest_date(filepath)
    if latest and latest >= END_DATE:
        return  # 已是最新

    fetch_start = next_day_str(latest) if latest else START_DATE

    # ── 量价数据 ──
    price_df = download_daily_price(symbol, fetch_start, END_DATE)
    time.sleep(SLEEP_BETWEEN)

    # ── 估值指标（全历史，每次全量拉取，合并覆盖） ──
    try:
        val_df = download_valuation(symbol)
    except Exception as e:
        log.debug(f"  valuation overall error {symbol}: {e}")
        val_df = None
    time.sleep(SLEEP_BETWEEN)

    # ── 读取已有 CSV ──
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, dtype=str)
    else:
        existing = pd.DataFrame()

    # ── 追加新量价行 ──
    if price_df is not None and not price_df.empty:
        price_df["ts_code"] = ts_code
        if latest:
            price_df = price_df[price_df["trade_date"] > latest]
        if not price_df.empty:
            existing = pd.concat([existing, price_df.astype(str)], ignore_index=True)

    # ── 合并估值列（左连接） ──
    if val_df is not None and not val_df.empty and not existing.empty:
        val_cols = [c for c in val_df.columns if c != "trade_date"]
        existing = existing.drop(columns=[c for c in val_cols if c in existing.columns], errors="ignore")
        existing["trade_date"] = existing["trade_date"].astype(str)
        val_df["trade_date"] = val_df["trade_date"].astype(str)
        existing = existing.merge(val_df.astype(str), on="trade_date", how="left")

    # ── 去重、排序、保存 ──
    if not existing.empty:
        existing = existing.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        existing = existing.sort_values("trade_date").reset_index(drop=True)
        existing.to_csv(filepath, index=False, encoding="utf-8-sig")


# ════════════════════════════════════════════════════
#  模块 3：基本面财务数据
#  ① 综合财务指标（ROE/ROA/毛利率/净利率增速…）
#     来源：ak.stock_financial_analysis_indicator（东方财富）
#  ② 资产负债表（同花顺，按报告期）
#  ③ 利润表（同花顺，按报告期）
#  ④ 现金流量表（同花顺，按报告期）
# ════════════════════════════════════════════════════

def download_fin_analysis(ts_code: str):
    """综合财务指标：ROE、ROA、毛利率、营收增速、净利增速、资产负债率…"""
    symbol = ts_to_ak_code(ts_code)
    try:
        df = retry_call(ak.stock_financial_analysis_indicator, symbol=symbol, start_year="2010")
        if df is None or df.empty:
            return None
        df.insert(0, "ts_code", ts_code)
        return df
    except Exception as e:
        log.debug(f"  fin_analysis error {ts_code}: {e}")
        return None


def download_balance_sheet_ths(ts_code: str):
    """资产负债表（同花顺）：总资产、总负债、股东权益、货币资金…"""
    symbol = ts_to_ak_code(ts_code)
    try:
        df = retry_call(ak.stock_financial_debt_ths, symbol=symbol, indicator="按报告期")
        if df is None or df.empty:
            return None
        df.insert(0, "ts_code", ts_code)
        return df
    except Exception as e:
        log.debug(f"  balance_ths error {ts_code}: {e}")
        return None


def download_income_ths(ts_code: str):
    """利润表（同花顺）：营业总收入、净利润、归母净利润…"""
    symbol = ts_to_ak_code(ts_code)
    try:
        df = retry_call(ak.stock_financial_benefit_ths, symbol=symbol, indicator="按报告期")
        if df is None or df.empty:
            return None
        df.insert(0, "ts_code", ts_code)
        return df
    except Exception as e:
        log.debug(f"  income_ths error {ts_code}: {e}")
        return None


def download_cashflow_ths(ts_code: str):
    """现金流量表（同花顺）：经营/投资/融资净现金流…"""
    symbol = ts_to_ak_code(ts_code)
    try:
        df = retry_call(ak.stock_financial_cash_ths, symbol=symbol, indicator="按报告期")
        if df is None or df.empty:
            return None
        df.insert(0, "ts_code", ts_code)
        return df
    except Exception as e:
        log.debug(f"  cashflow_ths error {ts_code}: {e}")
        return None


# ════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════

def main():
    stock_df = load_stock_list()
    ts_codes = stock_df["ts_code"].tolist()

    log.info("=" * 60)
    log.info(f"开始下载，共 {len(ts_codes)} 只股票")
    log.info(f"日期范围: {START_DATE} → {END_DATE}")
    log.info("=" * 60)

    # 基本面输出路径
    fin_analysis_path = os.path.join(FUND_DIR, "financial_analysis.csv")
    balance_path      = os.path.join(FUND_DIR, "balance_sheet.csv")
    income_path       = os.path.join(FUND_DIR, "income_statement.csv")
    cashflow_path     = os.path.join(FUND_DIR, "cashflow.csv")

    errors_daily = []

    for ts_code in tqdm(ts_codes, desc="全市场下载进度"):

        # ── 模块1+2: 日频量价 + 估值指标 ──────────
        try:
            update_stock_daily(ts_code)
        except Exception as e:
            log.warning(f"[DAILY SKIP] {ts_code}: {e}")
            errors_daily.append(ts_code)

        # ── 模块3a: 综合财务指标 ────────────────────
        try:
            df = download_fin_analysis(ts_code)
            append_to_csv(df, fin_analysis_path, ["ts_code", "日期"])
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            log.debug(f"  fin_analysis append error {ts_code}: {e}")

        # ── 模块3b: 资产负债表 ──────────────────────
        try:
            df = download_balance_sheet_ths(ts_code)
            append_to_csv(df, balance_path, ["ts_code", "报告期"])
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            log.debug(f"  balance append error {ts_code}: {e}")

        # ── 模块3c: 利润表 ──────────────────────────
        try:
            df = download_income_ths(ts_code)
            append_to_csv(df, income_path, ["ts_code", "报告期"])
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            log.debug(f"  income append error {ts_code}: {e}")

        # ── 模块3d: 现金流量表 ──────────────────────
        try:
            df = download_cashflow_ths(ts_code)
            append_to_csv(df, cashflow_path, ["ts_code", "报告期"])
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            log.debug(f"  cashflow append error {ts_code}: {e}")

    # ── 汇总报告 ──────────────────────────────────
    log.info("=" * 60)
    log.info("下载完成！")
    log.info(f"  日频数据失败股票数: {len(errors_daily)}")
    if errors_daily:
        log.info(f"  失败列表（前20）: {errors_daily[:20]}")
    log.info(f"  基本面文件目录: {FUND_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
