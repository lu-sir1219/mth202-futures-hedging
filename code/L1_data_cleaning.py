# -*- coding: utf-8 -*-
"""L1 - data cleaning.

读取原始价格，去休市日、算日对数收益率和超额收益，输出清洗后价格、收益率、
描述性统计三张表到 ../Source_Data_034。
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"

EST_START, EST_END = "2023-01-01", "2024-12-31"    # 估计期（估计 beta 与对冲比）
EVAL_START, EVAL_END = "2025-01-01", "2025-12-31"  # 对冲期（样本外回测）

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "V", "MA",
    "XOM", "CVX",
    "PG", "KO", "WMT", "MCD",
    "JNJ", "UNH",
    "CAT", "HON",
]


def main():
    raw = pd.read_csv(DATA / "L0-data collection-raw daily prices.csv",
                      index_col=0, parse_dates=True)
    print(f"[read] raw daily prices: {raw.shape[0]} rows x {raw.shape[1]} cols")

    # 美股休市日表现为 SPY 缺失，直接删掉该行
    clean = raw.dropna(subset=["SPY"]).copy()
    missing = clean.isna().sum()
    if missing.sum() > 0:
        clean = clean.ffill()
    clean.to_csv(DATA / "L1-data cleaning-cleaned daily prices.csv")
    print(f"[saved] L1-data cleaning-cleaned daily prices.csv  {clean.shape[0]} rows")

    returns = pd.DataFrame(index=clean.index)
    for t in STOCKS:
        # 日对数收益率 r_t = ln(P_t / P_{t-1})
        returns[t] = np.log(clean[t] / clean[t].shift(1))
    returns["SPY"] = np.log(clean["SPY"] / clean["SPY"].shift(1))
    returns["ES_F"] = np.log(clean["ES_F"] / clean["ES_F"].shift(1))

    # 无风险日利率：年化 % 换成日度小数（一年按 252 个交易日）
    rf_daily = clean["IRX"] / 100.0 / 252.0
    returns["RF_daily"] = rf_daily
    for t in STOCKS:
        returns[t + "_excess"] = returns[t] - rf_daily
    returns["SPY_excess"] = returns["SPY"] - rf_daily

    returns = returns.dropna(subset=["SPY"])   # 去掉首行（无前一日）
    returns.index.name = "Date"
    returns.to_csv(DATA / "L1-data cleaning-daily returns.csv")
    print(f"[saved] L1-data cleaning-daily returns.csv  {returns.shape[0]} rows")

    # 描述性统计（全样本日收益率）
    stat_cols = STOCKS + ["SPY", "ES_F"]
    stats = pd.DataFrame(index=stat_cols)
    stats["mean_daily"] = returns[stat_cols].mean()
    stats["sd_daily"] = returns[stat_cols].std()
    stats["skewness"] = returns[stat_cols].skew()
    stats["kurtosis"] = returns[stat_cols].kurtosis()
    stats["minimum"] = returns[stat_cols].min()
    stats["maximum"] = returns[stat_cols].max()
    stats["annualized_mean_pct"] = returns[stat_cols].mean() * 252 * 100
    # 年化波动率 = 日度标准差 x sqrt(252)
    stats["annualized_sd_pct"] = returns[stat_cols].std() * np.sqrt(252) * 100
    stats = stats.round(6)
    stats.to_csv(DATA / "L1-data cleaning-descriptive statistics.csv", index_label="Ticker")
    print("[saved] L1-data cleaning-descriptive statistics.csv")
    print(stats.round(4).to_string())


if __name__ == "__main__":
    main()
