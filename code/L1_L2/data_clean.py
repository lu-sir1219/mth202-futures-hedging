# -*- coding: utf-8 -*-
"""
L1 — 数据清洗与收益率计算脚本
MTH202 CW1 课题 A：股票组合对冲

功能：
  1) 读入 raw_data.csv，去掉美股休市日（如独立日、感恩节等），处理缺失值
  2) 计算日对数收益率（18 只股票 / SPY / E-mini 期货）
  3) 换算无风险日利率（IRX 年化% -> 日度小数），计算超额收益
  4) 按对冲起始日 2025-01-01 切分估计期 / 对冲期
  5) 输出 clean_data.csv、returns_data.csv、descriptive_stats.csv

依赖：../数据集/raw_data.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# ---- 路径与参数 ----
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "数据集"

HEDGE_INCEPTION = "2025-01-01"   # 对冲起始日
EST_START, EST_END = "2023-01-01", "2024-12-31"    # 估计期（24 个月，两个完整年）
EVAL_START, EVAL_END = "2025-01-01", "2025-12-31"  # 对冲期（12 个月，完整年）

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "V", "MA",
    "XOM", "CVX",
    "PG", "KO", "WMT", "MCD",
    "JNJ", "UNH",
    "CAT", "HON",
]

# ============ 1. 读入 ============
raw = pd.read_csv(DATA_DIR / "raw_data.csv", index_col=0, parse_dates=True)

# ============ 2. 去休市日 ============
# 美股休市日表现为 SPY 为 NaN（如独立日、感恩节等），直接删掉该行
n_before = len(raw)
clean = raw.dropna(subset=["SPY"]).copy()
print(f"[清洗] 去掉休市日: {n_before} -> {len(clean)} 行 (删 {n_before - len(clean)} 天)")

# ============ 3. 剩余缺失检查 + 前向填充 ============
missing = clean.isna().sum()
if missing.sum() > 0:
    print("[清洗] 剩余缺失(填充前):")
    print(missing[missing > 0])
    clean = clean.ffill()

# ============ 4. 日对数收益率 ============
returns = pd.DataFrame(index=clean.index)
for t in STOCKS:
    returns[t] = np.log(clean[t] / clean[t].shift(1))
returns["SPY"] = np.log(clean["SPY"] / clean["SPY"].shift(1))
returns["ES_F"]  = np.log(clean["ES_F"] / clean["ES_F"].shift(1))

# ============ 5. 无风险日利率 + 超额收益 ============
rf_daily = clean["IRX"] / 100.0 / 252.0          # 年化% -> 日度小数
returns["RF_daily"] = rf_daily
for t in STOCKS:
    returns[t + "_excess"] = returns[t] - rf_daily   # 个股超额收益
returns["SPY_excess"] = returns["SPY"] - rf_daily  # 市场超额收益

returns = returns.dropna(subset=["SPY"])        # 去掉首行(无前一日)
returns.index.name = "Date"

# ============ 6. 保存 ============
clean.to_csv(DATA_DIR / "clean_data.csv")
returns.to_csv(DATA_DIR / "returns_data.csv")
print(f"[保存] clean_data.csv {clean.shape} | returns_data.csv {returns.shape}")

# ============ 7. 切分估计期 / 对冲期 ============
est = returns.loc[EST_START:EST_END]
evl = returns.loc[EVAL_START:EVAL_END]
print(f"[切分] 估计期 {est.index.min().date()} ~ {est.index.max().date()} ({len(est)} 天)")
print(f"[切分] 对冲期 {evl.index.min().date()} ~ {evl.index.max().date()} ({len(evl)} 天)")

# ============ 8. 描述性统计（全样本日收益率） ============
stat_cols = STOCKS + ["SPY", "ES_F"]
stats = pd.DataFrame(index=stat_cols)
stats["均值(日)"]     = returns[stat_cols].mean()
stats["标准差(日)"]   = returns[stat_cols].std()
stats["偏度"]         = returns[stat_cols].skew()
stats["峰度"]         = returns[stat_cols].kurtosis()
stats["最小值"]       = returns[stat_cols].min()
stats["最大值"]       = returns[stat_cols].max()
stats["年化均值%"]    = returns[stat_cols].mean() * 252 * 100
stats["年化波动率%"]  = returns[stat_cols].std() * np.sqrt(252) * 100
stats = stats.round(6)
stats.to_csv(DATA_DIR / "descriptive_stats.csv")

print("\n[描述性统计] 日收益率 (关键列):")
print(stats[["均值(日)", "标准差(日)", "年化均值%", "年化波动率%"]].round(4).to_string())

print("\nL1 完成。")
