# -*- coding: utf-8 -*-
"""
市值权重计算脚本（独立、可复现）
MTH202 CW1 课题 A

功能：
  1) 下载 18 只股票的未复权收盘价（真实交易价，用于算市值）
  2) 取对冲起始日(2025-01-01)前一交易日 2024-12-31 的收盘价
  3) 读历史总股本，取同一日期的 asof 值
  4) 市值 = 股本 × 未复权价；权重 = 市值 / 总市值
  5) 输出 market_cap.csv

依赖：shares_outstanding.csv
说明：市值用「未复权价」而非「复权价」，因为市值 = 真实交易价 × 真实股本。
"""

import warnings
warnings.filterwarnings("ignore")
import os

# ---- 代理设置（国内访问 Yahoo 需要 VPN）----
PROXY = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = PROXY
os.environ["HTTP_PROXY"] = PROXY
os.environ["ALL_PROXY"] = PROXY

import yfinance as yf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "数据集"

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "V", "MA",
    "XOM", "CVX",
    "PG", "KO", "WMT", "MCD",
    "JNJ", "UNH",
    "CAT", "HON",
]
VALUE_DATE = "2024-12-31"   # 对冲起始日 2025-01-01 的前一交易日（作为估值基准）

# ============ 1. 下载未复权收盘价 ============
print("[1/3] 下载未复权收盘价 ...")
px = yf.download(STOCKS, start="2024-12-01", end="2025-01-10",
                 auto_adjust=False, group_by="column", threads=False, progress=False)
close = px["Close"]   # 未复权真实收盘价

price = close.loc[VALUE_DATE] if VALUE_DATE in close.index else close.asof(VALUE_DATE)
print(f"  取 {VALUE_DATE} 收盘价, 有效 {price.notna().sum()}/18")

# ============ 2. 历史总股本（asof 同一日期） ============
print("[2/3] 读历史总股本 ...")
sh = pd.read_csv(DATA_DIR / "shares_outstanding.csv", index_col=0)
sh.index = pd.to_datetime(sh.index, utc=True).tz_localize(None)
shares = sh.apply(lambda c: c.dropna().asof(pd.Timestamp(VALUE_DATE)))

# ============ 3. 市值与权重 ============
print("[3/3] 计算市值权重 ...")
mcap = pd.DataFrame({"股本": shares, "未复权价": price})
mcap["市值(十亿美元)"] = (mcap["股本"] * mcap["未复权价"]) / 1e9
mcap["权重%"] = mcap["市值(十亿美元)"] / mcap["市值(十亿美元)"].sum() * 100
mcap = mcap.sort_values("市值(十亿美元)", ascending=False)
mcap.index.name = "Ticker"
mcap.to_csv(DATA_DIR / "market_cap.csv", encoding="utf-8-sig")

print("\n【市值权重】(未复权价 × 股本, 2024-12-31)")
print(mcap[["市值(十亿美元)", "权重%"]].round(2).to_string())
print(f"\n权重合计: {mcap['权重%'].sum():.4f}%")
print("\n市值权重计算完成。")
