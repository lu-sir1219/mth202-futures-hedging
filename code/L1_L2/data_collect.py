# -*- coding: utf-8 -*-
"""
L0 — 数据采集脚本
MTH202 CW1 课题 A：股票组合对冲（Equity Portfolio Hedging）

功能：
  1) 下载 18 只美股、SPY（标普500 ETF 市场代理）、E-mini S&P500 期货、13 周国库券利率的日度数据
  2) 下载 18 只股票的历史总股本（季度披露，用于市值加权组合 beta）

输出（写入 ../数据集/）：
  raw_data.csv            日度价格（股票复权收盘 + SPY + 期货 + 利率）
  shares_outstanding.csv  18 只股票历史总股本

时间区间：2022-12-01 ~ 2026-01-31（估计期 2023-2024，对冲期 2025，对冲起始日 2025-01-01）
"""

import warnings
warnings.filterwarnings("ignore")
import os

# ---- 代理设置（国内访问 Yahoo Finance 需要 VPN；把端口改成你的客户端端口）----
# 你的 VPN 是 Clash，混合端口 7897；其他客户端请改成对应端口（如 7890 / 1080）
PROXY = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = PROXY
os.environ["HTTP_PROXY"] = PROXY
os.environ["ALL_PROXY"] = PROXY

import yfinance as yf
import pandas as pd
from pathlib import Path

# ---- 路径 ----
BASE_DIR = Path(__file__).resolve().parent.parent   # D:\MTH202
DATA_DIR = BASE_DIR / "数据集"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- 参数 ----
START = "2022-12-01"
END   = "2026-01-31"

# 18 只跨行业美股（等权持有的候选组合）
STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",   # 科技 (5)
    "JPM", "V", "MA",                          # 金融 (3)
    "XOM", "CVX",                              # 能源 (2)
    "PG", "KO", "WMT", "MCD",                  # 消费 (4)
    "JNJ", "UNH",                              # 医疗 (2)
    "CAT", "HON",                              # 工业 (2)
]
MARKET  = "SPY"     # SPY（标普500 ETF，CAPM 市场代理，用复权价=总回报）
FUTURES = "ES=F"    # E-mini S&P500 期货（对冲工具，乘数 $50）
RF      = "^IRX"    # 13 周国库券利率（无风险利率，年化 %）

ALL_TICKERS = STOCKS + [MARKET, FUTURES, RF]

# ================= 1. 下载日度价格 =================
print(f"[1/2] 下载价格数据 {START} ~ {END} ...")
prices = yf.download(
    ALL_TICKERS,
    start=START,
    end=END,
    auto_adjust=False,   # 保留 Close 与 Adj Close 两个字段
    group_by="column",   # 列层级 = (Price, Ticker)，便于 prices["Adj Close"] 取全部
    threads=False,       # 顺序下载，避免本地缓存并发锁冲突
    progress=False,
)

adj = prices["Adj Close"]   # 各 ticker 的复权收盘价
cls = prices["Close"]       # 各 ticker 的原始收盘价

raw = pd.DataFrame(index=adj.index)
for t in STOCKS:
    raw[t] = adj[t]              # 股票 -> 复权收盘价（已含分红/拆股调整）
raw["SPY"] = adj[MARKET]          # SPY -> 复权收盘价（总回报市场代理）
raw["ES_F"]  = cls[FUTURES]      # 期货 -> 收盘价（期货不复权）
raw["IRX"]   = cls[RF]           # 无风险利率 -> 收盘价（年化 %）
raw.index.name = "Date"
raw = raw.dropna(how="all")      # 去掉全空行

out1 = DATA_DIR / "raw_data.csv"
raw.to_csv(out1)
print(f"  已保存 {out1}  (shape={raw.shape})")

# ================= 2. 下载历史总股本 =================
print("[2/2] 下载 18 只股票的历史总股本 ...")
shares = {}
for i, t in enumerate(STOCKS, 1):
    try:
        s = yf.Ticker(t).get_shares_full(start=START, end=END)
        if s is not None and len(s) > 0:
            # 去重（同一日期可能重复出现），保留最后一个值，并按日期排序
            s = s[~s.index.duplicated(keep="last")].sort_index()
            shares[t] = s
            print(f"  [{i:2d}/18] {t}: {len(s)} 个数据点, 最新 {s.iloc[-1]:,.0f} 股")
        else:
            print(f"  [{i:2d}/18] {t}: 无数据")
    except Exception as e:
        print(f"  [{i:2d}/18] {t}: 失败 ({e})")

shares_df = pd.concat(shares, axis=1)   # 按日期并集合并，列 = 各股票
shares_df = shares_df.sort_index()
shares_df.index.name = "Date"
out2 = DATA_DIR / "shares_outstanding.csv"
shares_df.to_csv(out2)
print(f"  已保存 {out2}  (shape={shares_df.shape})")

print("\nL0 完成。")
