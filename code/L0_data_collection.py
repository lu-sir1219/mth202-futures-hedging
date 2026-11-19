# -*- coding: utf-8 -*-
"""L0 - data collection (Yahoo Finance).

下载 18 只股票 + SPY + ES 期货 + 13 周国库券利率的日度数据，以及历史总股本，
写出两张 CSV 到 ../Source_Data_034。这是唯一需要联网的脚本（结果已随附）。
"""

import os
import socket
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"
DATA.mkdir(parents=True, exist_ok=True)

# 数据起止日期：2022-12-01 ~ 2026-01-31
START = "2022-12-01"
END = "2026-01-31"

# 18 只跨行业美股
STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",   # 科技
    "JPM", "V", "MA",                          # 金融
    "XOM", "CVX",                              # 能源
    "PG", "KO", "WMT", "MCD",                  # 消费
    "JNJ", "UNH",                              # 医疗
    "CAT", "HON",                              # 工业
]
MARKET = "SPY"      # CAPM 市场代理，用复权价（含总回报）
FUTURES = "ES=F"    # E-mini S&P500 期货，乘数 $50
RF = "^IRX"         # 13 周国库券利率（年化 %）


def configure_proxy():
    """仅当本机 7897 端口有代理在监听时才启用代理，否则直连。"""
    host, port = "127.0.0.1", 7897
    try:
        s = socket.create_connection((host, port), timeout=1.5)
        s.close()
        os.environ["HTTPS_PROXY"] = f"http://{host}:{port}"
        os.environ["HTTP_PROXY"] = f"http://{host}:{port}"
        os.environ["ALL_PROXY"] = f"http://{host}:{port}"
        print(f"[proxy] using {host}:{port}")
    except OSError:
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
            os.environ.pop(k, None)
        print("[proxy] none detected, downloading directly")


def main():
    configure_proxy()
    print("Downloading daily prices ...")
    prices = yf.download(STOCKS + [MARKET, FUTURES, RF], start=START, end=END,
                         auto_adjust=False, group_by="column", threads=False, progress=False)
    adj = prices["Adj Close"]
    cls = prices["Close"]

    raw = pd.DataFrame(index=adj.index)
    for t in STOCKS:
        raw[t] = adj[t]          # 股票用复权收盘价（含分红/拆股调整）
    raw["SPY"] = adj[MARKET]
    raw["ES_F"] = cls[FUTURES]   # 期货不复权
    raw["IRX"] = cls[RF]
    raw.index.name = "Date"
    raw = raw.dropna(how="all")
    # 保存 18 只股票 + SPY + 期货 + 利率的日度价格表
    raw.to_csv(DATA / "L0-data collection-raw daily prices.csv")
    print(f"[saved] L0-data collection-raw daily prices.csv  {raw.shape[0]} rows x {raw.shape[1]} cols")

    print("Downloading shares outstanding ...")
    # 下载每股历史总股本（季度披露，用于后面的市值加权）
    shares = {}
    for t in STOCKS:
        try:
            s = yf.Ticker(t).get_shares_full(start=START, end=END)
            if s is not None and len(s) > 0:
                shares[t] = s[~s.index.duplicated(keep="last")].sort_index()
        except Exception:
            pass
    shares_df = pd.concat(shares, axis=1).sort_index()
    shares_df.index.name = "Date"
    shares_df.to_csv(DATA / "L0-data collection-shares outstanding.csv")
    print(f"[saved] L0-data collection-shares outstanding.csv  {shares_df.shape[0]} rows x {shares_df.shape[1]} cols")


if __name__ == "__main__":
    main()
