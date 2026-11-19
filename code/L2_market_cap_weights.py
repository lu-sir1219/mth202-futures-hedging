# -*- coding: utf-8 -*-
"""L2 - market-cap weights (2024-12-31 valuation date).

市值 = 真实成交价 x 股本。未复权价从 Yahoo 下载（需联网）；权重表结果已随附，
所以后面的 L3/L4 可以完全离线运行。
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

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "V", "MA",
    "XOM", "CVX",
    "PG", "KO", "WMT", "MCD",
    "JNJ", "UNH",
    "CAT", "HON",
]
VALUE_DATE = "2024-12-31"   # 对冲起始日前最后一个交易日，作为估值基准


def configure_proxy():
    """仅当 127.0.0.1:7897 有代理在监听时才启用，否则直连。"""
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
    print("Downloading unadjusted closes around 2024-12-31 ...")
    px = yf.download(STOCKS, start="2024-12-01", end="2025-01-10",
                     auto_adjust=False, group_by="column", threads=False, progress=False)
    close = px["Close"]
    # 取 2024-12-31 的未复权收盘价（市值要用真实成交价，不用复权价）
    price = close.loc[VALUE_DATE] if VALUE_DATE in close.index else close.asof(VALUE_DATE)

    sh = pd.read_csv(DATA / "L0-data collection-shares outstanding.csv", index_col=0)
    sh.index = pd.to_datetime(sh.index, utc=True).tz_localize(None)
    shares = sh.apply(lambda c: c.dropna().asof(pd.Timestamp(VALUE_DATE)))

    mcap = pd.DataFrame({"shares_outstanding": shares, "unadjusted_price": price})
    # 市值（十亿美元）= 股本 x 未复权价 / 1e9
    mcap["market_cap_usd_bn"] = (mcap["shares_outstanding"] * mcap["unadjusted_price"]) / 1e9
    # 权重 % = 个股市值 / 组合总市值
    mcap["weight_pct"] = mcap["market_cap_usd_bn"] / mcap["market_cap_usd_bn"].sum() * 100
    mcap = mcap.sort_values("market_cap_usd_bn", ascending=False)
    mcap.index.name = "Ticker"
    mcap.to_csv(DATA / "L2-market cap weights.csv", encoding="utf-8-sig")
    print("[saved] L2-market cap weights.csv")
    print(mcap.round(4).to_string())


if __name__ == "__main__":
    main()
