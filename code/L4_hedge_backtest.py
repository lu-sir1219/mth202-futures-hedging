# -*- coding: utf-8 -*-
"""L4 - out-of-sample hedge backtest (2025): 2 portfolios x 3 hedges = 6 paths.

对六条策略路径（等权/市值 x 不对冲/MV/CAPM）算全年收益、日波动、方差降低、
最大回撤，并按市场状态切分，写出三张 CSV 并打印绩效表。
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "MA",
    "XOM", "CVX", "PG", "KO", "WMT", "MCD", "JNJ", "UNH", "CAT", "HON",
]
BASE_DATE = "2024-12-31"
EVAL_END = "2025-12-31"
PORTFOLIO_VALUE = 25_000_000.0
ES_MULTIPLIER = 50.0

# L3 估计出的买入持有整数手数（做空）
HEDGE_RATIOS = {
    ("Equal-weight", "MV"): 60,
    ("Equal-weight", "CAPM"): 66,
    ("Market-cap-weight", "MV"): 88,
    ("Market-cap-weight", "CAPM"): 103,
}


def main():
    prices = pd.read_csv(DATA / "L1-data cleaning-cleaned daily prices.csv",
                         index_col="Date", parse_dates=True).sort_index()
    market_cap = pd.read_csv(DATA / "L2-market cap weights.csv", index_col="Ticker")
    roll = pd.read_csv(DATA / "L3-roll adjustment-roll adjusted futures series.csv",
                       index_col=0, parse_dates=True)
    roll_points = roll["ES_roll_adj_dF"].astype(float)

    # 2024-12-31 锁定股数，买入持有
    base_price = prices.loc[BASE_DATE, STOCKS].astype(float)
    w_equal = pd.Series(1.0 / len(STOCKS), index=STOCKS)
    w_mcap = (market_cap.loc[STOCKS, "weight_pct"].astype(float) / 100.0)
    shares_equal = PORTFOLIO_VALUE * w_equal / base_price
    shares_mcap = PORTFOLIO_VALUE * w_mcap / base_price

    hist = prices.loc[BASE_DATE:EVAL_END, STOCKS].astype(float)
    dP_equal = hist.dot(shares_equal).diff().dropna()
    dP_mcap = hist.dot(shares_mcap).diff().dropna()

    stock_days = dP_equal.index
    dF_points = roll_points.reindex(stock_days)
    if dF_points.isna().any():
        raise ValueError("futures series has gaps vs stock calendar")
    dF_one = ES_MULTIPLIER * dF_points

    daily = pd.DataFrame(index=stock_days)
    daily.index.name = "Date"
    daily["Equal-weight_Unhedged"] = dP_equal
    daily["Market-cap-weight_Unhedged"] = dP_mcap
    for (pf, method), N in HEDGE_RATIOS.items():
        dP = dP_equal if pf == "Equal-weight" else dP_mcap
        # 对冲后日盈亏 = 股票盈亏 - N x 一张合约盈亏（做空）
        daily[f"{pf}_{method}"] = dP - N * dF_one

    rows = []
    for pf in ["Equal-weight", "Market-cap-weight"]:
        dP = dP_equal if pf == "Equal-weight" else dP_mcap
        for strat, N in [("Unhedged", 0), ("MV", HEDGE_RATIOS[(pf, "MV")]),
                         ("CAPM", HEDGE_RATIOS[(pf, "CAPM")])]:
            pnl = daily[f"{pf}_{strat}"]
            cum = pnl.cumsum()
            equity = PORTFOLIO_VALUE + cum
            peaks = equity.cummax().clip(lower=PORTFOLIO_VALUE)
            rows.append({
                "Portfolio": pf, "Strategy": strat, "Contracts_Short": N,
                "Total_PnL_USD": float(pnl.sum()), "Total_Return": float(pnl.sum()) / PORTFOLIO_VALUE,
                "Daily_PnL_Std": float(pnl.std()),
                # 方差降低 = 1 - 对冲后方差 / 不对冲方差（对冲有效性的核心指标）
                "Variance_Reduction": (1.0 - pnl.var() / dP.var()) if strat != "Unhedged" else 0.0,
                # 最大回撤 = 组合价值相对历史峰值的最小跌幅
                "Max_Drawdown": float((equity / peaks - 1.0).min()),
            })

    perf = pd.DataFrame(rows)
    perf.to_csv(DATA / "L4-hedge backtest-performance summary.csv",
                index=False, encoding="utf-8-sig", float_format="%.12g")
    daily.to_csv(DATA / "L4-hedge backtest-daily P&L.csv",
                 encoding="utf-8-sig", float_format="%.12g")

    spy = prices.loc[:, "SPY"].astype(float)
    spy20 = spy.pct_change(20)
    regime = pd.Series("Flat", index=spy.index)
    # 过去 20 日 SPY 涨跌 ±3% 划分牛/熊/横盘
    regime[spy20 > 0.03] = "Bullish"
    regime[spy20 < -0.03] = "Bearish"
    regime = regime.reindex(stock_days)
    spy_daily = spy.pct_change().reindex(stock_days)
    is_down = spy_daily < 0

    regime_rows = []
    for pf in ["Equal-weight", "Market-cap-weight"]:
        for strat in ["Unhedged", "MV", "CAPM"]:
            pnl = daily[f"{pf}_{strat}"]
            for rg in ["Bullish", "Bearish", "Flat"]:
                mask = regime == rg
                regime_rows.append({
                    "Portfolio": pf, "Strategy": strat, "Regime": rg,
                    "Days": int(mask.sum()), "Total_PnL_USD": float(pnl[mask].sum()),
                    "Avg_Daily_PnL_USD": float(pnl[mask].mean()),
                })
            regime_rows.append({
                "Portfolio": pf, "Strategy": strat, "Regime": "SPY_Down_Days",
                "Days": int(is_down.sum()), "Total_PnL_USD": float(pnl[is_down].sum()),
                "Avg_Daily_PnL_USD": float(pnl[is_down].mean()),
            })

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(DATA / "L4-hedge backtest-market regime summary.csv",
                     index=False, encoding="utf-8-sig", float_format="%.12g")

    print("[saved] L4-hedge backtest-performance summary.csv")
    print("[saved] L4-hedge backtest-daily P&L.csv")
    print("[saved] L4-hedge backtest-market regime summary.csv\n")
    print(perf[["Portfolio", "Strategy", "Contracts_Short", "Total_PnL_USD", "Total_Return",
                "Daily_PnL_Std", "Variance_Reduction", "Max_Drawdown"]].to_string(
        index=False,
        formatters={"Total_PnL_USD": "{:,.0f}".format, "Total_Return": "{:.2%}".format,
                    "Daily_PnL_Std": "{:,.0f}".format, "Variance_Reduction": "{:.2%}".format,
                    "Max_Drawdown": "{:.2%}".format}))
    print("\nRegime counts:\n" + regime.value_counts().to_string())


if __name__ == "__main__":
    main()
