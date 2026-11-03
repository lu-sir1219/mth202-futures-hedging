# -*- coding: utf-8 -*-
"""MTH202 Scenario A — L4: 2025 样本外对冲回测（真实合约换月调整）。

策略：两种股票组合(等权/市值) × 三种对冲(不对冲/MV/CAPM) = 6 条。
口径：
  - 股票买入持有(2024-12-31 固定股数)，$25,000,000 起始；
  - 期货用 Bloomberg 具体合约按"到期前第5个有效日度报价日"换月构造的序列；
  - 手数为 L3 估出的整数(等权 MV60/CAPM66，市值 MV88/CAPM103)；
  - 交易成本忽略(毛收益口径)。

输出：L4_daily_backtest.csv, L4_performance_summary.csv, L4_market_regime_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

# =================== 配置 ===================
DATA_DIR = Path(r"D:\MTH202\L1+L2\数据集")
FUTURES_DIR = Path(r"D:\MTH202\期货数据")
OUT_DIR = Path(r"D:\MTH202\L4")
STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "MA",
    "XOM", "CVX", "PG", "KO", "WMT", "MCD", "JNJ", "UNH", "CAT", "HON",
]
BASE_DATE = "2024-12-31"
EVAL_START = "2025-01-01"
EVAL_END = "2025-12-31"
PORTFOLIO_VALUE = 25_000_000.0
ES_MULTIPLIER = 50.0

# L3 结果（买入持有整数手数）
HEDGE_RATIOS = {
    ("Equal-weight", "MV"): 60,
    ("Equal-weight", "CAPM"): 66,
    ("Market-cap-weight", "MV"): 88,
    ("Market-cap-weight", "CAPM"): 103,
}


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ---- 数据 ----
    prices = pd.read_csv(DATA_DIR / "clean_data.csv", index_col="Date", parse_dates=True).sort_index()
    market_cap = pd.read_csv(DATA_DIR / "market_cap.csv", index_col="Ticker")
    roll = pd.read_csv(FUTURES_DIR / "ES_roll_adjusted.csv", index_col=0, parse_dates=True)
    roll_points = roll["ES_roll_adj_dF"].astype(float)

    # ---- 组合股数(2024-12-31 固定) ----
    base_price = prices.loc[BASE_DATE, STOCKS].astype(float)
    w_equal = pd.Series(1.0 / len(STOCKS), index=STOCKS)
    w_mcap = (market_cap.loc[STOCKS, "权重%"].astype(float) / 100.0)
    shares_equal = PORTFOLIO_VALUE * w_equal / base_price
    shares_mcap = PORTFOLIO_VALUE * w_mcap / base_price

    # ---- 组合每日美元盈亏(2025) ----
    hist = prices.loc[BASE_DATE:EVAL_END, STOCKS].astype(float)
    dP_equal = hist.dot(shares_equal).diff().dropna()
    dP_mcap = hist.dot(shares_mcap).diff().dropna()

    # ---- 期货每日美元盈亏(换月调整, 对齐股票日历) ----
    stock_days = dP_equal.index
    dF_points = roll_points.reindex(stock_days)
    if dF_points.isna().any():
        raise ValueError("期货序列与股票日历存在缺口")
    dF_one = ES_MULTIPLIER * dF_points          # 一张多头合约的美元盈亏
    # 做空: 盈亏 = -N * 一张多头盈亏

    # ---- 逐日盈亏(6策略) ----
    daily = pd.DataFrame(index=stock_days)
    daily.index.name = "Date"
    daily["Equal-weight_Unhedged"] = dP_equal
    daily["Market-cap-weight_Unhedged"] = dP_mcap
    for (pf, method), N in HEDGE_RATIOS.items():
        dP = dP_equal if pf == "Equal-weight" else dP_mcap
        daily[f"{pf}_{method}"] = dP - N * dF_one

    # ---- 绩效指标 ----
    rows = []
    for pf in ["Equal-weight", "Market-cap-weight"]:
        dP = dP_equal if pf == "Equal-weight" else dP_mcap
        for strat, N in [("Unhedged", 0), ("MV", HEDGE_RATIOS[(pf, "MV")]),
                         ("CAPM", HEDGE_RATIOS[(pf, "CAPM")])]:
            col = f"{pf}_{strat}"
            pnl = daily[col]
            total = float(pnl.sum())
            ret = total / PORTFOLIO_VALUE
            std = float(pnl.std())
            vr = 1.0 - pnl.var() / dP.var() if strat != "Unhedged" else 0.0
            # 最大回撤(组合价值 + 期货累计盈亏; 初始$25m纳入峰值序列)
            cum = pnl.cumsum()
            equity = PORTFOLIO_VALUE + cum
            peaks = equity.cummax().clip(lower=PORTFOLIO_VALUE)
            dd = (equity / peaks - 1.0).min()
            rows.append({
                "Portfolio": pf, "Strategy": strat, "Contracts_Short": N,
                "Total_PnL_USD": total, "Total_Return": ret,
                "Daily_PnL_Std": std, "Variance_Reduction": vr,
                "Max_Drawdown": dd,
            })

    perf = pd.DataFrame(rows)
    perf.to_csv(OUT_DIR / "L4_performance_summary.csv", index=False, encoding="utf-8-sig",
                float_format="%.12g")
    daily.to_csv(OUT_DIR / "L4_daily_backtest.csv", encoding="utf-8-sig", float_format="%.12g")

    # ---- 市场状态(过去20交易日 SPY 收益 ±3%) ----
    spy = prices.loc[:, "SPY"].astype(float)
    spy20 = spy.pct_change(20)
    regime = pd.Series("Flat", index=spy.index)
    regime[spy20 > 0.03] = "Bullish"
    regime[spy20 < -0.03] = "Bearish"
    regime = regime.reindex(stock_days)

    # 每日 SPY 涨跌(下跌日)
    spy_daily = spy.pct_change().reindex(stock_days)
    is_down = spy_daily < 0

    regime_rows = []
    for pf in ["Equal-weight", "Market-cap-weight"]:
        for strat in ["Unhedged", "MV", "CAPM"]:
            col = f"{pf}_{strat}"
            pnl = daily[col]
            for rg in ["Bullish", "Bearish", "Flat"]:
                mask = regime == rg
                regime_rows.append({
                    "Portfolio": pf, "Strategy": strat, "Regime": rg,
                    "Days": int(mask.sum()),
                    "Total_PnL_USD": float(pnl[mask].sum()),
                    "Avg_Daily_PnL_USD": float(pnl[mask].mean()),
                })
            # 真实下跌日
            regime_rows.append({
                "Portfolio": pf, "Strategy": strat, "Regime": "SPY_Down_Days",
                "Days": int(is_down.sum()),
                "Total_PnL_USD": float(pnl[is_down].sum()),
                "Avg_Daily_PnL_USD": float(pnl[is_down].mean()),
            })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT_DIR / "L4_market_regime_summary.csv", index=False,
                     encoding="utf-8-sig", float_format="%.12g")

    # ---- 打印 ----
    print("=== L4 绩效汇总 (2025, 真实合约换月, 零成本) ===")
    print(perf[["Portfolio", "Strategy", "Contracts_Short", "Total_PnL_USD",
                "Total_Return", "Daily_PnL_Std", "Variance_Reduction", "Max_Drawdown"]]
          .to_string(index=False,
                     formatters={"Total_PnL_USD": "{:,.0f}".format,
                                 "Total_Return": "{:.2%}".format,
                                 "Daily_PnL_Std": "{:,.0f}".format,
                                 "Variance_Reduction": "{:.2%}".format,
                                 "Max_Drawdown": "{:.2%}".format}))
    print(f"\n已保存: {OUT_DIR/'L4_performance_summary.csv'}")
    print(f"        {OUT_DIR/'L4_daily_backtest.csv'}")
    print(f"        {OUT_DIR/'L4_market_regime_summary.csv'}")
    print("\n=== 市场状态分布 ===")
    print(regime.value_counts().to_string())


if __name__ == "__main__":
    main()
