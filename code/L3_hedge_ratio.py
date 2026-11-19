# -*- coding: utf-8 -*-
"""L3 - hedge ratios (minimum-variance and CAPM, in short contracts).

用最小方差法（Cov/Var）和 CAPM 法（beta x VP/VF）各算一套对冲手数，
并把两种组合（等权/市值）在买入持有/再平衡口径下的结果写出。
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"
BB = DATA / "bloomberg_futures contract"

STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "MA",
    "XOM", "CVX", "PG", "KO", "WMT", "MCD", "JNJ", "UNH", "CAT", "HON",
]
EST_START = "2023-01-01"
EST_END = "2024-12-31"
BASE_DATE = "2024-12-31"        # 建仓/估值基准日
ANCHOR_BASE = "2022-12-30"      # 估计期前一交易日，构造首日价格变动用
PORTFOLIO_VALUE = 25_000_000.0
ES_MULTIPLIER = 50.0


def load_csv(name, **kwargs):
    fp = DATA / name
    if not fp.is_file():
        raise FileNotFoundError(f"missing {fp}")
    return pd.read_csv(fp, **kwargs)


def round_contracts(n):
    return int(np.sign(n) * np.floor(abs(n) + 0.5))


def buy_and_hold_dollar_pnl(hist_prices, weights):
    # 买入持有：2024-12-31 锁定股数，之后算组合每日美元盈亏
    base_price = hist_prices.loc[BASE_DATE, weights.index]
    shares = PORTFOLIO_VALUE * weights / base_price
    value = hist_prices[weights.index].dot(shares)
    return value.diff().dropna()


def rebalanced_dollar_pnl(hist_prices, weights, freq):
    # 月度/季度再平衡到目标权重，组合价值锚定 2024-12-31 = $25m
    simple = hist_prices[weights.index].pct_change().dropna()
    dates = simple.index
    w = weights.copy()
    R_list = []
    for i, date in enumerate(dates):
        r_i = simple.loc[date]
        R_t = float((w * r_i).sum())
        R_list.append(R_t)
        w = w * (1.0 + r_i) / (1.0 + R_t)
        if i + 1 < len(dates):
            nxt = dates[i + 1]
            reset = (freq == "monthly" and nxt.month != date.month) or \
                    (freq == "quarterly" and nxt.quarter != date.quarter)
            if reset:
                w = weights.copy()
    R = pd.Series(R_list, index=dates)
    v_units = (1.0 + R).cumprod()
    v_units = pd.concat([pd.Series([1.0], index=[hist_prices.index[0]]), v_units])
    scale = PORTFOLIO_VALUE / v_units.iloc[-1]
    return (v_units * scale).diff().dropna()


def mv_contracts(dP, dF):
    # 最小方差手数 = Cov(组合盈亏, 合约盈亏) / Var(合约盈亏)
    return float(dP.cov(dF) / dF.var())


def main():
    prices = load_csv("L1-data cleaning-cleaned daily prices.csv", index_col="Date", parse_dates=True).sort_index()
    market_cap = load_csv("L2-market cap weights.csv", index_col="Ticker")
    betas = load_csv("L2-beta estimation.csv", index_col="Ticker")

    if set(STOCKS) - set(prices.columns) or "ES_F" not in prices.columns:
        raise ValueError("cleaned prices are missing stocks or ES_F")
    if set(STOCKS) - set(market_cap.index) or "weight_pct" not in market_cap.columns:
        raise ValueError("market-cap weights are missing stocks or weight_pct")
    if set(STOCKS) - set(betas.index) or "beta" not in betas.columns:
        raise ValueError("betas are missing stocks or beta")
    if pd.Timestamp(BASE_DATE) not in prices.index or pd.Timestamp(ANCHOR_BASE) not in prices.index:
        raise ValueError(f"missing {BASE_DATE} or {ANCHOR_BASE}")

    hist = prices.loc[ANCHOR_BASE:BASE_DATE][STOCKS + ["ES_F"]].astype(float)
    if not np.isfinite(hist.to_numpy()).all() or (hist <= 0).any().any():
        raise ValueError("non-positive or non-finite prices in estimation window")

    w_equal = pd.Series(1.0 / len(STOCKS), index=STOCKS)
    w_mcap = (market_cap.loc[STOCKS, "weight_pct"].astype(float) / 100.0)
    for name, w in [("Equal", w_equal), ("Market-cap", w_mcap)]:
        if not np.isclose(w.sum(), 1.0, atol=1e-6) or (w < 0).any():
            raise ValueError(f"{name} weights invalid")
    stock_betas = betas.loc[STOCKS, "beta"].astype(float)

    roll = load_csv("L3-roll adjustment-roll adjusted futures series.csv", index_col=0, parse_dates=True)
    roll_points = roll["ES_roll_adj_dF"].astype(float)
    stock_days = hist[STOCKS].diff().dropna().index
    dF_points = roll_points.reindex(stock_days)
    if dF_points.isna().any():
        raise ValueError("futures series has gaps vs stock calendar")
    dF = ES_MULTIPLIER * dF_points

    # f0 取 2024-12-31 实际持有的那张合约（ESH25）的收盘价
    esh25 = pd.read_excel(BB / "L0-data collection-ESH25 futures contract.xlsx", header=6)
    esh25 = esh25.rename(columns={esh25.columns[0]: "Date"})
    esh25["Date"] = pd.to_datetime(esh25["Date"])
    f0 = float(esh25.set_index("Date")["PX_LAST"].loc[BASE_DATE])
    vf = f0 * ES_MULTIPLIER

    dP = {
        ("Equal-weight", "Buy-and-hold"): buy_and_hold_dollar_pnl(hist, w_equal),
        ("Equal-weight", "Monthly rebalance"): rebalanced_dollar_pnl(hist, w_equal, "monthly"),
        ("Equal-weight", "Quarterly rebalance"): rebalanced_dollar_pnl(hist, w_equal, "quarterly"),
        ("Market-cap-weight", "Buy-and-hold"): buy_and_hold_dollar_pnl(hist, w_mcap),
    }

    rows = []
    daily_pnl = pd.DataFrame(index=dF.index)
    daily_pnl.index.name = "Date"
    daily_pnl["ES_Change_Points_Roll_Adjusted"] = dF_points
    daily_pnl["ES_One_Contract_Dollar_Change"] = dF

    for (pf, hold), pnl in dP.items():
        if not pnl.index.equals(dF.index):
            raise ValueError(f"{pf}/{hold} dates not aligned")
        daily_pnl[f"{pf}_{hold}_Portfolio_Dollar_Change"] = pnl
        n = mv_contracts(pnl, dF)
        rows.append({
            "Portfolio": pf, "Holding_Assumption": hold, "Method": "Minimum-Variance",
            "Hedge_Ratio_Value_Basis": n * vf / PORTFOLIO_VALUE, "Contracts_Unrounded": n,
            "Contracts_Signed": round_contracts(n),
            "Direction": "SHORT" if round_contracts(n) > 0 else "NONE",
        })

    for pf, w in [("Equal-weight", w_equal), ("Market-cap-weight", w_mcap)]:
        beta_p = float(w.dot(stock_betas))
        # CAPM 手数 = beta_P x V_P / V_F
        n = beta_p * PORTFOLIO_VALUE / vf
        rows.append({
            "Portfolio": pf, "Holding_Assumption": "N/A (target weights)",
            "Method": "CAPM", "Hedge_Ratio_Value_Basis": beta_p,
            "Contracts_Unrounded": n, "Contracts_Signed": round_contracts(n),
            "Direction": "SHORT" if round_contracts(n) > 0 else "NONE",
        })

    result = pd.DataFrame(rows)
    result.to_csv(DATA / "L3-hedge ratio-MV and CAPM contract counts.csv",
                  index=False, encoding="utf-8-sig", float_format="%.12g")
    daily_pnl.to_csv(DATA / "L3-hedge ratio-estimation daily dollar P&L.csv",
                     encoding="utf-8-sig", float_format="%.12g")
    print("[saved] L3-hedge ratio-MV and CAPM contract counts.csv")
    print("[saved] L3-hedge ratio-estimation daily dollar P&L.csv")
    print(result[["Portfolio", "Holding_Assumption", "Method", "Hedge_Ratio_Value_Basis",
                  "Contracts_Unrounded", "Contracts_Signed", "Direction"]].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nEstimation days: {len(dF)} | F0 (2024-12-31 ESH25) = {f0:.2f} | VF = ${vf:,.2f}")


if __name__ == "__main__":
    main()
