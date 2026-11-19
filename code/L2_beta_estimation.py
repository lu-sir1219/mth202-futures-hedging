# -*- coding: utf-8 -*-
"""L2 - CAPM beta estimation (OLS on 2023-2024 excess returns).

对 18 只股票各跑一次超额收益回归，得到个股 beta，再聚合出等权/市值加权两个
组合 beta，写入 ../Source_Data_034。
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"

STOCKS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "MA", "XOM", "CVX",
          "PG", "KO", "WMT", "MCD", "JNJ", "UNH", "CAT", "HON"]
EST_START, EST_END = "2023-01-01", "2024-12-31"   # 24 个月估计期


def ols(y, x):
    # 一元回归 y = alpha + beta*x，返回 alpha/beta/R^2/t 值/标准误
    Xm = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    alpha, beta = coef
    resid = y - Xm @ coef
    n, k = len(y), 2
    dof = n - k
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(Xm.T @ Xm)
    se_beta = np.sqrt(cov[1, 1])
    t_beta = beta / se_beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / ss_tot
    return alpha, beta, r2, t_beta, se_beta


def beta_of(y, x):
    xc = x - x.mean()
    yc = y - y.mean()
    return (xc * yc).sum() / (xc * xc).sum()


def main():
    ret = pd.read_csv(DATA / "L1-data cleaning-daily returns.csv", index_col=0, parse_dates=True)
    est = ret.loc[EST_START:EST_END]
    X = est["SPY_excess"].values

    rows = []
    for t in STOCKS:
        a, b, r2, tb, seb = ols(est[t + "_excess"].values, X)
        rows.append({"alpha_daily": a, "beta": b, "R2": r2, "t_stat_beta": tb, "se_beta": seb})

    betas = pd.DataFrame(rows, index=STOCKS)
    betas.index.name = "Ticker"

    # 等权组合 beta = 18 只个股 beta 的简单平均
    beta_eq = betas["beta"].mean()
    mcap = pd.read_csv(DATA / "L2-market cap weights.csv", index_col=0)
    w = (mcap["weight_pct"] / 100.0).reindex(STOCKS)
    # 市值加权组合 beta = Σ w_i * beta_i
    beta_mcap = (w * betas["beta"]).sum()

    # 交叉验证：协方差是线性的，所以"先加权再回归"应等于"先回归再加权"
    excess_cols = [t + "_excess" for t in STOCKS]
    R_P_EW = est[excess_cols].mean(axis=1)
    R_P_MW = (est[excess_cols] * w.values).sum(axis=1)
    beta_EW_direct = beta_of(R_P_EW, X)
    beta_MW_direct = beta_of(R_P_MW, X)

    betas = betas.round(4)
    portfolio = pd.DataFrame(
        {"alpha_daily": [np.nan, np.nan], "beta": [beta_eq, beta_mcap],
         "R2": [np.nan, np.nan], "t_stat_beta": [np.nan, np.nan], "se_beta": [np.nan, np.nan]},
        index=["Portfolio_Equal-weight", "Portfolio_Market-cap-weight"])
    portfolio.index.name = "Ticker"
    betas = pd.concat([betas, portfolio])
    betas.to_csv(DATA / "L2-beta estimation.csv", encoding="utf-8-sig")
    print("[saved] L2-beta estimation.csv")
    print(betas.to_string())
    print(f"\nEqual-weight portfolio beta = {beta_eq:.4f}")
    print(f"Market-cap-weighted portfolio beta = {beta_mcap:.4f}")
    print("\nCross-check (weighted average vs direct regression, should be ~0):")
    print(f"  Equal-weight: {beta_eq:.6f} vs {beta_EW_direct:.6f}, diff {abs(beta_eq - beta_EW_direct):.2e}")
    print(f"  Market-cap  : {beta_mcap:.6f} vs {beta_MW_direct:.6f}, diff {abs(beta_mcap - beta_MW_direct):.2e}")


if __name__ == "__main__":
    main()
