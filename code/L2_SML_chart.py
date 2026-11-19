# -*- coding: utf-8 -*-
"""L2 - Security Market Line plot (annualized return vs beta).

证券市场线：把 18 只股票的年化收益对 beta 画散点，叠加 CAPM 定价线 E(R)=Rf+beta·溢价。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"
FIGS = ROOT / "Source_Diagram_034"
FIGS.mkdir(parents=True, exist_ok=True)

STOCKS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "MA", "XOM", "CVX",
          "PG", "KO", "WMT", "MCD", "JNJ", "UNH", "CAT", "HON"]
EST_START, EST_END = "2023-01-01", "2024-12-31"


def main():
    betas = pd.read_csv(DATA / "L2-beta estimation.csv", index_col=0)["beta"]
    ret = pd.read_csv(DATA / "L1-data cleaning-daily returns.csv", index_col=0, parse_dates=True)
    est = ret.loc[EST_START:EST_END]

    # 年化无风险利率（^IRX 日度均值 x 252）与市场收益、风险溢价
    rf_annual = est["RF_daily"].mean() * 252
    rm_annual = est["SPY"].mean() * 252
    rp_annual = rm_annual - rf_annual
    actual = {t: est[t].mean() * 252 for t in STOCKS}

    fig, ax = plt.subplots(figsize=(10, 6.5))
    beta_range = np.linspace(0, betas.max() * 1.1, 100)
    # 证券市场线：E(R) = Rf + beta x 风险溢价
    ax.plot(beta_range, (rf_annual + beta_range * rp_annual) * 100, color="tab:red", lw=2,
            label=f"SML: E(R) = {rf_annual*100:.2f}% + beta x {rp_annual*100:.2f}%")
    for t in STOCKS:
        b, a = betas[t], actual[t] * 100
        pred = (rf_annual + b * rp_annual) * 100
        # 蓝点=实际收益在 SML 上方（正 alpha），灰点=下方（负 alpha）
        ax.scatter(b, a, color="tab:blue" if a >= pred else "tab:gray", s=45, zorder=3)
        ax.annotate(t, (b, a), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.scatter([0], [rf_annual * 100], color="black", s=60, marker="s", zorder=4)
    ax.annotate("Rf", (0, rf_annual * 100), textcoords="offset points", xytext=(8, -14), fontsize=10)
    ax.scatter([1], [rm_annual * 100], color="black", s=60, marker="D", zorder=4)
    ax.annotate("Market (beta=1)", (1, rm_annual * 100), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.set_xlabel("Beta", fontsize=12)
    ax.set_ylabel("Annualized Return (%)", fontsize=12)
    ax.set_title("Security Market Line (SML)", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGS / "L2-SML chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[saved] L2-SML chart.png")
    print(f"Rf (annualised)      = {rf_annual*100:.2f}%")
    print(f"Market return (ann.) = {rm_annual*100:.2f}%")
    print(f"Risk premium         = {rp_annual*100:.2f}%")


if __name__ == "__main__":
    main()
