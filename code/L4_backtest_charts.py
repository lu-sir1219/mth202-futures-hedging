# -*- coding: utf-8 -*-
"""L4 - backtest charts (cumulative P&L, drawdown, regime bars, P&L distribution).

读取 ../Source_Data_034 里的 L4 回测结果，画出四张分析图，写到 ../Source_Diagram_034。
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
FIGS.mkdir(exist_ok=True)

daily = pd.read_csv(DATA / "L4-hedge backtest-daily P&L.csv", index_col="Date", parse_dates=True)
regime = pd.read_csv(DATA / "L4-hedge backtest-market regime summary.csv")
PORTFOLIO_VALUE = 25_000_000.0

COLORS = {"Unhedged": "#1f77b4", "MV": "#2ca02c", "CAPM": "#d62728"}
PORTS = [("Equal-weight", "Equal-weight Portfolio"),
         ("Market-cap-weight", "Market-cap-weighted Portfolio")]


def main():
    # 图 1：累计盈亏（逐日盈亏累加，单位百万美元）
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (pf, pname) in zip(axes, PORTS):
        for strat in ["Unhedged", "MV", "CAPM"]:
            # 累计盈亏 = 逐日盈亏累加，单位百万美元
            cum = daily[f"{pf}_{strat}"].cumsum() / 1e6
            ax.plot(cum.index, cum.values, label=strat, color=COLORS[strat], lw=1.8)
        ax.set_title(pname, fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative P&L (USD millions)")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date (2025)")
    fig.suptitle("Cumulative P&L | 2025 Out-of-Sample Backtest", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGS / "L4-backtest charts-cumulative P&L.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 2：回撤曲线
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (pf, pname) in zip(axes, PORTS):
        for strat in ["Unhedged", "MV", "CAPM"]:
            equity = PORTFOLIO_VALUE + daily[f"{pf}_{strat}"].cumsum()
            # 回撤 = 组合价值相对历史峰值的跌幅（%）
            dd = (equity / equity.cummax() - 1.0) * 100
            ax.plot(dd.index, dd.values, label=strat, color=COLORS[strat], lw=1.8)
        ax.set_title(pname, fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date (2025)")
    fig.suptitle("Portfolio Drawdown | 2025 Out-of-Sample Backtest", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGS / "L4-backtest charts-drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 3：各市场状态下的平均日盈亏
    regimes = ["Bullish", "Bearish", "Flat", "SPY_Down_Days"]
    labels = ["Bullish", "Bearish", "Flat", "SPY down days"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(regimes))
    width = 0.26
    for ax, (pf, pname) in zip(axes, PORTS):
        for i, strat in enumerate(["Unhedged", "MV", "CAPM"]):
            sub = regime[(regime["Portfolio"] == pf) & (regime["Strategy"] == strat)]
            # 各市场状态下的平均日盈亏（千美元）
            vals = [sub[sub["Regime"] == r]["Avg_Daily_PnL_USD"].values[0] / 1000 for r in regimes]
            ax.bar(x + (i - 1) * width, vals, width, label=strat, color=COLORS[strat])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(pname, fontsize=12, fontweight="bold")
        ax.set_ylabel("Average daily P&L (USD thousands)")
        ax.axhline(0, color="black", lw=0.6)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Average Daily P&L by Market Regime (2025)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "L4-backtest charts-market regime bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 图 4：日度盈亏分布
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (pf, pname) in zip(axes, PORTS):
        for strat in ["Unhedged", "MV", "CAPM"]:
            data = daily[f"{pf}_{strat}"] / 1000
            ax.hist(data.values, bins=40, alpha=0.5, label=strat, color=COLORS[strat], density=True)
        ax.set_title(pname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Daily P&L (USD thousands)")
        ax.set_ylabel("Probability density")
        ax.axvline(0, color="black", lw=0.6, ls="--")
        ax.legend(fontsize=9)
    fig.suptitle("Daily P&L Distribution: Hedged vs Unhedged (2025)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGS / "L4-backtest charts-P&L distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[saved] L4-backtest charts-* (4 figures) ->", FIGS)


if __name__ == "__main__":
    main()
