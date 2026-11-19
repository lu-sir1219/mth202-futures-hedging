# -*- coding: utf-8 -*-
"""L4 poster figures #1: cumulative P&L, drawdown and average-daily-P&L-by-regime.

海报图 1：累计盈亏、回撤、以及按市场状态划分的平均日盈亏柱状图。
读取 ../Source_Data_034 里的 L4 回测 CSV，输出 PNG 到 ../Source_Diagram_034/poster。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"
FIG = ROOT / "Source_Diagram_034" / "poster"
FIG.mkdir(parents=True, exist_ok=True)

PORTFOLIO_VALUE = 25_000_000.0   # 组合名义本金 2500 万美元
COLORS = {"Unhedged": "#1f77b4", "MV": "#2ca02c", "CAPM": "#d62728"}
PORTS = [("Equal-weight", "Equal-weight Portfolio"),
         ("Market-cap-weight", "Market-cap-weighted Portfolio")]

# 读取 L4 的逐日美元盈亏（250 个样本外交易日，六条策略路径）
daily = pd.read_csv(DATA / "L4-hedge backtest-daily P&L.csv", index_col="Date", parse_dates=True)


def main():
    # ---- 图 1：累计盈亏曲线（累计求和，单位百万美元） ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (pf, pname) in zip(axes, PORTS):
        for strat in ["Unhedged", "MV", "CAPM"]:
            cum = daily[f"{pf}_{strat}"].cumsum() / 1e6
            ax.plot(cum.index, cum.values, label=strat, color=COLORS[strat], lw=1.8)
        ax.set_title(pname, fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative P&L (USD millions)")
        ax.axhline(0, color="black", lw=0.6, ls="--")   # 盈亏平衡线
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date (2025)")
    fig.suptitle("Cumulative P&L | 2025 Out-of-Sample Backtest", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cumulative P&L (2025).png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图 2：回撤曲线（组合价值相对历史峰值的最大跌幅，%） ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (pf, pname) in zip(axes, PORTS):
        for strat in ["Unhedged", "MV", "CAPM"]:
            equity = PORTFOLIO_VALUE + daily[f"{pf}_{strat}"].cumsum()   # 组合价值 = 本金 + 累计盈亏
            dd = (equity / equity.cummax() - 1.0) * 100                  # 相对历史峰值回撤
            ax.plot(dd.index, dd.values, label=strat, color=COLORS[strat], lw=1.8)
        ax.set_title(pname, fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date (2025)")
    fig.suptitle("Portfolio Drawdown | 2025 Out-of-Sample Backtest", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "drawdown (2025).png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 图 3：各市场状态下的平均日盈亏（20 日涨跌 ±3% 划分牛/熊/横盘） ----
    regime = pd.read_csv(DATA / "L4-hedge backtest-market regime summary.csv")
    regimes = ["Bullish", "Bearish", "Flat", "SPY_Down_Days"]
    labels = ["Bullish", "Bearish", "Flat", "SPY down days"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(regimes))
    width = 0.26
    for ax, (pf, pname) in zip(axes, PORTS):
        for i, strat in enumerate(["Unhedged", "MV", "CAPM"]):
            sub = regime[(regime["Portfolio"] == pf) & (regime["Strategy"] == strat)]
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
    fig.savefig(FIG / "average daily P&L by regime (2025).png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[saved]", FIG)
    for f in ["cumulative P&L (2025).png", "drawdown (2025).png", "average daily P&L by regime (2025).png"]:
        print("  -", f)


if __name__ == "__main__":
    main()
