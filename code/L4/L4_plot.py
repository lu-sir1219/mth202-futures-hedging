# -*- coding: utf-8 -*-
"""L4 图表生成：累计盈亏曲线 / 回撤曲线 / 市场状态对比。"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(r"D:\MTH202\L4")
daily = pd.read_csv(OUT / "L4_daily_backtest.csv", index_col="Date", parse_dates=True)
PORTFOLIO_VALUE = 25_000_000.0

COLORS = {"Unhedged": "#1f77b4", "MV": "#2ca02c", "CAPM": "#d62728"}
PORTS = [("Equal-weight", "等权组合"), ("Market-cap-weight", "市值加权组合")]

# ============ 图1: 累计盈亏曲线 ============
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for ax, (pf, pname) in zip(axes, PORTS):
    for strat in ["Unhedged", "MV", "CAPM"]:
        cum = daily[f"{pf}_{strat}"].cumsum() / 1e6   # 百万美元
        ax.plot(cum.index, cum.values, label=f"{strat}",
                color=COLORS[strat], lw=1.8)
    ax.set_title(f"{pname}", fontsize=12, fontweight="bold")
    ax.set_ylabel("累计盈亏 (百万美元)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("2025 年")
fig.suptitle("L4 样本外对冲回测：累计盈亏曲线（2025）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "L4_cumulative_pnl.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============ 图2: 回撤曲线 ============
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for ax, (pf, pname) in zip(axes, PORTS):
    for strat in ["Unhedged", "MV", "CAPM"]:
        cum = daily[f"{pf}_{strat}"].cumsum()
        equity = PORTFOLIO_VALUE + cum
        dd = (equity / equity.cummax() - 1.0) * 100
        ax.plot(dd.index, dd.values, label=f"{strat}",
                color=COLORS[strat], lw=1.8)
    ax.set_title(f"{pname}", fontsize=12, fontweight="bold")
    ax.set_ylabel("回撤 (%)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("2025 年")
fig.suptitle("L4 样本外对冲回测：回撤曲线（2025）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "L4_drawdown.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============ 图3: 市场状态对比 ============
regime = pd.read_csv(OUT / "L4_market_regime_summary.csv")
regimes = ["Bullish", "Bearish", "Flat", "SPY_Down_Days"]
labels_cn = ["牛市", "熊市", "横盘", "下跌日"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
x = np.arange(len(regimes))
width = 0.26
for ax, (pf, pname) in zip(axes, PORTS):
    for i, strat in enumerate(["Unhedged", "MV", "CAPM"]):
        sub = regime[(regime["Portfolio"] == pf) & (regime["Strategy"] == strat)]
        vals = [sub[sub["Regime"] == r]["Avg_Daily_PnL_USD"].values[0] for r in regimes]
        vals = [v / 1000 for v in vals]   # 千美元
        ax.bar(x + (i - 1) * width, vals, width, label=strat,
               color=COLORS[strat])
    ax.set_xticks(x)
    ax.set_xticklabels(labels_cn, fontsize=10)
    ax.set_title(f"{pname}", fontsize=12, fontweight="bold")
    ax.set_ylabel("平均日盈亏 (千美元)")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
fig.suptitle("L4 各市场状态下的平均日盈亏（2025）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "L4_regime_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ============ 图4: 日度盈亏分布 ============
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (pf, pname) in zip(axes, PORTS):
    for strat in ["Unhedged", "MV", "CAPM"]:
        data = daily[f"{pf}_{strat}"] / 1000   # 千美元
        ax.hist(data.values, bins=40, alpha=0.5, label=strat,
                color=COLORS[strat], density=True)
    ax.set_title(f"{pname}", fontsize=12, fontweight="bold")
    ax.set_xlabel("日盈亏 (千美元)")
    ax.set_ylabel("概率密度")
    ax.axvline(0, color="black", lw=0.6, ls="--")
    ax.legend(fontsize=9)
fig.suptitle("L4 日度盈亏分布：不对冲 vs 对冲（2025）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "L4_pnl_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("已生成:")
for f in ["L4_cumulative_pnl.png", "L4_drawdown.png", "L4_regime_bar.png",
          "L4_pnl_distribution.png"]:
    print(f"  {OUT/f}")
