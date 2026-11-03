# -*- coding: utf-8 -*-
"""
L2 补充 — 证券市场线 SML 图
MTH202 CW1 课题 A

功能：
  1) 读 betas.csv 与 returns_data.csv
  2) 用估计期(2023-2024)算年化期望收益、市场风险溢价、无风险利率
  3) 画 SML：E(R_i) = Rf + β_i × (E(Rm) - Rf)
  4) 标出 18 只股票的实际收益，对比 CAPM 预测
  5) 输出 SML.png

依赖：betas.csv, returns_data.csv
"""

import matplotlib
matplotlib.use("Agg")   # 无窗口后端
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "数据集"

STOCKS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","JPM","V","MA","XOM","CVX",
          "PG","KO","WMT","MCD","JNJ","UNH","CAT","HON"]
EST_START, EST_END = "2023-01-01", "2024-12-31"

# ============ 1. 读数据 ============
betas = pd.read_csv(DATA_DIR / "betas.csv", index_col=0)["beta"]
ret = pd.read_csv(DATA_DIR / "returns_data.csv", index_col=0, parse_dates=True)
est = ret.loc[EST_START:EST_END]

# ============ 2. 年化参数 ============
rf_annual = est["RF_daily"].mean() * 252        # 无风险利率(年化)
rm_annual = est["SPY"].mean() * 252             # 市场期望收益(年化)
rp_annual = rm_annual - rf_annual               # 市场风险溢价(年化)

# 每只股票的年化实际收益
actual = {t: est[t].mean() * 252 for t in STOCKS}

# ============ 3. 画 SML ============
fig, ax = plt.subplots(figsize=(10, 6.5))

# SML 直线: E(R) = Rf + β × 风险溢价
beta_range = np.linspace(0, betas.max() * 1.1, 100)
sml_y = rf_annual + beta_range * rp_annual
ax.plot(beta_range, sml_y * 100, color="tab:red", lw=2,
        label=f"SML: E(R) = {rf_annual*100:.2f}% + β × {rp_annual*100:.2f}%")

# 18 只股票的点
for t in STOCKS:
    b = betas[t]
    a = actual[t] * 100
    pred = (rf_annual + b * rp_annual) * 100
    above = a >= pred
    ax.scatter(b, a, color="tab:blue" if above else "tab:gray", s=45, zorder=3)
    ax.annotate(t, (b, a), textcoords="offset points", xytext=(6, 4), fontsize=8)

# 无风险利率点与市场组合点
ax.scatter([0], [rf_annual * 100], color="black", s=60, zorder=4, marker="s")
ax.annotate("Rf", (0, rf_annual * 100), textcoords="offset points", xytext=(8, -14), fontsize=10)
ax.scatter([1], [rm_annual * 100], color="black", s=60, zorder=4, marker="D")
ax.annotate("Market (β=1)", (1, rm_annual * 100), textcoords="offset points", xytext=(8, 6), fontsize=10)

ax.set_xlabel("Beta (β)", fontsize=12)
ax.set_ylabel("Annualized Return (%)", fontsize=12)
ax.set_title("Security Market Line (SML)", fontsize=14, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
ax.axhline(0, color="black", lw=0.8)

plt.tight_layout()
out = BASE_DIR / "SML.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"已保存 SML 图: {out}")

print(f"\n参数: Rf(年化)={rf_annual*100:.2f}%, 市场收益(年化)={rm_annual*100:.2f}%, 风险溢价={rp_annual*100:.2f}%")
print("个股实际年化收益 vs CAPM 预测:")
print(f"{'股票':6s} {'beta':>6s} {'实际%':>8s} {'预测%':>8s} {'偏离(α)%':>8s}")
for t in STOCKS:
    b = betas[t]
    act = actual[t] * 100
    pred = (rf_annual + b * rp_annual) * 100
    print(f"{t:6s} {b:6.2f} {act:8.2f} {pred:8.2f} {act-pred:8.2f}")
