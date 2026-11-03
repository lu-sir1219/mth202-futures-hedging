# -*- coding: utf-8 -*-
"""
L2 — CAPM 回归：个股 beta + 组合 beta（4 种策略的公共底座）
MTH202 CW1 课题 A

功能：
  1) 用估计期(2023-2024)对 18 只股票各跑一次回归: (R_i - Rf) = α + β(R_m - Rf)
  2) 得到每只股票的 α、β、R²、t 值、标准误
  3) 算两个组合 beta：等权 βP(均值) 和 市值加权 βP(Σ权重×β)

输出：
  betas.csv  18 只股票的回归结果
依赖：returns_data.csv, market_cap.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "数据集"

STOCKS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","JPM","V","MA","XOM","CVX",
          "PG","KO","WMT","MCD","JNJ","UNH","CAT","HON"]
EST_START, EST_END = "2023-01-01", "2024-12-31"   # 估计期(24个月)

# ============ 1. 读数据 + 切估计期 ============
ret = pd.read_csv(DATA_DIR / "returns_data.csv", index_col=0, parse_dates=True)
est = ret.loc[EST_START:EST_END]
X = est["SPY_excess"].values            # 市场超额收益(自变量, SPY代理)

print(f"估计期: {est.index.min().date()} ~ {est.index.max().date()} ({len(est)} 天)\n")

# ============ 2. OLS 回归函数 ============
def ols(y, x):
    """一元线性回归 y = α + β·x，返回 α, β, R², t(β), se(β)"""
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

# ============ 3. 对 18 只股票回归 ============
rows = []
for t in STOCKS:
    y = est[t + "_excess"].values
    a, b, r2, tb, seb = ols(y, X)
    rows.append({"alpha(日)": a, "beta": b, "R2": r2, "t值(beta)": tb, "se(beta)": seb})

betas = pd.DataFrame(rows, index=STOCKS)
betas.index.name = "Ticker"

# ============ 4. 组合 beta ============
beta_eq = betas["beta"].mean()                      # 等权 = 简单平均

mcap = pd.read_csv(DATA_DIR / "market_cap.csv", index_col=0)
w = (mcap["权重%"] / 100.0).reindex(STOCKS)         # 市值权重(小数)
beta_mcap = (w * betas["beta"]).sum()               # 市值加权 = Σw·β

# ============ 4.5 交叉验证：直接回归组合收益序列 ============
# 原理：协方差线性 -> "先加权再回归" 应等于 "先回归再加权"
excess_cols = [t + "_excess" for t in STOCKS]
R_P_EW = est[excess_cols].mean(axis=1)              # 等权组合超额收益
R_P_MW = (est[excess_cols] * w.values).sum(axis=1)  # 市值加权组合超额收益

def beta_of(y, x):
    xc = x - x.mean(); yc = y - y.mean()
    return (xc * yc).sum() / (xc * xc).sum()

beta_EW_direct = beta_of(R_P_EW, X)
beta_MW_direct = beta_of(R_P_MW, X)

# ============ 5. 保存 + 打印 ============
betas = betas.round(4)
betas.to_csv(DATA_DIR / "betas.csv", encoding="utf-8-sig")

print("【个股 beta 回归结果】(估计期 2023-2024)")
print(betas.to_string())
print("\n【组合 beta 对比】")
print(f"  等权组合 βP       = {beta_eq:.4f}")
print(f"  市值加权组合 βP   = {beta_mcap:.4f}")
print(f"  两者之差           = {beta_eq - beta_mcap:+.4f}")
print(f"\n  个股 beta 范围: {betas['beta'].min():.2f} ~ {betas['beta'].max():.2f}")
print(f"  平均 R^2: {betas['R2'].mean():.3f} (说明市场解释了个股约 {betas['R2'].mean()*100:.0f}% 的波动)")

print("\n【交叉验证: 加权平均 vs 直接回归】")
print(f"  等权 βP:   加权平均 {beta_eq:.6f} | 直接回归 {beta_EW_direct:.6f} | 差异 {abs(beta_eq-beta_EW_direct):.2e}")
print(f"  市值 βP:   加权平均 {beta_mcap:.6f} | 直接回归 {beta_MW_direct:.6f} | 差异 {abs(beta_mcap-beta_MW_direct):.2e}")
print("  (差异≈0 说明两条路径一致, 代码与数据正确)")

print("\nL2 共享底座完成。")
