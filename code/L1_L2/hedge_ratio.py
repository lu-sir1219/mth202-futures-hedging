# -*- coding: utf-8 -*-
"""MTH202 Scenario A — L3: 静态对冲比率（买入持有 + 月度/季度再平衡）。

方法：
  - MV 最小方差：组合每日美元盈亏 对 一张期货每日美元盈亏 回归，斜率即手数。
  - CAPM：N = βP × VP / VF（βP 用 L2 目标权重加权 beta）。

组合：
  - Equal-weight：买入持有、月度再平衡、季度再平衡 三个变体 + CAPM。
  - Market-cap-weight：买入持有（市值加权自我维持，不需再平衡）+ CAPM。

重要说明：
  - 组合美元盈亏均以 2024-12-31 = $25,000,000 锚定，回溯 2023–2024 历史；
    这是回溯风险估计，不是 2023 年真实持仓。
  - 交易成本忽略（毛收益口径），报告中需注明。
  - ES=F 是连续期货代理，换月与可成交价未核验。

输入：clean_data.csv, market_cap.csv, betas.csv
输出：hedge_ratios.csv, L3_estimation_daily_dollar_pnl.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

# =================== 配置 ===================
DATA_DIR = Path(r"D:\MTH202\L1+L2\数据集")
FUTURES_DIR = Path(r"D:\MTH202\期货数据")
STOCKS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "JPM", "V", "MA",
    "XOM", "CVX", "PG", "KO", "WMT", "MCD", "JNJ", "UNH", "CAT", "HON",
]
EST_START = "2023-01-01"
EST_END = "2024-12-31"
BASE_DATE = "2024-12-31"          # 建仓/估值基准日（对冲起始 2025-01-01 前一交易日）
ANCHOR_BASE = "2022-12-30"        # 估计期前一交易日（构造首日价格变动用）
PORTFOLIO_VALUE = 25_000_000.0
ES_MULTIPLIER = 50.0


def load_csv(folder: Path, name: str, **kwargs) -> pd.DataFrame:
    fp = folder / name
    if not fp.is_file():
        raise FileNotFoundError(f"找不到 {fp}；请修改 DATA_DIR")
    return pd.read_csv(fp, **kwargs)


def round_contracts(n: float) -> int:
    """四舍五入到最近整数，0.5 远离零。"""
    return int(np.sign(n) * np.floor(abs(n) + 0.5))


def buy_and_hold_dollar_pnl(hist_prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """固定股数（2024-12-31 锁定）的每日美元盈亏。"""
    base_price = hist_prices.loc[BASE_DATE, weights.index]  # 只取股票列
    shares = PORTFOLIO_VALUE * weights / base_price
    value = hist_prices[weights.index].dot(shares)
    return value.diff().dropna()


def rebalanced_dollar_pnl(hist_prices: pd.DataFrame, weights: pd.Series, freq: str) -> pd.Series:
    """按 freq('monthly'/'quarterly') 再平衡到目标权重的每日美元盈亏。

    季度内/月内买入持有，期末收盘后权重重置回目标权重。
    组合价值以 2024-12-31 = $25,000,000 锚定。
    """
    simple = hist_prices[weights.index].pct_change().dropna()
    dates = simple.index
    w = weights.copy()
    R_list = []
    for i, date in enumerate(dates):
        r_i = simple.loc[date]
        R_t = float((w * r_i).sum())
        R_list.append(R_t)
        w = w * (1.0 + r_i) / (1.0 + R_t)          # 漂移
        if i + 1 < len(dates):
            nxt = dates[i + 1]
            reset = (freq == "monthly" and nxt.month != date.month) or \
                    (freq == "quarterly" and nxt.quarter != date.quarter)
            if reset:
                w = weights.copy()                  # 期末收盘后重置
    R = pd.Series(R_list, index=dates)
    v_units = (1.0 + R).cumprod()
    v_units = pd.concat([pd.Series([1.0], index=[hist_prices.index[0]]), v_units])
    scale = PORTFOLIO_VALUE / v_units.iloc[-1]      # 锚定 2024-12-31 = $25m
    return (v_units * scale).diff().dropna()


def mv_contracts(dP: pd.Series, dF: pd.Series) -> float:
    """最小方差手数 = Cov(dP, dF)/Var(dF)。"""
    return float(dP.cov(dF) / dF.var())


def main() -> pd.DataFrame:
    prices = load_csv(DATA_DIR, "clean_data.csv", index_col="Date", parse_dates=True).sort_index()
    market_cap = load_csv(DATA_DIR, "market_cap.csv", index_col="Ticker")
    betas = load_csv(DATA_DIR, "betas.csv", index_col="Ticker")

    # ---- 校验 ----
    for name, df in [("clean_data", prices), ("market_cap", market_cap), ("betas", betas)]:
        if not df.index.is_unique:
            raise ValueError(f"{name} 索引重复")
    if set(STOCKS) - set(prices.columns) or "ES_F" not in prices.columns:
        raise ValueError("clean_data.csv 缺少股票或 ES_F 列")
    if set(STOCKS) - set(market_cap.index) or "权重%" not in market_cap.columns:
        raise ValueError("market_cap.csv 缺少股票或 权重% 列")
    if set(STOCKS) - set(betas.index) or "beta" not in betas.columns:
        raise ValueError("betas.csv 缺少股票或 beta 列")
    if pd.Timestamp(BASE_DATE) not in prices.index or pd.Timestamp(ANCHOR_BASE) not in prices.index:
        raise ValueError(f"缺少 {BASE_DATE} 或 {ANCHOR_BASE}")

    hist = prices.loc[ANCHOR_BASE:BASE_DATE][STOCKS + ["ES_F"]].astype(float)
    if not np.isfinite(hist.to_numpy()).all() or (hist <= 0).any().any():
        raise ValueError("估计期价格存在非法值")

    # ---- 权重 & beta ----
    w_equal = pd.Series(1.0 / len(STOCKS), index=STOCKS)
    w_mcap = (market_cap.loc[STOCKS, "权重%"].astype(float) / 100.0)
    for name, w in [("Equal", w_equal), ("Market-cap", w_mcap)]:
        if not np.isclose(w.sum(), 1.0, atol=1e-6) or (w < 0).any():
            raise ValueError(f"{name} 权重异常")
    stock_betas = betas.loc[STOCKS, "beta"].astype(float)

    # ---- 期货一张合约美元盈亏（换月调整后的真实合约序列）----
    roll = load_csv(FUTURES_DIR, "ES_roll_adjusted.csv", index_col=0, parse_dates=True)
    roll_points = roll["ES_roll_adj_dF"].astype(float)          # 每日点数变动（换月调整后）
    stock_days = hist[STOCKS].diff().dropna().index              # 股票交易日历（估计期基准）
    dF_points = roll_points.reindex(stock_days)                  # 对齐股票日历，剔除期货特有假期 stale 行
    if dF_points.isna().any():
        raise ValueError("期货序列与股票日历存在缺口，请检查")
    dF = ES_MULTIPLIER * dF_points

    # CAPM 的 f0：用 2024-12-31 实际持有的那张合约（ESH25）收盘价
    esh25 = pd.read_excel(FUTURES_DIR / "ESH25.csv.xlsx", header=6)
    esh25 = esh25.rename(columns={esh25.columns[0]: "Date"})
    esh25["Date"] = pd.to_datetime(esh25["Date"])
    f0 = float(esh25.set_index("Date")["PX_LAST"].loc[BASE_DATE])
    vf = f0 * ES_MULTIPLIER

    # ---- 各组合美元盈亏 ----
    dP = {
        ("Equal-weight", "Buy-and-hold"): buy_and_hold_dollar_pnl(hist, w_equal),
        ("Equal-weight", "Monthly rebalance"): rebalanced_dollar_pnl(hist, w_equal, "monthly"),
        ("Equal-weight", "Quarterly rebalance"): rebalanced_dollar_pnl(hist, w_equal, "quarterly"),
        ("Market-cap-weight", "Buy-and-hold"): buy_and_hold_dollar_pnl(hist, w_mcap),
    }

    # ---- 计算并汇总 ----
    rows = []
    daily_pnl = pd.DataFrame(index=dF.index)
    daily_pnl.index.name = "Date"
    daily_pnl["ES_Change_Points_Roll_Adjusted"] = dF_points
    daily_pnl["ES_One_Contract_Dollar_Change"] = dF

    for (pf, hold), pnl in dP.items():
        if not pnl.index.equals(dF.index):
            raise ValueError(f"{pf}/{hold} 日期未对齐")
        daily_pnl[f"{pf}_{hold}_Portfolio_Dollar_Change"] = pnl
        n = mv_contracts(pnl, dF)
        h_value = n * vf / PORTFOLIO_VALUE
        rows.append({
            "Portfolio": pf, "Holding_Assumption": hold, "Method": "Minimum-Variance",
            "Hedge_Ratio_Value_Basis": h_value, "Contracts_Unrounded": n,
            "Contracts_Signed": round_contracts(n),
            "Direction": "SHORT" if round_contracts(n) > 0 else "NONE",
        })

    for pf, w in [("Equal-weight", w_equal), ("Market-cap-weight", w_mcap)]:
        beta_p = float(w.dot(stock_betas))
        n = beta_p * PORTFOLIO_VALUE / vf
        rows.append({
            "Portfolio": pf, "Holding_Assumption": "N/A (target weights)",
            "Method": "CAPM", "Hedge_Ratio_Value_Basis": beta_p,
            "Contracts_Unrounded": n, "Contracts_Signed": round_contracts(n),
            "Direction": "SHORT" if round_contracts(n) > 0 else "NONE",
        })

    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / "hedge_ratios.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    daily_pnl.to_csv(DATA_DIR / "L3_estimation_daily_dollar_pnl.csv", encoding="utf-8-sig", float_format="%.12g")

    print("L3 对冲比率（2024-12-31 锁定，2023–2024 回溯美元盈亏估计）")
    print(result[["Portfolio", "Holding_Assumption", "Method", "Hedge_Ratio_Value_Basis",
                  "Contracts_Unrounded", "Contracts_Signed", "Direction"]].to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\n估计期: {len(dF)} 日 | F0(2024-12-31 ESH25)={f0:.2f} | VF=${vf:,.2f}")
    print(f"已保存: {DATA_DIR/'hedge_ratios.csv'} 和 {DATA_DIR/'L3_estimation_daily_dollar_pnl.csv'}")
    print("注意: 期货用 Bloomberg 具体合约按'到期前第5个有效日度报价日'换月构造; 交易成本未核验。")
    return result


if __name__ == "__main__":
    main()
