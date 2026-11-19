# -*- coding: utf-8 -*-
"""L4 poster figures #3: market-episode P&L bars (bearish / bullish / flat).

海报图 3：三个连续市场区间（熊/牛/横盘）的区间累计盈亏柱状图。
区间按 SPY 走势、独立于对冲结果选取，首尾两个交易日都计入（与报告 Table 4 一致）。
本脚本同时复现海报用的描述统计 CSV（5 只代表性股票），并核对区间盈亏与报告一致。
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

COLORS = {"Unhedged": "#1f77b4", "MV": "#2ca02c", "CAPM": "#d62728"}
STRATEGIES = ["Unhedged", "MV", "CAPM"]
STOCKS_FOR_TABLE = ["NVDA", "JPM", "XOM", "KO", "SPY"]   # 海报统计表的 5 只代表股票

# 三个连续市场区间：(标签, 首日, 末日, SPY 区间涨跌幅%)
EPISODES = [
    ("Bearish", "2025-02-19", "2025-04-08", -18.76),
    ("Bullish", "2025-04-09", "2025-09-30", 34.90),
    ("Flat", "2025-11-03", "2025-12-31", 0.07),
]

# 报告 Table 4 的四舍五入区间盈亏（百万美元），用于核对本脚本复算结果
REPORT_PNL_M = {
    ("Bearish", "Equal-weight"): (-3.55, -0.01, 0.34),
    ("Bearish", "Market-cap-weight"): (-5.31, -0.13, 0.76),
    ("Bullish", "Equal-weight"): (5.46, 0.64, 0.16),
    ("Bullish", "Market-cap-weight"): (8.96, 1.89, 0.68),
    ("Flat", "Equal-weight"): (0.40, 0.52, 0.54),
    ("Flat", "Market-cap-weight"): (-0.20, -0.02, 0.01),
}

# 最终版 L4 的六条策略年度总盈亏（美元），用于防呆：避免误用旧版 Yahoo 数据
FINAL_ANNUAL_PNL = {
    "Equal-weight": [4_484_746.3319, 2_281_246.3319, 2_060_896.3319],
    "Market-cap-weight": [5_666_123.1934, 2_434_323.1934, 1_883_448.1934],
}


def main():
    # ---- 1) 描述统计（5 只代表股票，2023-2024，502 个交易日） ----
    ret = pd.read_csv(DATA / "L1-data cleaning-daily returns.csv", index_col=0, parse_dates=True)
    est = ret.loc["2023-01-01":"2024-12-31", STOCKS_FOR_TABLE]
    if len(est) != 502 or est.isna().any().any():
        raise ValueError(f"estimation window should be 502 complete rows, got {len(est)}")
    rows = []
    for t in STOCKS_FOR_TABLE:
        v = pd.to_numeric(est[t], errors="raise")
        rows.append({"Asset": t, "Mean_Daily_Return_Pct": v.mean() * 100,
                     "Daily_SD_Pct": v.std(ddof=1) * 100,
                     "Annualized_Volatility_Pct": v.std(ddof=1) * np.sqrt(252) * 100,
                     "Observations": len(v)})
    stats = pd.DataFrame(rows)
    stats.to_csv(DATA / "L1-data cleaning-descriptive statistics (5 of 18, poster).csv",
                 index=False, float_format="%.6f", encoding="utf-8-sig")
    print("[saved] L1-data cleaning-descriptive statistics (5 of 18, poster).csv")

    # ---- 2) 区间盈亏复算 + 与报告核对 ----
    daily = pd.read_csv(DATA / "L4-hedge backtest-daily P&L.csv", index_col="Date", parse_dates=True)
    # 防呆：年度总和必须等于最终版，否则说明用了旧数据
    for pf, exp in FINAL_ANNUAL_PNL.items():
        for s, e in zip(STRATEGIES, exp):
            actual = daily[f"{pf}_{s}"].sum()
            if abs(actual - e) > 5.0:
                raise ValueError(f"{pf} {s} annual P&L = {actual:,.2f}, expected {e:,.2f} "
                                 "(old L4 data?)")

    records = []
    for label, first, last, spy_pct in EPISODES:
        part = daily.loc[first:last]   # 首尾都包含
        for pf in FINAL_ANNUAL_PNL:
            vals = [float(part[f"{pf}_{s}"].sum()) / 1e6 for s in STRATEGIES]
            for s, actual_m, expected_m in zip(STRATEGIES, vals, REPORT_PNL_M[(label, pf)]):
                if abs(actual_m - expected_m) > 0.011:
                    raise ValueError(f"{label} {pf} {s}: recalc={actual_m:+.4f}M vs report={expected_m:+.2f}M")
                records.append({"Episode": label, "Portfolio": pf, "Strategy": s,
                                "Days": len(part), "Period_PnL_USD_Millions": actual_m})
    epi = pd.DataFrame(records)
    epi.to_csv(DATA / "L4-hedge backtest-market episodes (2025).csv",
               index=False, float_format="%.6f", encoding="utf-8-sig")
    print("[saved] L4-hedge backtest-market episodes (2025).csv")
    print(epi.pivot_table(index=["Episode", "Portfolio"], columns="Strategy",
                          values="Period_PnL_USD_Millions").round(2).to_string())

    # ---- 3) 两张柱状图（等权 / 市值） ----
    labels = ["Bearish\n19 Feb\u20138 Apr", "Bullish\n9 Apr\u201330 Sep", "Flat\n3 Nov\u201331 Dec"]
    for pf, title, stub in [("Equal-weight", "Equal-weight Portfolio", "market episode P&L - equal weight"),
                            ("Market-cap-weight", "Market-cap-weighted Portfolio", "market episode P&L - market cap weight")]:
        vals = {s: [epi[(epi.Episode == e) & (epi.Portfolio == pf) & (epi.Strategy == s)]
                    .iloc[0]["Period_PnL_USD_Millions"] for e, *_ in EPISODES] for s in STRATEGIES}
        fig, ax = plt.subplots(figsize=(7.2, 7.3))
        x = np.arange(len(EPISODES))
        width = 0.25
        for j, s in enumerate(STRATEGIES):
            ax.bar(x + (j - 1) * width, vals[s], width, color=COLORS[s], label=s)
        ax.set_xticks(x, labels, fontsize=11.5)
        ax.set_ylabel("Cumulative P&L (USD millions)", fontsize=11.5)
        ax.tick_params(axis="y", labelsize=11)
        ax.axhline(0, color="#272727", lw=0.95)
        ax.grid(axis="y", alpha=0.22, ls="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Hedging Performance by Market Episode (2025)\n{title}",
                     loc="left", weight="bold", fontsize=13, pad=13)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=10)
        fig.text(0.5, 0.025, "Selected consecutive episodes | Gross portfolio P&L | Different episode lengths",
                 ha="center", fontsize=9, color="#545454")
        fig.tight_layout(rect=(0, 0.06, 1, 0.99))
        fig.savefig(FIG / f"{stub}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(FIG / f"{stub}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[saved] {stub}.png / .pdf")


if __name__ == "__main__":
    main()
