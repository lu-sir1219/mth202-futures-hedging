# -*- coding: utf-8 -*-
"""L4 poster figures #4: performance table (six strategies, 2025).

海报图 4：六策略绩效汇总表（等权/市值 x 不对冲/MV/CAPM），渲染成一张 PNG。
数据来自 L4-hedge backtest-performance summary.csv。
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"
FIG = ROOT / "Source_Diagram_034" / "poster"
FIG.mkdir(parents=True, exist_ok=True)

PORTFOLIO_VALUE = 25_000_000.0   # 名义本金，用于把美元盈亏换算成收益率（%）


def main():
    perf = pd.read_csv(DATA / "L4-hedge backtest-performance summary.csv")

    # 整理成表格显示的六行：组合 / 策略 / 年化收益 / 方差降低 / 最大回撤
    header = ["Portfolio", "Strategy", "Annual return", "Variance reduction", "Max drawdown"]
    cells = []
    for _, r in perf.iterrows():
        vr = f"{r['Variance_Reduction']*100:.2f}%" if r["Strategy"] != "Unhedged" else "\u2014"
        cells.append([
            "Equal-weight" if r["Portfolio"] == "Equal-weight" else "Market-cap",
            r["Strategy"],
            f"{r['Total_Return']*100:+.2f}%",
            vr,
            f"{r['Max_Drawdown']*100:.2f}%",
        ])

    fig, ax = plt.subplots(figsize=(10.9, 3.55))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 2.0)
    # 表头深蓝底白字，数据行隔行浅灰
    for j in range(len(header)):
        tbl[(0, j)].set_facecolor("#1f3864")
        tbl[(0, j)].set_text_props(color="white", weight="bold")
    for i in range(1, len(cells) + 1):
        tbl[(i, 0)].set_text_props(weight="bold")
        for j in range(len(header)):
            tbl[(i, j)].set_facecolor("#eef2f7" if i % 2 else "white")
    fig.suptitle("2025 Out-of-Sample Hedge Performance", fontsize=16, fontweight="bold", y=0.97)
    fig.text(0.5, 0.02,
             "Source: authors' calculations from the 2025 L4 backtest "
             "(Bloomberg ES futures; Yahoo Finance equities). Six strategies, $25 million base, "
             "before transaction costs.",
             ha="center", fontsize=9.5, color="#555555")
    fig.savefig(FIG / "performance table (2025).png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[saved] performance table (2025).png")
    print(perf[["Portfolio", "Strategy", "Total_Return", "Variance_Reduction", "Max_Drawdown"]].to_string(index=False))


if __name__ == "__main__":
    main()
