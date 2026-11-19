# -*- coding: utf-8 -*-
"""L4 poster figures #2: daily P&L distributions (equal-weight and market-cap-weight).

海报图 2：等权 / 市值加权组合的日度盈亏分布直方图（对冲前 vs 最小方差 vs CAPM）。
竖版、300 dpi，并把三条分布的日度标准差写进图例，直观展示"对冲把分布收窄"。
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
LABELS = {
    "Equal-weight": ("Equal-weight Portfolio", "daily P&L distribution - equal weight"),
    "Market-cap-weight": ("Market-cap-weighted Portfolio", "daily P&L distribution - market cap weight"),
}


def main():
    df = pd.read_csv(DATA / "L4-hedge backtest-daily P&L.csv", index_col="Date", parse_dates=True)

    for pf, (title, stub) in LABELS.items():
        # 美元盈亏换算成千美元，便于分箱显示
        data_k = {s: df[f"{pf}_{s}"].to_numpy(float) / 1000.0 for s in ("Unhedged", "MV", "CAPM")}
        all_values = np.concatenate(list(data_k.values()))

        bin_width = 50.0   # 每 5 万美元一个箱子
        bins = np.arange(np.floor(all_values.min() / bin_width) * bin_width,
                         np.ceil(all_values.max() / bin_width) * bin_width + 1.01 * bin_width,
                         bin_width)
        sd = {s: np.std(v, ddof=1) for s, v in data_k.items()}   # 日度标准差（千美元）

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                             "pdf.fonttype": 42, "ps.fonttype": 42})
        fig, ax = plt.subplots(figsize=(7.3, 8.0))
        # 不对冲用填充面积 + 轮廓；MV/CAPM 用轮廓线（CAPM 虚线），三层叠加便于对比
        ax.hist(data_k["Unhedged"], bins=bins, color=COLORS["Unhedged"], alpha=0.19, histtype="stepfilled")
        ax.hist(data_k["Unhedged"], bins=bins, color=COLORS["Unhedged"], lw=1.8, histtype="step",
                label=f"Unhedged (SD {sd['Unhedged']:.0f}k)")
        ax.hist(data_k["MV"], bins=bins, color=COLORS["MV"], lw=2.2, histtype="step",
                label=f"MV ({sd['MV']:.0f}k)")
        _, _, arts = ax.hist(data_k["CAPM"], bins=bins, color=COLORS["CAPM"], lw=1.9, histtype="step",
                             label=f"CAPM ({sd['CAPM']:.0f}k)")
        for a in arts:
            a.set_linestyle("--")
        ax.axvline(0, lw=0.9, color="#353535", ls=":")   # 零盈亏参考线
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#888888", alpha=0.26, ls="--")
        ax.set_ylabel("Number of trading days", fontsize=11)
        ax.set_xlabel("Daily P&L (USD thousands)", fontsize=11)
        ax.set_title(f"{title}\nDaily P&L Distribution (2025)",
                     loc="left", fontsize=14, fontweight="bold", pad=11, linespacing=1.35)
        ax.legend(loc="upper right", frameon=False, fontsize=10.3)
        fig.text(0.125, 0.034, f"{len(df)} trading days  |  Bin width: USD 50,000  |  gross P&L",
                 fontsize=9.0, color="#555555")
        fig.subplots_adjust(left=0.125, right=0.97, top=0.855, bottom=0.14)

        fig.savefig(FIG / f"{stub}.png", dpi=300, facecolor="white")
        fig.savefig(FIG / f"{stub}.pdf", facecolor="white")
        plt.close(fig)
        print(f"[saved] {stub}.png / .pdf   (SD: Unhedged={sd['Unhedged']:.0f}k, "
              f"MV={sd['MV']:.0f}k, CAPM={sd['CAPM']:.0f}k)")


if __name__ == "__main__":
    main()
