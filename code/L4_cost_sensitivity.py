# -*- coding: utf-8 -*-
"""L4 - transaction-cost sensitivity (four per-side cost tiers).

在四档每边成本假设下，算全年总成本和净收益率，证明交易成本对结论影响很小。
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"

# 全年 9 个单边交易 = 1 次建仓 + 4 次换月 x 2 腿
N_SIDES = 9
SCENARIOS = {
    "zero cost": 0.00,
    "low ($3/side)": 3.00,
    "medium ($9.25/side)": 9.25,
    "high ($15.5/side)": 15.50,
}
PORTFOLIO_VALUE = 25_000_000.0


def main():
    perf = pd.read_csv(DATA / "L4-hedge backtest-performance summary.csv")

    rows = []
    for _, r in perf.iterrows():
        pf, strat, N, gross = r["Portfolio"], r["Strategy"], r["Contracts_Short"], r["Total_PnL_USD"]
        gross_ret = r["Total_Return"]
        for scen, c in SCENARIOS.items():
            # 总成本 = 单边次数 x 合约数 x 每边成本
            cost = N_SIDES * N * c if strat != "Unhedged" else 0.0
            rows.append({
                "Portfolio": pf, "Strategy": strat, "Scenario": scen,
                "Gross_PnL": gross, "Cost": cost, "Net_PnL": gross - cost,
                "Gross_Return": gross_ret, "Net_Return": (gross - cost) / PORTFOLIO_VALUE,
                "Cost_Pct_Portfolio": cost / PORTFOLIO_VALUE,
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "L4-cost sensitivity-cost scenarios.csv",
              index=False, encoding="utf-8-sig", float_format="%.12g")
    print("[saved] L4-cost sensitivity-cost scenarios.csv")

    hedged = df[df["Strategy"] != "Unhedged"]
    piv_cost = hedged.pivot_table(index=["Portfolio", "Strategy"], columns="Scenario", values="Cost")
    piv_net = hedged.pivot_table(index=["Portfolio", "Strategy"], columns="Scenario", values="Net_Return")
    print("\nTotal transaction cost (USD):")
    print(piv_cost.to_string(formatters={c: "{:,.0f}".format for c in piv_cost.columns}))
    print("\nNet return after cost:")
    print(piv_net.to_string(formatters={c: "{:.4%}".format for c in piv_net.columns}))


if __name__ == "__main__":
    main()
