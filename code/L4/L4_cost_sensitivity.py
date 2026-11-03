# -*- coding: utf-8 -*-
"""L4 交易成本敏感性分析。

成本来源（做空 ES 期货）：
  1) 建仓：卖出 N 张（1 次）
  2) 换月：2025 年有 4 次换月（3/6/9/12月），每次 = 平旧(买) + 开新(卖) = 2 次
  => 全年共 1 + 4*2 = 9 个单边交易，每个单边 = N 张合约。

成本情景（每张合约、每个单边）：
  - 零成本  $0.00   基准（毛收益）
  - 低      $3.00   仅佣金，假设无滑点
  - 中      $9.25   佣金$3 + 半价差 0.125点×$50=$6.25
  - 高      $15.50  佣金$3 + 整价差 0.25点×$50=$12.50
"""
import pandas as pd
from pathlib import Path

OUT = Path(r"D:\MTH202\L4")
perf = pd.read_csv(OUT / "L4_performance_summary.csv")

N_SIDES = 9  # 1 建仓 + 4 换月×2
SCENARIOS = {
    "零成本": 0.00,
    "低($3/边)": 3.00,
    "中($9.25/边)": 9.25,
    "高($15.5/边)": 15.50,
}
PORTFOLIO_VALUE = 25_000_000.0

rows = []
for _, r in perf.iterrows():
    pf, strat, N, gross = r["Portfolio"], r["Strategy"], r["Contracts_Short"], r["Total_PnL_USD"]
    gross_ret = r["Total_Return"]
    for scen, c in SCENARIOS.items():
        cost = N_SIDES * N * c if strat != "Unhedged" else 0.0
        net = gross - cost
        net_ret = net / PORTFOLIO_VALUE
        rows.append({
            "Portfolio": pf, "Strategy": strat, "Scenario": scen,
            "Gross_PnL": gross, "Cost": cost, "Net_PnL": net,
            "Gross_Return": gross_ret, "Net_Return": net_ret,
            "Cost_Pct_Portfolio": cost / PORTFOLIO_VALUE,
        })

df = pd.DataFrame(rows)
df.to_csv(OUT / "L4_cost_sensitivity.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

# 打印：只显示对冲策略（不对冲无成本）
hedged = df[df["Strategy"] != "Unhedged"]
piv_cost = hedged.pivot_table(index=["Portfolio", "Strategy"], columns="Scenario", values="Cost")
piv_net = hedged.pivot_table(index=["Portfolio", "Strategy"], columns="Scenario", values="Net_Return")

print("=== 交易成本（美元） ===")
print(piv_cost.to_string(formatters={c: "{:,.0f}".format for c in piv_cost.columns}))
print("\n=== 净收益率（扣除成本后） ===")
print(piv_net.to_string(formatters={c: "{:.4%}".format for c in piv_net.columns}))
print("\n=== 成本占组合价值比例（最高成本情景） ===")
print(hedged[hedged["Scenario"] == "高($15.5/边)"][["Portfolio", "Strategy", "Cost_Pct_Portfolio"]]
      .to_string(index=False, formatters={"Cost_Pct_Portfolio": "{:.5%}".format}))

print(f"\n已保存: {OUT/'L4_cost_sensitivity.csv'}")
