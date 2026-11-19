# -*- coding: utf-8 -*-
"""L3 - roll adjustment.

把 13 张 Bloomberg 季度 ES 合约按换月规则拼成一条每日点数变动序列：到期前第 5 个
有效报价日换月，只算同一张合约自己的涨跌，剔除跨合约换月的假跳变。
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Source_Data_034"
BB = DATA / "bloomberg_futures contract"

# 13 张季度合约的到期日（每个季度的第三个周五）
EXPIRIES = {
    "ESH23": "2023-03-17", "ESM23": "2023-06-16", "ESU23": "2023-09-15", "ESZ23": "2023-12-15",
    "ESH24": "2024-03-15", "ESM24": "2024-06-21", "ESU24": "2024-09-20", "ESZ24": "2024-12-20",
    "ESH25": "2025-03-21", "ESM25": "2025-06-20", "ESU25": "2025-09-19", "ESZ25": "2025-12-19",
    "ESH26": "2026-03-20",
}
EVAL_END = pd.Timestamp("2025-12-31")
ROLL_BEFORE = 5   # 到期前第 N 个有效报价日换月


def load_contract(ticker):
    fn = BB / f"L0-data collection-{ticker} futures contract.xlsx"
    df = pd.read_excel(fn, header=6)   # Bloomberg 导出，前 6 行是元信息
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "PX_LAST"]).sort_values("Date")
    return df.set_index("Date")["PX_LAST"].astype(float)


def main():
    contracts = {t: load_contract(t) for t in EXPIRIES}
    for t, s in contracts.items():
        print(f"[read] {t}: {len(s)} rows  {s.index.min().date()} -> {s.index.max().date()}")
    all_dates = sorted(set().union(*[set(c.index) for c in contracts.values()]))

    # 换月日 = 到期日前第 5 个有效报价日
    def roll_date(expiry):
        return [d for d in all_dates if d < pd.Timestamp(expiry)][-ROLL_BEFORE]

    order = list(EXPIRIES.keys())
    roll_out = {t: (roll_date(EXPIRIES[t]) if i < len(order) - 1 else EVAL_END)
                for i, t in enumerate(order)}

    # 逐日归属到当天实际持有的那张合约
    held = {}
    for i, t in enumerate(order):
        roll_in = roll_out[order[i - 1]] if i > 0 else pd.Timestamp("2022-12-01")
        for d in all_dates:
            if roll_in < d <= roll_out[t]:
                held[d] = t

    # 每日点数变动 = 同一张合约自己今日减昨日，不含跨合约换月的假跳变
    dF = {}
    for d in all_dates:
        if d in held:
            c = held[d]
            idx = contracts[c].index
            if d in idx:
                j = idx.get_loc(d)
                if j > 0:
                    dF[d] = contracts[c].iloc[j] - contracts[c].iloc[j - 1]

    series = pd.Series(dF).sort_index()
    series.name = "ES_roll_adj_dF"
    series.to_csv(DATA / "L3-roll adjustment-roll adjusted futures series.csv", index_label="Date")
    print(f"[saved] L3-roll adjustment-roll adjusted futures series.csv  {len(series)} daily changes")
    print(series.head(5).to_string())
    print("...")
    print(series.tail(5).to_string())


if __name__ == "__main__":
    main()
