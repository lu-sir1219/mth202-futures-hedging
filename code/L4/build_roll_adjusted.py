# -*- coding: utf-8 -*-
"""构造换月调整后的 ES 期货序列（ES_roll_adjusted.csv）。

输入：本目录下的 13 张具体合约 (ESH23 ... ESZ25, ESH26)，Bloomberg 导出的 xlsx。
输出：ES_roll_adjusted.csv（每日点数变动，2022-12-02 ~ 2025-12-31）。

方法：
  - 每张合约取其"当主力"期间的每日收盘价 PX_LAST；
  - 换月规则：到期前第 5 个 Bloomberg 有效日度报价日换月（最后一张 ESH26 持有到评估期末 2025-12-31）；
    注：此为"有效日终报价日期"口径，与独立 CME 交易时段日历在 2024/2025 年 6 月各差一天。
  - 每日变动 = 当天实际持有的那张合约"自己"的价格变动（同一合约今日 − 昨日），
    因此不含跨合约换月的假跳变。

注意：
  - 部分合约含休市日 stale 行（如 2023-04-07 耶稣受难日，PX_LAST 沿用前一日）；
    下游 L3/L4 脚本会按股票交易日历 reindex 剔除这些行。
"""
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parent

# 13 张合约到期日（第三个周五）
EXPIRIES = {
    "ESH23": "2023-03-17", "ESM23": "2023-06-16", "ESU23": "2023-09-15", "ESZ23": "2023-12-15",
    "ESH24": "2024-03-15", "ESM24": "2024-06-21", "ESU24": "2024-09-20", "ESZ24": "2024-12-20",
    "ESH25": "2025-03-21", "ESM25": "2025-06-20", "ESU25": "2025-09-19", "ESZ25": "2025-12-19",
    "ESH26": "2026-03-20",
}
EVAL_END = pd.Timestamp("2025-12-31")
ROLL_BEFORE = 5   # 到期前第 N 个 Bloomberg 有效日度报价日换月


def load_contract(ticker: str) -> pd.Series:
    """读取单张合约的 Date + PX_LAST，去休市日，按日期升序。"""
    fn = D / f"{ticker}.csv.xlsx"
    if not fn.exists():
        fn = D / f"{ticker}.xlsx"
    df = pd.read_excel(fn, header=6)          # Bloomberg 导出：前 6 行是元信息
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "PX_LAST"]).sort_values("Date")
    return df.set_index("Date")["PX_LAST"].astype(float)


def main():
    contracts = {t: load_contract(t) for t in EXPIRIES}
    all_dates = sorted(set().union(*[set(c.index) for c in contracts.values()]))

    # 每张合约的换月日：到期前 ROLL_BEFORE 个交易日；最后一张持有到评估期末
    def roll_date(expiry):
        return [d for d in all_dates if d < pd.Timestamp(expiry)][-ROLL_BEFORE]

    order = list(EXPIRIES.keys())
    roll_out = {}
    for i, t in enumerate(order):
        roll_out[t] = roll_date(EXPIRIES[t]) if i < len(order) - 1 else EVAL_END

    # 逐日归属"持有合约"
    held = {}
    for i, t in enumerate(order):
        roll_in = roll_out[order[i - 1]] if i > 0 else pd.Timestamp("2022-12-01")
        for d in all_dates:
            if roll_in < d <= roll_out[t]:
                held[d] = t

    # 每日点数变动 = 持有合约自己的今日−昨日（同一合约内，无跨合约跳变）
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
    series.to_csv(D / "ES_roll_adjusted.csv")

    print(f"已生成 ES_roll_adjusted.csv：{series.index.min().date()} ~ "
          f"{series.index.max().date()}，{len(series)} 个交易日")


if __name__ == "__main__":
    main()
