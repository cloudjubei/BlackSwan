"""Fetch single-commodity fund NAVs (daily close) via yfinance, for the carry-proxy test.

A commodity fund holds futures and mechanically earns/pays the roll each cycle, so r(fund) − r(front-continuous)
isolates the realized CARRY (roll yield) — a free proxy for the (M1-M2) basis where per-expiry term structure is
data-gated (metals/grains). Funds: CORN (corn 2010+), WEAT (wheat 2011+), CPER (copper 2011+), GLD/SLV (2006+,
near-zero carry). yfinance handles the Yahoo crumb/session/rate-limiting. Writes funds/<SYM>.json = {date: close}."""
import json
import os

import yfinance as yf

SYMBOLS = ["CORN", "WEAT", "CPER", "GLD", "SLV"]
OUT = "funds"


def main():
    os.makedirs(OUT, exist_ok=True)
    for sym in SYMBOLS:
        df = yf.Ticker(sym).history(period="max", interval="1d", auto_adjust=False)
        if df.empty:
            print(f"  {sym}: EMPTY")
            continue
        series = {}
        for ts, row in df.iterrows():
            c = row.get("Close")
            if c is None or c != c:
                continue
            series[ts.strftime("%Y-%m-%d")] = round(float(c), 4)
        with open(f"{OUT}/{sym}.json", "w") as fh:
            json.dump(series, fh)
        keys = sorted(series)
        print(f"  {sym}: {len(series)} days {keys[0]}..{keys[-1]} -> {OUT}/{sym}.json")


if __name__ == "__main__":
    main()
