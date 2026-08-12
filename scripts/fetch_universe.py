"""Fetch a WIDE + DEEP free cross-asset universe via yfinance for the powered-null battery — the only lever
that raises statistical power (t ~= Sharpe * sqrt(years); detecting a 0.5 Sharpe needs ~25y, so depth is
everything, and breadth cleans the factor via IR = IC*sqrt(breadth)). Writes universe/<SYM>.json =
{class, data: {date: [open,high,low,close,volume]}} + universe/manifest.json. Idempotent (overwrites)."""
import json
import os

import yfinance as yf

UNIVERSE = {
    "equity": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "WMT", "XOM", "JNJ", "PG", "KO", "PEP",
               "MRK", "PFE", "CVX", "HD", "MCD", "CSCO", "INTC", "IBM", "ORCL", "CAT", "BA", "GE", "MMM", "DIS",
               "VZ", "T", "WFC", "BAC", "C", "GS", "MS", "AXP", "UNH", "ABT", "TMO", "COST", "LOW", "HON", "UPS",
               "NKE", "SBUX", "TXN", "QCOM", "AMD", "ADBE", "CRM", "NFLX"],
    "crypto": ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD", "DOGE-USD", "DOT-USD",
               "LINK-USD", "XLM-USD", "TRX-USD", "ETC-USD", "XMR-USD", "EOS-USD", "BNB-USD", "SOL-USD",
               "AVAX-USD", "ATOM-USD", "ALGO-USD", "XTZ-USD"],
    "commodity": ["GLD", "SLV", "USO", "UNG", "DBC", "DBA", "CORN", "WEAT", "CPER", "PPLT", "PALL", "UGA",
                  "GSG", "DJP", "PDBC", "SOYB", "CANE", "JO"],
    "bond": ["TLT", "IEF", "SHY", "AGG", "BND", "LQD", "HYG", "JNK", "TIP", "MBB", "EMB", "GOVT", "VCIT",
             "VCSH", "MUB", "BWX"],
    "fx": ["UUP", "FXE", "FXY", "FXB", "FXA", "FXF", "FXC", "CEW"],
}
OUT = "universe"


def main():
    os.makedirs(OUT, exist_ok=True)
    sym_class = {s: c for c, syms in UNIVERSE.items() for s in syms}
    tickers = list(sym_class)
    print(f"fetching {len(tickers)} tickers via yfinance (period=max, daily) ...")
    df = yf.download(tickers, period="max", interval="1d", auto_adjust=False, progress=False,
                     group_by="ticker", threads=True)
    manifest = {}
    for sym in tickers:
        try:
            sub = df[sym]
        except (KeyError, TypeError):
            print(f"  {sym}: MISSING"); continue
        sub = sub.dropna(subset=["Close"])
        if len(sub) < 250:
            print(f"  {sym}: too short ({len(sub)})"); continue
        data = {}
        for ts, row in sub.iterrows():
            c = row.get("Close")
            if c != c:
                continue
            def g(k):
                v = row.get(k)
                return round(float(v), 6) if v == v else None
            data[ts.strftime("%Y-%m-%d")] = [g("Open"), g("High"), g("Low"), round(float(c), 6), g("Volume")]
        keys = sorted(data)
        rec = {"class": sym_class[sym], "data": data}
        with open(f"{OUT}/{sym}.json", "w") as fh:
            json.dump(rec, fh)
        years = (len(keys)) / 252.0
        manifest[sym] = {"class": sym_class[sym], "n": len(keys), "from": keys[0], "to": keys[-1], "years": round(years, 1)}
        print(f"  {sym:10s} {sym_class[sym]:10s} {len(keys):>6d} days {keys[0]}..{keys[-1]} ({years:.1f}y)")
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    by_cls = {}
    for s, m in manifest.items():
        by_cls.setdefault(m["class"], []).append(m["years"])
    print("\n=== per-class breadth x median years ===")
    for c, ys in sorted(by_cls.items()):
        print(f"  {c:10s}: {len(ys):>3d} symbols, median {sorted(ys)[len(ys)//2]:.1f}y, max {max(ys):.1f}y")
    print(f"\nwrote {len(manifest)} symbols -> {OUT}/")


if __name__ == "__main__":
    main()
