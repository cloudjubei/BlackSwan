"""Broad free CEX-perp pull to POWER the positioning-continuation test the Hyperliquid verification demanded.
Binance USDS-M futures publishes, for FREE, ~500 USDT perpetuals with premium-index klines (the UNCLAMPED
mark-vs-index positioning-pressure series -- strictly better than clamped funding) and price klines back to
~2019-2020. That is many-fold the breadth x length of 12 HL coins x 3.2y, which is the only lever that powers a
cross-sectional factor (IR = IC*sqrt(breadth); t ~= Sharpe*sqrt(years)) so a positioning-continuation factor can
be either a real survivor or an EARNED powered-null against a 0.5 Sharpe rather than left inconclusive.

Writes cexperps/<SYM>.json = {sym, onboard, premium:{date:close}, price:{date:[o,h,l,c,v]}} + manifest.json.
Idempotent (overwrites)."""
import json
import os
import time
import urllib.request

FAPI = "https://fapi.binance.com"
OUT = "cexperps"
DAYMS = 86400000
LIMIT = 1500


def _get(path, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(FAPI + path, timeout=30) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return None


def universe():
    d = _get("/fapi/v1/exchangeInfo")
    out = []
    for s in d["symbols"]:
        if (s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"):
            out.append((s["symbol"], int(s.get("onboardDate", 0))))
    out.sort(key=lambda r: r[1])
    return out


def _klines(endpoint, sym, onboard, now_ms):
    rows = []
    start = onboard - onboard % DAYMS if onboard else 1568000000000
    while start < now_ms:
        page = _get(f"{endpoint}?symbol={sym}&interval=1d&startTime={start}&limit={LIMIT}")
        if not page:
            break
        rows.extend(page)
        if len(page) < LIMIT:
            break
        start = page[-1][0] + DAYMS
        time.sleep(0.05)
    return rows


def fetch_symbol(sym, onboard, now_ms):
    pk = _klines("/fapi/v1/premiumIndexKlines", sym, onboard, now_ms)
    premium = {time.strftime("%Y-%m-%d", time.gmtime(r[0] / 1000)): float(r[4]) for r in pk}
    ck = _klines("/fapi/v1/klines", sym, onboard, now_ms)
    price = {time.strftime("%Y-%m-%d", time.gmtime(r[0] / 1000)):
             [float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in ck}
    return premium, price


def main():
    os.makedirs(OUT, exist_ok=True)
    now_ms = int(time.time() * 1000)
    uni = universe()
    print(f"universe: {len(uni)} TRADING USDT perps")
    manifest = {}
    for i, (sym, onboard) in enumerate(uni):
        try:
            premium, price = fetch_symbol(sym, onboard, now_ms)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: FAIL {e}")
            continue
        dates = sorted(set(premium) & set(price))
        if len(dates) < 200:
            continue
        rec = {"sym": sym, "onboard": onboard, "premium": premium, "price": price}
        json.dump(rec, open(f"{OUT}/{sym}.json", "w"))
        manifest[sym] = {"n": len(dates), "from": dates[0], "to": dates[-1],
                         "years": round(len(dates) / 365.0, 2)}
        if i % 25 == 0:
            print(f"  [{i:>3}/{len(uni)}] {sym:14s} {len(dates):>5d} days {dates[0]}..{dates[-1]}")
        time.sleep(0.05)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    yrs = sorted(v["years"] for v in manifest.values())
    med = yrs[len(yrs) // 2] if yrs else 0
    print(f"wrote {len(manifest)} symbols -> {OUT}/  (median {med}y, "
          f"{sum(1 for v in manifest.values() if v['years']>=4)} with >=4y)")


if __name__ == "__main__":
    main()
