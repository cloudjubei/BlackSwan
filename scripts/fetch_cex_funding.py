"""Pull Binance USDS-M perpetual FUNDING-RATE history (free) for the cexperps panel -- the exact series the crypto
folklore watches ('negative funding = bullish, extreme funding = reversal'). Needed to refute the DIRECTIONAL-
signal claim with a clean funding-carry-vs-price-prediction decomposition (the premium-index already in cexperps/
is the basis component). Binance funding is every 8h (paginated 1000/call, back to ~2019-2020). Writes
cexfunding/<SYM>.json = {sym, funding:[[fundingTime_ms, fundingRate], ...]}. Idempotent."""
import json
import os
import time
import urllib.request

FAPI = "https://fapi.binance.com"
OUT = "cexfunding"
START = 1560000000000  # ~2019-06


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


def fetch_funding(sym, now_ms):
    out = []
    start = START
    while start < now_ms:
        page = _get(f"/fapi/v1/fundingRate?symbol={sym}&startTime={start}&limit=1000")
        if not page:
            break
        for p in page:
            out.append([int(p["fundingTime"]), float(p["fundingRate"])])
        if len(page) < 1000:
            break
        start = page[-1]["fundingTime"] + 1
        time.sleep(0.05)
    dedup, seen = [], set()
    for t, r in sorted(out):
        if t not in seen:
            seen.add(t)
            dedup.append([t, r])
    return dedup


def main():
    os.makedirs(OUT, exist_ok=True)
    now_ms = int(time.time() * 1000)
    syms = sorted(json.load(open("cexperps/manifest.json")))
    print(f"pulling funding for {len(syms)} perps ...")
    manifest = {}
    for i, sym in enumerate(syms):
        try:
            f = fetch_funding(sym, now_ms)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: FAIL {e}")
            continue
        if len(f) < 100:
            continue
        json.dump({"sym": sym, "funding": f}, open(f"{OUT}/{sym}.json", "w"))
        manifest[sym] = {"n": len(f), "from": time.strftime("%Y-%m-%d", time.gmtime(f[0][0] / 1000)),
                         "to": time.strftime("%Y-%m-%d", time.gmtime(f[-1][0] / 1000))}
        if i % 50 == 0:
            print(f"  [{i:>3}/{len(syms)}] {sym:14s} {len(f):>5d} funding pts {manifest[sym]['from']}..{manifest[sym]['to']}")
        time.sleep(0.03)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"wrote {len(manifest)} symbols -> {OUT}/")


if __name__ == "__main__":
    main()
