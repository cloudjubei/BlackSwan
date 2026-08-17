"""Free Hyperliquid perp-DEX pull for the DATA-bet Phase 1 (positioning-pressure wedge). The public
`POST api.hyperliquid.xyz/info` endpoint is genuinely free but only exposes: hourly funding+premium history
(back to 2023-05-12) and OHLCV candles (daily = full history; 1h = rolling ~200d). Historical open-interest,
observed liquidations and wallet-attributable fills are NOT here -- they live in the requester-pays S3 archive
(Phase 2). So Phase 1 tests the cheap positioning-PRESSURE proxy (premium = mark-oracle, funding) as the gate
that decides whether the expensive S3 liquidation/wallet layer is worth paying for.

Writes hyperliquid/<COIN>.json = {coin, meta:{maxLeverage,szDecimals}, daily:{date:[o,h,l,c,v]},
funding:[[time_ms, fundingRate, premium], ...]} + hyperliquid/manifest.json. Idempotent (overwrites)."""
import json
import os
import time
import urllib.request

API = "https://api.hyperliquid.xyz/info"
OUT = "hyperliquid"
INCEPTION_MS = 1683849600000  # 2023-05-12, earliest funding
N_TOP = 12


def _post(body, tries=5):
    data = json.dumps(body).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as fh:
                return json.load(fh)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return None


def top_coins(n):
    meta, ctxs = _post({"type": "metaAndAssetCtxs"})
    rows = []
    for u, c in zip(meta["universe"], ctxs):
        if u.get("isDelisted"):
            continue
        try:
            notional = float(c["openInterest"]) * float(c["markPx"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((u["name"], notional, u.get("maxLeverage"), u.get("szDecimals")))
    rows.sort(key=lambda r: -r[1])
    return rows[:n]


def fetch_funding(coin, now_ms):
    out = []
    start = INCEPTION_MS
    while True:
        page = _post({"type": "fundingHistory", "coin": coin, "startTime": start, "endTime": now_ms})
        if not page:
            break
        for p in page:
            out.append([p["time"], float(p["fundingRate"]), float(p["premium"])])
        if len(page) < 500:
            break
        start = page[-1]["time"] + 1
        time.sleep(0.15)
    out.sort(key=lambda r: r[0])
    dedup = []
    seen = set()
    for r in out:
        if r[0] in seen:
            continue
        seen.add(r[0])
        dedup.append(r)
    return dedup


def fetch_daily(coin, now_ms):
    page = _post({"type": "candleSnapshot",
                  "req": {"coin": coin, "interval": "1d", "startTime": INCEPTION_MS, "endTime": now_ms}})
    data = {}
    for c in page or []:
        date = time.strftime("%Y-%m-%d", time.gmtime(c["t"] / 1000))
        data[date] = [float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]), float(c["v"])]
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    now_ms = int(time.time() * 1000)
    coins = top_coins(N_TOP)
    print(f"top {len(coins)} perps by notional OI: {', '.join(c[0] for c in coins)}")
    manifest = {}
    for name, notional, maxlev, szdec in coins:
        funding = fetch_funding(name, now_ms)
        daily = fetch_daily(name, now_ms)
        rec = {"coin": name, "meta": {"maxLeverage": maxlev, "szDecimals": szdec,
                                      "oi_notional_usd": round(notional)},
               "daily": daily, "funding": funding}
        with open(f"{OUT}/{name}.json", "w") as fh:
            json.dump(rec, fh)
        dkeys = sorted(daily)
        fdays = (funding[-1][0] - funding[0][0]) / 86400000.0 if funding else 0
        manifest[name] = {"oi_notional_usd": round(notional), "daily_bars": len(dkeys),
                          "daily_from": dkeys[0] if dkeys else None, "daily_to": dkeys[-1] if dkeys else None,
                          "funding_pts": len(funding), "funding_years": round(fdays / 365.0, 2)}
        print(f"  {name:8s} OI ${notional/1e6:8.1f}M  daily {len(dkeys):>5d} bars  "
              f"funding {len(funding):>6d} pts ({fdays/365.0:.2f}y)  {dkeys[0] if dkeys else '-'}..{dkeys[-1] if dkeys else '-'}")
        time.sleep(0.2)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    print(f"wrote {len(manifest)} coins -> {OUT}/")


if __name__ == "__main__":
    main()
