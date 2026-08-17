"""DATA-bet Phase 1, intraday cut -- the honest completion of the free-data test. The positioning-reversion
thesis lives INTRADAY (liquidation cascades reverse in minutes-to-hours), so a daily test can wash it out. Free
REST gives 1h candles for the recent ~200 days only (rolling ~5000 bars) -- underpowered (~0.55y) but the right
place to see whether a GROSS intraday pulse exists and whether taker cost kills it. Signals: premium/funding
positioning-reversion, a liquidation-cascade PROXY (fade a sharp move made while positioning was extreme -- a
heuristic stand-in, NOT observed liquidations, which need the S3 archive), and a pure 1h price-reversal baseline
to reproduce the known gross-pulse/net-dead pattern inside this harness. Reuses the daily battery's signal
helpers and the shared powered-null referee at hourly frequency (periods_per_year=8760)."""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyperliquid_positioning import ex_ante_lead, signal_position, smoothed_position, zscore  # noqa: E402
from powered_battery import hac_verdict  # noqa: E402
from trainer.sharpe import benjamini_hochberg, benjamini_yekutieli  # noqa: E402

API = "https://api.hyperliquid.xyz/info"
DATA = "hyperliquid"
BPS = 1e-4
TAKER = 5.0 * BPS
PPY = 24.0 * 365.0
LOOKBACK = 72
HORIZONS = (1, 6, 24)
COST_MULT = (0.0, 1.0, 2.0, 4.0)
COINS = ("BTC", "ETH", "SOL", "XRP", "LINK", "AAVE", "NEAR")


def _post(body, tries=5):
    data = json.dumps(body).encode()
    for i in range(tries):
        try:
            req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return None


def fetch_hourly_close(coin, now_ms):
    start = now_ms - 5000 * 3600 * 1000
    page = _post({"type": "candleSnapshot", "req": {"coin": coin, "interval": "1h",
                                                    "startTime": start, "endTime": now_ms}})
    return {int(c["t"]): float(c["c"]) for c in (page or [])}


def load_hourly(coin, now_ms):
    rec = json.load(open(os.path.join(DATA, f"{coin}.json")))
    fund = {int(t) - (int(t) % 3600000): (float(r), float(p)) for t, r, p in rec["funding"]}
    close = fetch_hourly_close(coin, now_ms)
    hours = sorted(t for t in close if t in fund)
    if len(hours) < LOOKBACK + 100:
        return None
    px = np.array([close[t] for t in hours], dtype=float)
    ret = np.concatenate([[np.nan], px[1:] / px[:-1] - 1.0])
    funding = np.array([fund[t][0] for t in hours], dtype=float)
    premium = np.array([fund[t][1] for t in hours], dtype=float)
    return {"coin": coin, "ret": ret, "funding": funding, "premium": premium}


def price_reversal_position(ret, lb):
    return signal_position(zscore(ret, lb))


def cascade_proxy_position(ret, premium, lb):
    zr = zscore(ret, lb)
    zp = zscore(premium, lb)
    pos = np.full(ret.shape, np.nan)
    for t in range(len(ret)):
        if not np.isfinite(zr[t]) or not np.isfinite(zp[t]):
            continue
        sharp = abs(zr[t]) > 1.5
        extreme = abs(zp[t]) > 1.0
        pos[t] = -np.clip(zr[t], -3.0, 3.0) / 3.0 if (sharp and extreme) else 0.0
    return pos


def strat_returns(pos, ret, h, cost):
    eff = smoothed_position(pos, h)
    n = len(ret)
    out = np.full(n, np.nan)
    prev = 0.0
    for t in range(n - 1):
        p = eff[t]
        if not np.isfinite(p) or not np.isfinite(ret[t + 1]):
            continue
        out[t + 1] = p * ret[t + 1] - cost * abs(p - prev)
        prev = p
    return out[np.isfinite(out)]


def main():
    now_ms = int(time.time() * 1000)
    coins = [c for c in (load_hourly(cn, now_ms) for cn in COINS) if c]
    n_bars = int(np.median([np.isfinite(c["ret"]).sum() for c in coins]))
    print(f"loaded {len(coins)} coins hourly (~{n_bars} bars ≈ {n_bars/24/365:.2f}y), "
          f"cost {TAKER/BPS:.1f}bps/side, referee ppy={int(PPY)}\n")

    def sig_positions(c):
        return {"premium": signal_position(zscore(c["premium"], LOOKBACK)),
                "funding": signal_position(zscore(c["funding"], LOOKBACK)),
                "px_reversal": price_reversal_position(c["ret"], LOOKBACK),
                "cascade_proxy": cascade_proxy_position(c["ret"], c["premium"], LOOKBACK)}

    cells = []
    print(f"{'coin':<6}{'signal':<14}{'h':>3}  " + "".join(f"{'net@'+str(int(m))+'x':>10}" for m in COST_MULT)
          + f"{'exAnte':>9}{'contemp':>9}")
    for c in coins:
        sigs = sig_positions(c)
        for sname, pos in sigs.items():
            lead = ex_ante_lead(pos, c["ret"])
            for h in HORIZONS:
                row = {}
                for m in COST_MULT:
                    r = strat_returns(pos, c["ret"], h, TAKER * m)
                    if r.size < 200:
                        row[m] = None
                        continue
                    v = hac_verdict(r, periods_per_year=PPY)
                    row[m] = v
                    if m == 1.0:
                        cells.append({"coin": c["coin"], "signal": sname, "h": h,
                                      "sharpe_ann": v["sharpe_ann"], "p_one": v["p_one"],
                                      "verdict": v["verdict"], "lo_ann": v["lo_ann"], "hi_ann": v["hi_ann"],
                                      "n": v["n_obs"], "corr_ex_ante": lead["corr_ex_ante"],
                                      "corr_contemp": lead["corr_contemp"]})
                if row.get(1.0) is None:
                    continue
                cells_str = "".join(f"{(row[m]['sharpe_ann'] if row[m] else float('nan')):>10.2f}" for m in COST_MULT)
                print(f"{c['coin']:<6}{sname:<14}{h:>3}  {cells_str}"
                      f"{lead['corr_ex_ante']:>9.3f}{lead['corr_contemp']:>9.3f}")

    pvals = [c["p_one"] for c in cells]
    bh, by = benjamini_hochberg(pvals, 0.05), benjamini_yekutieli(pvals, 0.05)
    n_surv = sum(1 for c in cells if c["verdict"] == "survivor")
    n_pn = sum(1 for c in cells if c["verdict"] == "powered-null")
    print(f"\n=== FAMILY ({len(cells)} cells @ 1x cost) ===")
    print(f"per-cell: {n_surv} survivor / {n_pn} powered-null / {len(cells)-n_surv-n_pn} inconclusive")
    print(f"FDR: BH rejects {sum(bh)}, BY rejects {sum(by)} (q=0.05)")
    gross_surv = [c for c in cells if c["sharpe_ann"] > 0]
    print(f"cells net-positive @1x: {len(gross_surv)}/{len(cells)}")
    summary = {"n_coins": len(coins), "median_bars": n_bars, "n_cells": len(cells),
               "n_survivor": n_surv, "n_powered_null": n_pn, "n_bh": sum(bh), "n_by": sum(by), "cells": cells}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/hyperliquid_intraday.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/hyperliquid_intraday.json")


if __name__ == "__main__":
    main()
