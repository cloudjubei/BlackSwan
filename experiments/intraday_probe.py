"""SWING-ii: powered INTRADAY probe on crypto (the one place a free-data surprise can still live). Edges that
exist purely at intraday frequency have millions of bars, so a real per-bar edge is both high-Sharpe AND
powered — the only question is whether per-bar taker cost kills it. We test short-horizon reversal / momentum /
intraday-momentum on hourly BTC (~9y, ~78k bars) net of realistic taker cost, with the HAC powered verdict.
A surviving net edge = a genuine free-data surprise; a powered-null = "even with 78k bars, cost eliminates it"."""
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from powered_battery import hac_verdict  # noqa: E402
from trainer.sharpe import sharpe_stats, minimum_detectable_sharpe  # noqa: E402

BS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAKER = 0.00055   # 55 bps? no -> Binance taker ~5.5bps per side; use 0.00055 = 5.5bps
PPY_H = 24 * 365  # hourly periods per year


def load_hourly(sym):
    paths = sorted(glob.glob(f"{BS}/binance/{sym}-1h-*.json"))
    if not paths:
        return None
    df = pd.concat([pd.read_json(p) for p in paths], ignore_index=True)
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp_close"]).dt.tz_localize(None))
    px = pd.Series(df["price"].to_numpy(), index=idx).groupby(level=0).last().sort_index()
    return px


def verdict_hourly(net, label, turnover_per_bar):
    ann = math.sqrt(PPY_H)
    st = sharpe_stats(net.to_numpy())
    v = hac_verdict(net)  # hac_verdict annualizes by sqrt(252) internally; re-annualize to hourly below
    # hac_verdict uses ANN=sqrt(252); convert its per-obs numbers to hourly-annualized
    per_obs = {k: v[k] / math.sqrt(252) for k in ("sharpe_ann", "sr_adj_ann", "lo_ann", "hi_ann", "mde_ann")}
    out = {k: per_obs[k] * ann for k in per_obs}
    return dict(label=label, n=st["n_obs"], turnover_per_bar=turnover_per_bar,
                sharpe_ann=out["sharpe_ann"], lo=out["lo_ann"], hi=out["hi_ann"], mde=out["mde_ann"],
                verdict=("survivor" if out["lo_ann"] > 0 else ("powered-null" if out["hi_ann"] < 0.5 else "inconclusive")))


def main():
    px = load_hourly("BTCUSDT")
    if px is None:
        print("no BTC hourly"); return
    r = px.pct_change()
    print(f"BTC hourly: {len(px)} bars {px.index[0]}..{px.index[-1]} ({len(px)/PPY_H:.1f}y)")
    print("=" * 104)
    print(f"INTRADAY powered probe (taker {TAKER*1e4:.1f}bps/side, hourly-annualized Sharpe, HAC powered verdict @ SR_econ=0.5)")
    print("=" * 104)
    print(f"{'signal':28s}{'gross_Sh':>9s}{'net_Sh':>8s}{'lo':>7s}{'hi':>7s}{'MDE':>6s}{'trn/bar':>8s}  verdict")

    def report(name, pos):
        pos = pos.reindex(px.index).fillna(0.0)
        gross = pos.shift(1).fillna(0.0) * r
        dpos = pos.diff().abs().fillna(0.0)
        net = gross - TAKER * dpos
        g = sharpe_stats(gross.dropna().to_numpy())["sharpe"] * math.sqrt(PPY_H)
        v = verdict_hourly(net.dropna(), name, float(dpos.mean()))
        print(f"{name:28s}{g:>9.2f}{v['sharpe_ann']:>8.2f}{v['lo']:>7.2f}{v['hi']:>7.2f}{v['mde']:>6.2f}{v['turnover_per_bar']:>8.3f}  {v['verdict']}")
        return v

    results = []
    # short-horizon time-series REVERSAL: position = -sign(return over last h bars), hold 1 bar
    for h in [1, 3, 6, 12]:
        results.append(report(f"reversal {h}h (hold 1h)", -np.sign(px / px.shift(h) - 1.0)))
    # time-series MOMENTUM: position = sign(return over last h bars)
    for h in [24, 72, 168]:
        results.append(report(f"momentum {h}h (hold 1h)", np.sign(px / px.shift(h) - 1.0)))
    # INTRADAY MOMENTUM (Gao et al / Shen-Urquhart-Wang): first-half-of-UTC-day return sign -> hold last 12h
    hod = px.index.hour
    day = px.index.normalize()
    first_half = px.groupby(day).apply(lambda s: s[s.index.hour == 11].mean() / s[s.index.hour == 0].mean() - 1.0 if len(s) else np.nan)
    fh = pd.Series(first_half.values, index=pd.DatetimeIndex(first_half.index))
    sig = pd.Series(np.sign(fh.reindex(day).values), index=px.index)
    sig[hod < 12] = 0.0  # only hold in the second half of the day
    results.append(report("intraday-mom (1st half->2nd)", sig))
    # HOUR-OF-DAY seasonal: long the historically-best UTC hour, short the worst (in-sample-picked = optimistic)
    hourly_mean = r.groupby(hod).mean()
    best, worst = int(hourly_mean.idxmax()), int(hourly_mean.idxmin())
    pos = pd.Series(0.0, index=px.index); pos[hod == best] = 1.0; pos[hod == worst] = -1.0
    results.append(report(f"hour-of-day L{best}/S{worst} (IS-picked)", pos))

    surv = [x for x in results if x["verdict"] == "survivor"]
    print(f"\n  SURVIVORS (net, lo>0): {len(surv)}  {[s['label'] for s in surv]}")
    print(f"  median MDE across intraday signals: {np.median([x['mde'] for x in results]):.2f} annualized Sharpe "
          f"(n~{len(px)} bars = {len(px)/PPY_H:.1f}y -> power is YEARS-limited, not bar-count-limited).")
    import json
    json.dump({"taker_bps": TAKER * 1e4, "n_bars": len(px), "years": len(px) / PPY_H, "signals": results},
              open(os.path.join(os.path.dirname(__file__), "intraday_probe_results.json"), "w"), indent=2, default=float)
    print("  wrote intraday_probe_results.json")


if __name__ == "__main__":
    main()
