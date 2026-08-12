"""The DeMiguel-Martin-Utrera-Nogales-Uppal (2020) NETTING test on the free cross-asset battery.

DeMiguel et al. show transaction costs INCREASE the count of jointly-significant characteristics (6->15) because
trading one signal partly OFFSETS the trades of another: combining strategies at the POSITION level nets opposing
trades on the same asset, cutting turnover-per-unit-signal, so a COMBINATION can survive costs even when every
standalone does not. Test: reconstruct 5 flagship cross-sectional strategies (momentum, 5yr value/reversal,
short-term reversal, low-vol, trend) on the ONE shared diversified universe; compute each standalone net Sharpe;
then the equal-weight COMBINED position book with its NETTED turnover; and ask whether the combination clears
t>3 when the standalones don't — and how much of any gain is netting vs plain diversification."""
import math
import os
import numpy as np
import pandas as pd
from trainer import xsection

ASSETS = ["GOLD", "SILVER", "COPPER", "WTI", "SPY", "TLT", "IEF", "UUP"]
MONTHS = [(y, m) for y in range(2006, 2027) for m in range(1, 13)]
FEE = 0.0005          # 5 bps realistic per-trade cost for this liquid multi-asset panel
K = 2                 # k long / k short
REB = 21              # monthly rebalance
WINS = [(y, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")) for y in range(2012, 2024)]

PX = xsection.align_prices(xsection._load_universe(ASSETS, MONTHS))
ASSETS = [a for a in ASSETS if a in PX.columns]
RET = xsection.step_returns(PX)
print(f"panel: {len(ASSETS)} assets {ASSETS}  {len(PX)} bars {PX.index[0].date()}..{PX.index[-1].date()}")


def weight_book(score, invert=False, k=K, reb=REB):
    """k-long/k-short unit-gross cross-sectional book from a score, monthly rebalance, one-bar-lagged."""
    idx = PX.index
    W = pd.DataFrame(0.0, index=idx, columns=ASSETS)
    held = pd.Series(0.0, index=ASSETS); since = reb
    for i, ts in enumerate(idx):
        W.iloc[i] = held.values
        since += 1
        if since < reb:
            continue
        s = score.loc[ts].dropna()
        if len(s) < 2 * k:
            continue
        since = 0
        order = s.sort_values(ascending=invert)
        nxt = pd.Series(0.0, index=ASSETS)
        for sym in order.index[:k]:
            nxt[sym] = 1.0 / k
        for sym in order.index[-k:]:
            nxt[sym] = -1.0 / k
        held = nxt
    return W


def trend_book(lookback=200):
    """Time-series trend: each asset +/- by its own sign, scaled to unit gross."""
    sig = np.sign(PX / PX.shift(lookback) - 1.0)
    idx = PX.index; W = pd.DataFrame(0.0, index=idx, columns=ASSETS)
    held = pd.Series(0.0, index=ASSETS); since = REB
    for i, ts in enumerate(idx):
        W.iloc[i] = held.values; since += 1
        if since < REB:
            continue
        s = sig.loc[ts].dropna()
        if s.abs().sum() == 0:
            continue
        since = 0
        held = (s / s.abs().sum()).reindex(ASSETS).fillna(0.0)
    return W


def stats(W, fee=FEE):
    gross = (W.shift(1).fillna(0.0) * RET).sum(axis=1)
    dW = W.diff().abs().sum(axis=1)                 # daily turnover = sum of |position changes|
    net = gross - fee * dW
    return gross, net, dW


def perwin_sharpe(r):
    sh = [seg.mean() / seg.std() for _y, a, b in WINS
          for seg in [r[(r.index >= a) & (r.index <= b)].dropna()] if len(seg) > 30 and seg.std() > 0]
    if not sh:
        return np.nan, np.nan, 0
    m = np.mean(sh); t = m / (np.std(sh) / math.sqrt(len(sh))) if np.std(sh) > 0 else np.nan
    return m * math.sqrt(252), t, len(sh)


VOL = RET.rolling(60).std()
SCORES = {
    "momentum": (PX / PX.shift(252) - 1.0, False),      # long winners
    "value_5yr": (PX / PX.shift(1260) - 1.0, True),      # long 5yr losers (reversal)
    "st_reversal": (PX / PX.shift(21) - 1.0, True),      # long recent losers
    "lowvol": (VOL, True),                                # long lowest vol
}
books = {name: weight_book(sc, inv) for name, (sc, inv) in SCORES.items()}
books["trend"] = trend_book()

print("\n" + "=" * 84)
print("STANDALONE strategies (net of 5bps, per-year Sharpe over 2012-2023)")
print("=" * 84)
print(f"{'strategy':14s}{'annSharpe_net':>14s}{'t_win':>8s}{'turnover/yr':>13s}{'cost_drag/yr':>13s}")
standalone = {}
for name, W in books.items():
    gross, net, dW = stats(W)
    ann, t, n = perwin_sharpe(net)
    turn = dW.mean() * 252
    drag = FEE * turn
    standalone[name] = dict(W=W, net=net, gross=gross, dW=dW, ann=ann, t=t, turn=turn, drag=drag)
    print(f"{name:14s}{ann:>14.2f}{t:>8.2f}{turn:>13.2f}{drag*100:>12.2f}%")

# ---- COMBINED equal-weight POSITION book (netting) --------------------------------------------------------
Wcomb = sum(standalone[n]["W"] for n in books) / len(books)
gross_c, net_c, dW_c = stats(Wcomb)
ann_c, t_c, n_c = perwin_sharpe(net_c)
turn_c = dW_c.mean() * 252
# counterfactual: SAME combined gross but costs NOT netted (sum of standalone per-strategy turnover / N)
mean_standalone_turn = np.mean([standalone[n]["turn"] for n in books])
drag_netted = FEE * turn_c
drag_unnetted = FEE * mean_standalone_turn
# gross combined sharpe (no cost) for the diversification-only reference
ann_cg, t_cg, _ = perwin_sharpe(gross_c)
# combined net if costs were UN-netted (subtract the un-netted drag from the combined gross series proportionally)
net_c_unnetted = gross_c - FEE * (mean_standalone_turn / 252.0)  # flat avg daily un-netted turnover
ann_cu, t_cu, _ = perwin_sharpe(net_c_unnetted)

print("\n" + "=" * 84)
print("COMBINED book (equal-weight positions -> trades NET across strategies)")
print("=" * 84)
print(f"  combined GROSS (no cost)        : annSharpe {ann_cg:+.2f}  t {t_cg:+.2f}")
print(f"  combined NET (netted turnover)  : annSharpe {ann_c:+.2f}  t {t_c:+.2f}   turnover/yr {turn_c:.2f}  drag {drag_netted*100:.2f}%/yr")
print(f"  combined NET (UN-netted cost)   : annSharpe {ann_cu:+.2f}  t {t_cu:+.2f}   turnover/yr {mean_standalone_turn:.2f}  drag {drag_unnetted*100:.2f}%/yr")
print(f"  netting turnover reduction      : {mean_standalone_turn:.2f} -> {turn_c:.2f}  ({100*(1-turn_c/mean_standalone_turn):.0f}% lower)")
best_standalone_t = max((standalone[n]["t"] for n in books if standalone[n]["t"] == standalone[n]["t"]), default=np.nan)
print(f"\n  best STANDALONE t_win           : {best_standalone_t:+.2f}")
print(f"  COMBINED net t_win              : {t_c:+.2f}")
verdict = ("COMBINATION SURVIVES (t>3) where standalones do not" if (t_c == t_c and t_c > 3 and best_standalone_t < 3)
           else "combination does NOT clear t>3 either")
print(f"  VERDICT: {verdict}")

import json
out = dict(fee=FEE, assets=ASSETS,
          standalone={n: dict(annSharpe_net=standalone[n]["ann"], t_win=standalone[n]["t"], turnover_yr=standalone[n]["turn"]) for n in books},
          combined=dict(gross_annSharpe=ann_cg, gross_t=t_cg, net_annSharpe=ann_c, net_t=t_c, turnover_yr=turn_c,
                        net_unnetted_annSharpe=ann_cu, net_unnetted_t=t_cu, mean_standalone_turnover=mean_standalone_turn,
                        netting_turnover_reduction_pct=100 * (1 - turn_c / mean_standalone_turn)),
          best_standalone_t=best_standalone_t, verdict=verdict)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "netting_results.json"), "w"), indent=2, default=float)
print("\nwrote netting_results.json")
