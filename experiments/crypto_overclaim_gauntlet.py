"""REFEREE candidate #6 -- crypto-native directional-alpha overclaim, net of the costs the papers omit. Published
crypto cross-sectional momentum / long-short 'alpha' (e.g. weekly LS ~2.62%/wk t=4.22, ensemble gross 1640%) is
typically reported with NO taker fees, NO slippage, and -- uniquely to perps -- NO FUNDING (crowded-long winners
PAY funding, which momentum harvests the wrong way). We reproduce cross-sectional momentum on the free Binance-perp
panel we hold, charge taker + slippage + funding, deflate through the honesty gauntlet + matched-complexity
random-formula null, and split LIQUID vs ILLIQUID. PRE-REGISTERED KILL: if net-of-cost deflated alpha SURVIVES in
the LIQUID subset, it is a genuine tradeable edge -- report it, do not refute. Demonstration script."""
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.ml_trading_gauntlet import run_gauntlet  # noqa: E402
from trainer.random_formula_null import random_formula_null, signal_to_weights  # noqa: E402
from trainer.trading_costs import turnover_series  # noqa: E402

PPY = 365.0
TAKER = 0.0005
SLIP = 0.0005
LOOKBACK = 14
SKIP = 1
HOLD = 7


def ms_to_date(ms):
    return datetime.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d")


def load(n_assets, min_bars=900):
    px, vol, fund = {}, {}, {}
    for f in sorted(os.listdir("cexperps")):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join("cexperps", f)))
        p = d.get("price", {})
        if len(p) < min_bars:
            continue
        sym = d.get("sym", f[:-5])
        px[sym] = {k: p[k][3] for k in p if p[k][3]}
        vol[sym] = {k: p[k][3] * p[k][4] for k in p if p[k][3]}
    for f in sorted(os.listdir("cexfunding")):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join("cexfunding", f)))
        sym = d.get("sym", f[:-5])
        daily = {}
        for ms, rate in d.get("funding", []):
            daily[ms_to_date(ms)] = daily.get(ms_to_date(ms), 0.0) + float(rate)
        fund[sym] = daily
    top = sorted([s for s in px if s in fund], key=lambda s: len(px[s]), reverse=True)[:n_assets]
    common = sorted(set.intersection(*[set(px[s]) for s in top]))
    P = np.array([[px[s][dt] for s in top] for dt in common], dtype=float)
    V = np.array([[vol[s].get(dt, 0.0) for s in top] for dt in common], dtype=float)
    F = np.array([[fund[s].get(dt, 0.0) for s in top] for dt in common], dtype=float)
    return top, common, P, V[1:], F[1:]


def momentum_signal(R):
    T, N = R.shape
    M = np.zeros((T, N))
    for t in range(LOOKBACK + SKIP, T):
        M[t] = R[t - LOOKBACK - SKIP:t - SKIP].sum(axis=0)
    return M


def held_weights(signal):
    T, N = signal.shape
    W, cur = np.zeros((T, N)), np.zeros(N)
    for t in range(T):
        if t % HOLD == 0:
            cur = signal_to_weights(signal[t:t + 1])[0]
        W[t] = cur
    return W


def strat_returns(R, F, signal):
    W = held_weights(signal)
    lag = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gross = (lag * R).sum(axis=1)
    turn = turnover_series(lag)
    funding = -(lag * F).sum(axis=1)
    net = gross - (TAKER + SLIP) * turn + funding
    return gross, net


def judge(name, R, F, signal, features):
    gross, net = strat_returns(R, F, signal)
    g_sh = gross.mean() / gross.std(ddof=1) * np.sqrt(PPY) if gross.std() > 0 else 0.0
    n_sh = net.mean() / net.std(ddof=1) * np.sqrt(PPY) if net.std() > 0 else 0.0
    null = random_formula_null(np.random.default_rng(0), features, R, depth=2, k=300, fee=TAKER + SLIP,
                               periods_per_year=PPY)
    per, _ = run_gauntlet([net], n_trials=1, null_sharpes=null, cutoff=len(net) // 2, periods_per_year=PPY)
    p = per[0]
    print(f"  {name}: GROSS Sharpe {g_sh:+.2f} -> NET {n_sh:+.2f} | null_p {p['null_p']:.3f} "
          f"econ {p['econ_pass']} dsr {p['dsr']:.2f} postcut {p['postcutoff_sharpe_ann']:+.2f} "
          f"-> SURVIVES {p['survives']}")
    return bool(p["survives"])


def main():
    syms, dates, P, V, F = load(40)
    R = np.diff(np.log(P), axis=0)
    sig = momentum_signal(R)
    print(f"crypto XS momentum (lb{LOOKBACK}/skip{SKIP}/hold{HOLD}), {R.shape[1]} perps x {R.shape[0]} bars, "
          f"taker+slip {int((TAKER+SLIP)*1e4)}bps + funding\n")
    med_dollar_vol = np.median(V, axis=0)
    liquid = np.argsort(med_dollar_vol)[-R.shape[1] // 2:]
    illiquid = np.argsort(med_dollar_vol)[:R.shape[1] // 2]
    feats_all = [R, np.sign(R), sig]
    surv_all = judge("ALL", R, F, sig, feats_all)
    surv_liq = judge("LIQUID-half", R[:, liquid], F[:, liquid], sig[:, liquid], [R[:, liquid], sig[:, liquid]])
    surv_ill = judge("ILLIQUID-half", R[:, illiquid], F[:, illiquid], sig[:, illiquid],
                     [R[:, illiquid], sig[:, illiquid]])
    print()
    if surv_liq:
        print("KILL: net-of-cost crypto momentum SURVIVES in the LIQUID subset -> genuine tradeable edge, report it.")
    elif surv_all or surv_ill:
        print("REFUTATION STANDS (with nuance): crypto momentum survives only in the full/ILLIQUID set, not the "
              "liquid tradeable one -> the published alpha is an illiquid-tail/cost artifact, gone where you can "
              "actually trade it.")
    else:
        print("REFUTATION STANDS: crypto XS momentum does NOT survive net of taker+slippage+funding anywhere -> "
              "the gross-only published alpha is a cost artifact.")


if __name__ == "__main__":
    main()
