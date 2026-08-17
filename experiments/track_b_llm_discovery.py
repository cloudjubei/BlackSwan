"""Track B: an actual LLM discovery loop, end-to-end. Real Claude agents proposed executable market-timing
strategy specs (see the `track-b-llm-strategy-proposals` workflow -> experiments/results/track_b_specs.json); here
we backtest each on a real US equity index and run the whole family through the certification gauntlet
(trainer.certification, effective-trials deflation). The honest question the earlier gauntlet could not answer
because no LLM was involved: do REAL LLM-designed strategies behave differently from a mechanical random zoo?
  - do any CERTIFY after multiplicity + cost + FDR?
  - is their BETA-NEUTRAL out-of-sample Sharpe distinguishable from the mechanical zoo's (i.e. does LLM design add
    value over random rule-search), and from zero?
Prior/honest expectation: LLM proposals are educated guesses over the same overfit space -> ~0 certified, beta-
neutral Sharpe ~ the mechanical zoo ~ 0. A clean negative that punctures LLM-alpha hype; a positive (LLM > zoo,
beta-neutral, certified) would be the genuinely LLM-specific result. (A strict pre/post training-cutoff
contamination test is underpowered here: the proposer's knowledge cutoff is ~2026-01, leaving too little
post-cutoff data -- reported as a caveated secondary only.)"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_alpha_gauntlet import market_series, strategy_zoo  # noqa: E402
from trainer.certification import certify_family  # noqa: E402
from trainer.sharpe import sharpe_stats  # noqa: E402

ANN = math.sqrt(252.0)
COST = 1e-4
SPECS = "experiments/results/track_b_specs.json"


def _sma(a, w):
    w = max(2, int(w))
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = np.full(len(a), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _roll_std(a, w):
    w = max(2, int(w))
    out = np.full(len(a), np.nan)
    for t in range(w, len(a)):
        out[t] = np.std(a[t - w:t])
    return out


def _rsi(price, period):
    period = max(2, int(period))
    d = np.diff(price, prepend=price[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    out = np.full(len(price), np.nan)
    for t in range(period, len(price)):
        ag = gain[t - period + 1:t + 1].mean()
        al = loss[t - period + 1:t + 1].mean()
        out[t] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def build_position(spec, price):
    t, p, n = spec["type"], spec.get("params", {}), len(price)
    ret_lb = lambda L: np.concatenate([[np.nan] * int(L), price[int(L):] / price[:-int(L)] - 1.0]) if int(L) >= 1 else np.zeros(n)  # noqa: E731
    pos = np.zeros(n)
    try:
        if t == "ma_crossover":
            f, s = int(p.get("fast", 10)), int(p.get("slow", 50))
            if f >= s:
                s = f + 1
            pos = np.sign(np.nan_to_num(_sma(price, f) - _sma(price, s)))
        elif t == "momentum":
            pos = np.sign(np.nan_to_num(ret_lb(p.get("lookback", 60))))
        elif t == "reversal":
            pos = -np.sign(np.nan_to_num(ret_lb(p.get("lookback", 5))))
        elif t == "rsi":
            r = _rsi(price, p.get("period", 14))
            lo, hi = float(p.get("low", 30)), float(p.get("high", 70))
            pos = np.nan_to_num(np.where(r < lo, 1.0, np.where(r > hi, -1.0, 0.0)))
        elif t == "bollinger":
            w, k = int(p.get("window", 20)), float(p.get("k", 2))
            mid, sd = _sma(price, w), _roll_std(price, w)
            pos = np.nan_to_num(np.where(price < mid - k * sd, 1.0, np.where(price > mid + k * sd, -1.0, 0.0)))
        elif t == "breakout":
            w = int(p.get("window", 50))
            rmax = np.full(n, np.nan)
            rmin = np.full(n, np.nan)
            for i in range(w, n):
                rmax[i] = price[i - w:i].max()
                rmin[i] = price[i - w:i].min()
            pos = np.nan_to_num(np.where(price > rmax, 1.0, np.where(price < rmin, -1.0, 0.0)))
        elif t == "vol_target_momentum":
            lb, vw = int(p.get("lookback", 60)), int(p.get("vol_window", 20))
            mom = np.sign(np.nan_to_num(ret_lb(lb)))
            vol = _roll_std(price, vw)
            vol = np.where(np.isfinite(vol) & (vol > 0), vol, np.nan)
            pos = np.nan_to_num(mom / vol)
            sd = np.nanstd(pos)
            pos = pos / sd if sd > 0 else pos
    except Exception:  # noqa: BLE001
        return np.zeros(n)
    if spec.get("direction") == "long_only":
        pos = np.clip(pos, 0.0, 1.0)
    return np.clip(pos, -1.0, 1.0)


def pos_to_returns(pos, m):
    held = np.concatenate([[0.0], pos[:-1]])
    turn = np.abs(np.diff(held, prepend=0.0))
    return held * m - COST * turn


def beta_neutral_sharpe(r, mkt):
    m2 = np.isfinite(r) & np.isfinite(mkt)
    r, mkt = r[m2], mkt[m2]
    if r.size < 50 or mkt.var() == 0:
        return np.nan
    beta = np.cov(r, mkt)[0, 1] / mkt.var()
    return sharpe_stats(r - beta * mkt)["sharpe"] * ANN


def main():
    specs = json.load(open(SPECS))["strategies"]
    dates, m = market_series()
    price = np.cumprod(1.0 + np.nan_to_num(m))
    rets, kept = [], []
    for s in specs:
        pos = build_position(s, price)
        if np.all(pos == 0) or not np.isfinite(pos).all():
            continue
        rets.append(pos_to_returns(pos, m))
        kept.append(s)
    print(f"LLM proposed {len(specs)} strategies; {len(kept)} executable & non-degenerate")
    split = int(len(m) * 0.7)
    oos = [r[split:] for r in rets]
    per, fam = certify_family(oos, n_trials="effective")
    is_sh = [sharpe_stats(r[:split])["sharpe"] * ANN for r in rets]
    oos_sh = [p["sharpe_ann"] for p in per]
    exp_sh = [float(s.get("expected_annual_sharpe", 0)) for s in kept]
    bn = [beta_neutral_sharpe(r[split:], m[split:]) for r in rets]

    print(f"\n=== LLM strategies through the gauntlet (effective trials={fam['effective_trials']:.1f} of {len(kept)}) ===")
    print(f"  LLM's OWN expected Sharpe (mean):     {np.mean(exp_sh):+.2f}  (what the models believed)")
    print(f"  realized IN-SAMPLE Sharpe (mean):     {np.mean(is_sh):+.2f}")
    print(f"  realized OOS raw Sharpe (mean):       {np.mean(oos_sh):+.2f}")
    print(f"  realized OOS BETA-NEUTRAL Sharpe:     {np.nanmean(bn):+.2f}  (timing skill, market removed)")
    print(f"  OOS nominally significant: {fam['n_nominal_sig']}; DSR pass: {fam['n_dsr_pass']}; "
          f"CERTIFIED: {fam['n_certified']}/{len(kept)}")

    llm_bn = np.array([x for x in bn if np.isfinite(x)])

    def same_type_random(spec, rng):
        t = spec["type"]
        rp = {"ma_crossover": lambda: {"fast": int(rng.integers(2, 100)), "slow": int(rng.integers(101, 300))},
              "momentum": lambda: {"lookback": int(rng.integers(2, 300))},
              "reversal": lambda: {"lookback": int(rng.integers(1, 60))},
              "rsi": lambda: {"period": int(rng.integers(2, 50)), "low": float(rng.integers(5, 40)),
                              "high": float(rng.integers(60, 95))},
              "bollinger": lambda: {"window": int(rng.integers(5, 100)), "k": float(rng.uniform(1, 3))},
              "breakout": lambda: {"window": int(rng.integers(10, 300))},
              "vol_target_momentum": lambda: {"lookback": int(rng.integers(10, 300)),
                                              "vol_window": int(rng.integers(10, 100))}}[t]()
        return {"type": t, "params": rp, "direction": spec["direction"]}

    rng = np.random.default_rng(7)
    baseline_bn = []
    for s in kept:
        for _ in range(30):
            p = build_position(same_type_random(s, rng), price)
            if not np.all(p == 0):
                baseline_bn.append(beta_neutral_sharpe(pos_to_returns(p, m)[split:], m[split:]))
    baseline_bn = np.array([x for x in baseline_bn if np.isfinite(x)])
    diff = llm_bn.mean() - baseline_bn.mean()
    se = math.sqrt(llm_bn.var(ddof=1) / llm_bn.size + baseline_bn.var(ddof=1) / baseline_bn.size)
    tstat = diff / se if se > 0 else float("nan")
    print(f"\n=== FAIR baseline: LLM vs RANDOM-PARAM strategies of the SAME TYPES (beta-neutral OOS Sharpe) ===")
    print(f"  LLM mean {llm_bn.mean():+.3f} (n={llm_bn.size}) vs same-type random {baseline_bn.mean():+.3f} "
          f"(n={baseline_bn.size}); diff {diff:+.3f}, t={tstat:+.2f}")
    print(f"  => {'LLM param/design choice adds value over its own types' if tstat > 2 else 'LLM edge is just PICKING GOOD TYPES, not design (~same as random params of same type)'}")

    full_bn = [beta_neutral_sharpe(r, m) for r in rets]
    di = np.array([np.datetime64(d) for d in dates])
    eras = [("<=2005", di < np.datetime64("2005-01-01")), ("2005-2015", (di >= np.datetime64("2005-01-01")) & (di < np.datetime64("2015-01-01"))), (">=2015", di >= np.datetime64("2015-01-01"))]
    print(f"\n=== IS THE EDGE LIVE OR DECAYED? LLM beta-neutral Sharpe by era (full sample) ===")
    era_means = {}
    for name, mask in eras:
        vals = [beta_neutral_sharpe(r[mask], m[mask]) for r in rets]
        vals = np.array([x for x in vals if np.isfinite(x)])
        era_means[name] = float(vals.mean())
        print(f"  {name:10s}: mean beta-neutral Sharpe {vals.mean():+.3f}")

    summary = {"n_proposed": len(specs), "n_executable": len(kept), "effective_trials": fam["effective_trials"],
               "mean_expected_sharpe": float(np.mean(exp_sh)), "mean_is_sharpe": float(np.mean(is_sh)),
               "mean_oos_sharpe": float(np.mean(oos_sh)), "mean_oos_beta_neutral": float(np.nanmean(bn)),
               "n_certified": fam["n_certified"], "n_nominal_sig": fam["n_nominal_sig"],
               "llm_bn_mean": float(llm_bn.mean()), "same_type_random_bn_mean": float(baseline_bn.mean()),
               "llm_vs_same_type_t": float(tstat), "era_beta_neutral": era_means}
    json.dump(summary, open("experiments/results/track_b_llm_discovery.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/track_b_llm_discovery.json")


if __name__ == "__main__":
    main()
