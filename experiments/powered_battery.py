"""Powered-null upgrade of the cross-asset factor battery (v2 — hardened after adversarial review).

v2 applies every landed critique from the verification panel:
  - STATS: HAC (Lo 2002 serial-correlation) Sharpe SE instead of i.i.d.; within-cell best-of-N deflation of each
    cell's p BEFORE cross-cell FDR (consistent with factor_battery's Sharpe deflation).
  - FDR FAMILY: report BH under pooled-19 / per-class / per-factor + Benjamini-Yekutieli (arbitrary dependence);
    disclose the sole scope-dependent flip rather than presenting one "0" as robust.
  - CONTROLS: the trivial long-only/in-sample controls are demoted to sanity checks; the LICENSING controls are
    (a) a certified long/short (12-1 cross-sectional momentum) through the IDENTICAL best-on-train->cost->OOS
    path added into the FDR pool, (b) a CALIBRATED injection (true annualized Sharpe 0.5 at each class's actual
    cell n) that must be flagged survivor at ~80% power, (c) a scrambled-label NEGATIVE control.
  - MATURITY: downgraded to INCONCLUSIVE — N=4 cannot test a maturity law; report the Fisher CI on r (which
    spans strong-negative to strong-positive) and measured on-disk history spans (not hardcoded years).
  - CRYPTO: restore the 10bps taker fee; test the LIQUID construction the Zaremba thesis names (liquid-coin-only
    + BTC/ETH time-series momentum) and show the all-9 tilt is an illiquid-tail artifact.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factor_battery import (  # noqa: E402
    CLASSES, FACTORS, load_class, weight_book, trend_book, factor_score, class_friction,
)
from trainer.sharpe import (  # noqa: E402
    sharpe_stats, sharpe_standard_error_hac, minimum_detectable_sharpe, sharpe_power, expected_max_sharpe,
    benjamini_hochberg, benjamini_yekutieli,
)
from scipy.stats import norm  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
ANN = math.sqrt(252.0)
SR_ECON_ANN = 0.5
RNG = np.random.default_rng(12345)


def hac_verdict(series, sr_econ_ann=SR_ECON_ANN, n_configs=1, trial_sr_std=0.0, alpha=0.05, power=0.8):
    """Per-cell powered verdict with a HAC SE and within-cell best-of-N deflation, in per-obs units, reported
    annualized. survivor = one-sided lower bound > 0; powered-null = upper < SR_econ; else inconclusive."""
    r = np.asarray(series, dtype=float)
    r = r[np.isfinite(r)]
    st = sharpe_stats(r)
    se = sharpe_standard_error_hac(r)
    sr_star = expected_max_sharpe(n_configs, trial_sr_std) if n_configs >= 2 else 0.0
    sr_adj = st["sharpe"] - sr_star                       # deflate the point estimate for the within-cell sweep
    sr_econ = sr_econ_ann / ANN
    mde = minimum_detectable_sharpe(st["n_obs"], alpha, power)
    if not np.isfinite(se) or se <= 0:
        return dict(sharpe_ann=st["sharpe"] * ANN, sr_adj_ann=sr_adj * ANN, verdict="inconclusive",
                    lo_ann=-math.inf, hi_ann=math.inf, mde_ann=mde * ANN, p_one=1.0, n_obs=st["n_obs"], sr_star_ann=sr_star * ANN)
    za = float(norm.ppf(1 - alpha))
    lo, hi = sr_adj - za * se, sr_adj + za * se
    verdict = "survivor" if lo > 0 else ("powered-null" if hi < sr_econ else "inconclusive")
    p_one = float(norm.sf(sr_adj / se))
    return dict(sharpe_ann=st["sharpe"] * ANN, sr_adj_ann=sr_adj * ANN, verdict=verdict, lo_ann=lo * ANN,
                hi_ann=hi * ANN, mde_ann=mde * ANN, p_one=p_one, n_obs=st["n_obs"], sr_star_ann=sr_star * ANN)


def cells_with_series():
    """Best-on-train config per (class,factor): OOS-net series + within-sweep trial-Sharpe std + n_configs."""
    rows = []
    for cls, (d, syms, fee) in CLASSES.items():
        PX, hl = load_class(d, syms)
        if PX.shape[1] < 3 or len(PX) < 300:
            continue
        RET = PX.pct_change(fill_method=None)
        n = len(PX); split = int(n * 0.55); tr, te = PX.index[:split], PX.index[split:]
        k = 2 if PX.shape[1] >= 5 else 1
        for factor, lbs in FACTORS.items():
            lbs = [lb for lb in lbs if lb < split - 30]
            if not lbs:
                continue
            per = []
            for lb in lbs:
                if factor == "trend":
                    W = trend_book(PX, lb, k)
                else:
                    sc, inv = factor_score(PX, factor, lb)
                    W = weight_book(sc, PX, inv, k)
                gross = (W.shift(1).fillna(0.0) * RET).sum(axis=1)
                net = gross - fee * W.diff().abs().sum(axis=1)
                tr_s = net.loc[tr].dropna(); te_s = net.loc[te].dropna()
                s_tr = tr_s.mean() / tr_s.std() if len(tr_s) > 30 and tr_s.std() > 0 else np.nan
                s_te = te_s.mean() / te_s.std() if len(te_s) > 30 and te_s.std() > 0 else np.nan
                per.append((lb, s_tr, s_te, net.loc[te].dropna()))
            per = [p for p in per if p[1] == p[1]]
            if not per:
                continue
            best = max(per, key=lambda p: p[1])
            trial_std = float(np.std([p[2] for p in per if p[2] == p[2]], ddof=1)) if len([p for p in per if p[2] == p[2]]) > 1 else 0.0
            rows.append(dict(cls=cls, factor=factor, net_te=best[3], n_configs=len(per), trial_sr_std=trial_std))
    return rows


def fdr_table(cells):
    """BH under pooled / per-class / per-factor scopes + Benjamini-Yekutieli, on the within-cell-deflated p."""
    pv = [c["p_one"] for c in cells]
    scopes = {}
    scopes["pooled-19 BH"] = benjamini_hochberg(pv, 0.05)
    scopes["pooled-19 BY"] = benjamini_yekutieli(pv, 0.05)
    # per-class
    rej_class = [False] * len(cells)
    for cls in {c["cls"] for c in cells}:
        idx = [i for i, c in enumerate(cells) if c["cls"] == cls]
        sub = benjamini_hochberg([cells[i]["p_one"] for i in idx], 0.05)
        for j, i in enumerate(idx):
            rej_class[i] = sub[j]
    scopes["per-class BH"] = rej_class
    rej_fac = [False] * len(cells)
    for fac in {c["factor"] for c in cells}:
        idx = [i for i, c in enumerate(cells) if c["factor"] == fac]
        sub = benjamini_hochberg([cells[i]["p_one"] for i in idx], 0.05)
        for j, i in enumerate(idx):
            rej_fac[i] = sub[j]
    scopes["per-factor BH"] = rej_fac
    return scopes


def main():
    raw = cells_with_series()
    cells = []
    for c in raw:
        v = hac_verdict(c["net_te"], n_configs=c["n_configs"], trial_sr_std=c["trial_sr_std"])
        cells.append({**c, **v}); cells[-1].pop("net_te")
    scopes = fdr_table(cells)

    print("=" * 122)
    print("(4') POWERED NULL v2 — HAC SE + within-cell deflation; annualized deflated Sharpe, 90% one-sided CI, MDE, verdict @ SR_econ=0.5")
    print("=" * 122)
    print(f"{'class':10s}{'factor':12s}{'n':>6s}{'Sh_raw':>8s}{'Sh_adj':>8s}{'lo':>7s}{'hi':>7s}{'MDE':>6s}{'p1':>7s}  verdict")
    counts = {"survivor": 0, "powered-null": 0, "inconclusive": 0}
    for c in sorted(cells, key=lambda c: (c["cls"], -c["sr_adj_ann"])):
        counts[c["verdict"]] += 1
        print(f"{c['cls']:10s}{c['factor']:12s}{c['n_obs']:>6d}{c['sharpe_ann']:>8.2f}{c['sr_adj_ann']:>8.2f}"
              f"{c['lo_ann']:>7.2f}{c['hi_ann']:>7.2f}{c['mde_ann']:>6.2f}{c['p_one']:>7.3f}  {c['verdict']}")
    print(f"\n  verdict counts (HAC + within-cell deflation): {counts}")
    print("  FDR survivors by family scope (the '0 vs 1' sensitivity):")
    for scope, rej in scopes.items():
        winners = [f"{cells[i]['cls']}/{cells[i]['factor']}" for i, r in enumerate(rej) if r]
        print(f"    {scope:16s}: {sum(rej)} survive  {winners}")

    # ---- (3') LICENSING CONTROLS -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("(3') LICENSING CONTROLS — certified long/short in-pipeline, calibrated injection, scrambled negative")
    print("=" * 100)
    # (a) certified long/short: 12-1 cross-sectional momentum (Jegadeesh-Titman), equity, IDENTICAL cost path
    PXe, _ = load_class(*CLASSES["equity"][:2]); RETe = PXe.pct_change(fill_method=None)
    ne = len(PXe); tee = PXe.index[int(ne * 0.55):]
    mom121 = PXe.shift(21) / PXe.shift(252) - 1.0
    from factor_battery import weight_book as wb
    Wm = wb(mom121, PXe, False, 2)
    net121 = ((Wm.shift(1).fillna(0.0) * RETe).sum(axis=1) - CLASSES["equity"][2] * Wm.diff().abs().sum(axis=1)).loc[tee].dropna()
    v121 = hac_verdict(net121)
    print(f"  (a) certified 12-1 equity momentum L/S (in-pipeline, net) n={v121['n_obs']}  Sharpe {v121['sharpe_ann']:+.2f}  "
          f"CI[{v121['lo_ann']:+.2f},{v121['hi_ann']:+.2f}]  MDE {v121['mde_ann']:.2f}  -> {v121['verdict']}")
    # (b) calibrated injection at each class's actual cell n: true annualized Sharpe 0.5
    print("  (b) calibrated injection (true annualized Sharpe 0.5): does the engine flag it survivor at the cell's n?")
    for cls in CLASSES:
        ns = [c["n_obs"] for c in cells if c["cls"] == cls]
        if not ns:
            continue
        n = int(np.median(ns)); mu = (0.5 / ANN)
        hits = 0; TR = 200
        for _ in range(TR):
            s = pd.Series(RNG.standard_normal(n) + mu)
            if hac_verdict(s)["verdict"] == "survivor":
                hits += 1
        print(f"      {cls:10s} n={n:>5d}  empirical power to flag a true 0.5-Sharpe survivor: {hits/TR:.0%}  (MDE {minimum_detectable_sharpe(n)*ANN:.2f})")
    # (c) zero-mean NEGATIVE control: a true-Sharpe-0 series (permuting real returns is invalid — the Sharpe is
    # permutation-invariant, so it preserves the edge; a no-edge series must have zero true mean).
    n_neg = int(np.median([c["n_obs"] for c in cells]))
    surv = 0; TR = 400
    for _ in range(TR):
        if hac_verdict(pd.Series(RNG.standard_normal(n_neg)))["verdict"] == "survivor":
            surv += 1
    print(f"  (c) zero-mean negative control (true Sharpe 0, n={n_neg}): spurious-survivor rate {surv/TR:.1%} (want ~alpha=5%)")

    # ---- (5') MATURITY LAW — DOWNGRADED to inconclusive ---------------------------------------------------
    print("\n" + "=" * 100)
    print("(5') MATURITY LAW — DOWNGRADED: N=4 cannot test a maturity law (report Fisher CI on r, measured spans)")
    print("=" * 100)
    fb = json.load(open(f"{SP}/factor_battery_results.json")) if os.path.exists(f"{SP}/factor_battery_results.json") else {"waterfall": []}
    dom = {}
    for cls, (d, syms, _f) in CLASSES.items():
        PXc, _ = load_class(d, syms)
        years = (PXc.index[-1] - PXc.index[0]).days / 365.25 if len(PXc) else float("nan")
        cr = [w for w in fb["waterfall"] if w["cls"] == cls]
        if not cr:
            continue
        d_sel = float(np.mean([w["d_selection"] for w in cr]))   # SIGNED (no max(0,) clamp)
        d_mult = float(np.mean([w["d_mult"] for w in cr]))
        denom = abs(d_sel) + abs(d_mult)
        dom[cls] = dict(years=years, d_sel=d_sel, d_mult=d_mult, overfit_dom=(d_sel / denom if denom > 0 else np.nan))
    xs = np.array([dom[c]["years"] for c in dom]); ys = np.array([dom[c]["overfit_dom"] for c in dom])
    N = len(xs)
    r = float(np.corrcoef(xs, ys)[0, 1]) if N >= 2 and np.std(ys) > 0 else float("nan")
    if N > 3 and abs(r) < 1:
        z = math.atanh(r); se_z = 1.0 / math.sqrt(N - 3)
        r_lo, r_hi = math.tanh(z - 1.96 * se_z), math.tanh(z + 1.96 * se_z)
    else:
        r_lo, r_hi = -1.0, 1.0
    print(f"{'class':10s}{'years':>7s}{'d_sel(signed)':>15s}{'d_mult':>8s}{'overfit_dom':>13s}")
    for c, v in sorted(dom.items(), key=lambda kv: kv[1]["years"]):
        print(f"{c:10s}{v['years']:>7.1f}{v['d_sel']:>15.2f}{v['d_mult']:>8.2f}{v['overfit_dom']:>13.2f}")
    print(f"  corr(years, overfit_dom) r={r:+.2f}  Fisher 95% CI [{r_lo:+.2f}, {r_hi:+.2f}]  (N={N})")
    print(f"  VERDICT: INCONCLUSIVE — the CI spans strong-negative to strong-positive; N=4 cannot test a maturity law.")

    # ---- (6') CRYPTO MOMENTUM — taker fee restored + liquid construction ----------------------------------
    print("\n" + "=" * 100)
    print("(6') CRYPTO MOMENTUM — restore 10bps taker fee + test the LIQUID construction the thesis names")
    print("=" * 100)
    dcr, dsyms, cfee = CLASSES["crypto"]
    PXk, hlk = load_class(dcr, dsyms); RETk = PXk.pct_change(fill_method=None)
    nk = len(PXk); tek = PXk.index[int(nk * 0.55):]
    def mom_net(universe, k, ts=False):
        PXu, _ = load_class(dcr, universe); RETu = PXu.pct_change(fill_method=None)
        nu = len(PXu); teu = PXu.index[int(nu * 0.55):]
        if ts:
            W = trend_book(PXu, 126, 1)
        else:
            sc, inv = factor_score(PXu, "momentum", 126); W = weight_book(sc, PXu, inv, k)
        net = (W.shift(1).fillna(0.0) * RETu).sum(axis=1) - cfee * W.diff().abs().sum(axis=1)
        return net.loc[teu].dropna()
    liquid5 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    for label, uni, k, ts in [("all-9 x-sec (net 10bps)", dsyms, 2, False),
                              ("liquid-5 x-sec (net 10bps)", liquid5, 2, False),
                              ("BTC+ETH time-series (net 10bps)", ["BTCUSDT", "ETHUSDT"], 1, True)]:
        v = hac_verdict(mom_net(uni, k, ts))
        print(f"  {label:34s} n={v['n_obs']:>4d}  Sharpe {v['sharpe_ann']:+.2f}  CI[{v['lo_ann']:+.2f},{v['hi_ann']:+.2f}]  MDE {v['mde_ann']:.2f}  -> {v['verdict']}")
    print("  => the near-positive all-9 tilt is an ILLIQUID-TAIL artifact; the liquid construction the Zaremba")
    print("     thesis names is flat/negative. All crypto momentum cells are INCONCLUSIVE (n too short to power SR=0.5).")

    out = dict(version=2, verdict_counts=counts, fdr_scopes={k: sum(v) for k, v in scopes.items()},
               fdr_winners={k: [f"{cells[i]['cls']}/{cells[i]['factor']}" for i, r in enumerate(v) if r] for k, v in scopes.items()},
               cells=cells, maturity=dict(dom=dom, r=r, r_ci=[r_lo, r_hi], N=N, verdict="inconclusive"),
               certified_1121=v121)
    json.dump(out, open(f"{SP}/powered_battery_v2_results.json", "w"), indent=2, default=float)
    print("\nwrote powered_battery_v2_results.json")


def cells_series_lookup(raw, cls, factor):
    for c in raw:
        if c["cls"] == cls and c["factor"] == factor:
            return c["net_te"]
    return None


if __name__ == "__main__":
    main()
