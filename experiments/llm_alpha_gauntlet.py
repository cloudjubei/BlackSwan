"""Rigor-infrastructure demonstration for `trainer.certification` (option 3, HONESTLY DEMOTED after adversarial
verification). This is NOT a landmark and NOT LLM-specific: it composes standard, pre-existing primitives (Lo 2002
HAC Sharpe SE; Bailey-Lopez de Prado Deflated-Sharpe vs the expected-max Sharpe for a declared trial count;
Benjamini-Yekutieli FDR) into one honest PASS/FAIL, and the negative it shows (a market-timing zoo dies under
multiplicity) is established literature (Harvey-Liu-Zhu 2016; McLean-Pontiff 2016). The earlier framing claimed
two 'LLM-specific failure modes' -- both were withdrawn: 'hidden multiplicity' IS the exact problem DSR was built
for (DSR catches it here), and 'pretraining-cutoff look-ahead' was generic selection bias that turned out to be
pure market BETA (beta-neutral residual Sharpe ~0). No LLM is involved anywhere.

What it honestly shows on a survivorship-biased 64y equal-weight equity basket (universe/, 50 mega-cap survivors
-> the drift/survivor bias makes the negative only more conservative):
  (1) a 2040-strategy timing zoo is an EFFECTIVE NULL out-of-sample (OOS nominal-sig ~= the 5% false-positive
      rate) and the gauntlet certifies 0;
  (2) EFFECTIVE TRIALS: the zoo is massively redundant (~3-4 independent dims), so raw-n_trials DSR OVER-deflates
      -- the one genuinely non-trivial methodological point (a real 'certification standard' needs an effective-
      trials correction, or it will false-negative genuine alpha in correlated families);
  (3) under-reporting the trial count inflates DSR (textbook DSR-101, shown for honesty, not as a discovery);
  (4) SELECTION BIAS is pure beta: top-by-IS-Sharpe strategies persist OOS only via market exposure; beta-neutral
      residual ~0 -- selecting on Sharpe selects long-bias, not alpha;
  (5) CALIBRATION: injected genuine alpha certifies (realized Sharpe >~1.3) and a permutation null certifies 0, so
      the harness has power and controls false positives. The deflation LEVEL (SR* at DSR=0.5) and the DSR>0.95
      PASS bar are reported separately (they differ ~30%)."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.certification import certify_family  # noqa: E402
from trainer.sharpe import dsr_from_stats, expected_max_sharpe, sharpe_stats  # noqa: E402

UNI = "universe"
ANN = math.sqrt(252.0)
COST = 1e-4
SR_ECON = 0.5


def market_series():
    man = json.load(open(f"{UNI}/manifest.json"))
    rets = {}
    for sym, meta in man.items():
        if meta["class"] != "equity" or meta["n"] < 2000:
            continue
        d = json.load(open(f"{UNI}/{sym}.json"))["data"]
        dates = sorted(d)
        for i in range(1, len(dates)):
            c0, c1 = d[dates[i - 1]][3], d[dates[i]][3]
            if c0 and c1 and c0 > 0:
                rets.setdefault(dates[i], []).append(c1 / c0 - 1.0)
    dates = sorted(rets)
    return dates, np.array([np.mean(rets[dt]) for dt in dates], dtype=float)


def _pos_to_returns(pos, m):
    pos = np.clip(pos, -1.0, 1.0)
    held = np.concatenate([[0.0], pos[:-1]])
    turn = np.abs(np.diff(held, prepend=0.0))
    return held * m - COST * turn, held


def strategy_zoo(m):
    n = len(m)
    price = np.cumprod(1.0 + np.nan_to_num(m))
    strat, expo = {}, {}

    def sma(a, w):
        c = np.cumsum(np.insert(a, 0, 0.0))
        out = np.full(len(a), np.nan)
        out[w - 1:] = (c[w:] - c[:-w]) / w
        return out

    def add(name, pos):
        r, held = _pos_to_returns(np.nan_to_num(pos), m)
        strat[name] = r
        expo[name] = held
    for fast in (5, 10, 20, 50):
        for slow in (50, 100, 150, 200):
            if fast < slow:
                add(f"xover_{fast}_{slow}", np.sign(sma(price, fast) - sma(price, slow)))
    for L in (5, 10, 20, 60, 120, 250):
        tr = np.full(n, np.nan)
        tr[L:] = price[L:] / price[:-L] - 1.0
        add(f"tsmom_{L}", np.sign(tr))
        add(f"tsrev_{L}", -np.sign(tr))
    for L in (20, 50, 100, 200):
        rmax = np.full(n, np.nan)
        rmin = np.full(n, np.nan)
        for t in range(L, n):
            rmax[t] = price[t - L:t].max()
            rmin[t] = price[t - L:t].min()
        add(f"donchian_{L}", np.where(price > rmax, 1.0, np.where(price < rmin, -1.0, 0.0)))
    rng = np.random.default_rng(0)
    for j in range(2000):
        L = int(rng.integers(3, 250))
        thr = float(rng.normal(0, 0.8))
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        tr = np.full(n, np.nan)
        tr[L:] = price[L:] / price[:-L] - 1.0
        z = np.full(n, np.nan)
        for t in range(L + 20, n):
            seg = tr[t - 20:t]
            s = np.nanstd(seg)
            if s > 0:
                z[t] = (tr[t] - np.nanmean(seg)) / s
        add(f"rand_{j}", sgn * np.where(z > thr, 1.0, np.where(z < -thr, -1.0, 0.0)))
    return strat, expo


def effective_trials(oos_matrix):
    x = oos_matrix - oos_matrix.mean(axis=0, keepdims=True)
    sd = x.std(axis=0)
    keep = sd > 0
    x = x[:, keep] / sd[keep]
    c = (x.T @ x) / x.shape[0]
    ev = np.linalg.eigvalsh(c)
    ev = ev[ev > 0]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def main():
    dates, m = market_series()
    print(f"market: survivorship-biased equal-weight equity, {len(m)} days {dates[0]}..{dates[-1]}, "
          f"Sharpe(ann)={sharpe_stats(m)['sharpe']*ANN:.2f}")
    zoo, expo = strategy_zoo(m)
    names = list(zoo)
    split = int(len(m) * 0.7)
    oos_mat = np.column_stack([zoo[k][split:] for k in names])
    per, fam = certify_family([zoo[k][split:] for k in names], n_trials=len(names))
    is_sh = {k: sharpe_stats(zoo[k][:split])["sharpe"] * ANN for k in names}
    looks_is = sum(1 for k in names if is_sh[k] > 0.5)
    oos_sh = {k: sharpe_stats(zoo[k][split:])["sharpe"] * ANN for k in names}
    print(f"zoo: {len(names)} timing strategies; whole-zoo OOS mean Sharpe={np.mean(list(oos_sh.values())):+.2f}\n")

    exp_null = 0.05 * len(names)
    print("=== (1) TIMING ZOO IS AN EFFECTIVE NULL OOS (three INDEPENDENT counts, not a funnel) ===")
    print(f"  look profitable IN-SAMPLE (Sharpe>0.5):  {looks_is}/{len(names)} ({100*looks_is/len(names):.0f}%)  [mostly beta]")
    print(f"  OOS nominally significant (p<0.05):      {fam['n_nominal_sig']} "
          f"({100*fam['n_nominal_sig']/len(names):.1f}%)  vs pure-null expectation ~{exp_null:.0f}")
    print(f"  OOS economically sig (HAC-lo>{SR_ECON}) / DSR / BY: "
          f"{fam['n_econ_pass']} / {fam['n_dsr_pass']} / {fam['n_by_reject']}")
    print(f"  CERTIFIED (all three): {fam['n_certified']} ({100*fam['certified_rate']:.2f}%)")

    n_eff = effective_trials(oos_mat)
    print(f"\n=== (2) EFFECTIVE TRIALS (the real point: raw n_trials over-deflates a redundant zoo) ===")
    print(f"  nominal trials = {len(names)}; effective independent dims (eigenvalue participation ratio) = {n_eff:.1f}")
    print(f"  => DSR should deflate by ~{n_eff:.0f}, not {len(names)}; raw-N deflation false-negatives correlated alpha")

    bi = int(np.argmax([p["sharpe_ann"] for p in per]))
    st = sharpe_stats(zoo[names[bi]][split:])
    tsd = fam["trial_sr_std"]
    print(f"\n=== (3) UNDER-REPORTING TRIALS INFLATES DSR (textbook DSR, for honesty) -- best OOS strategy ===")
    for claimed in (1, 10, 100, round(n_eff), len(names)):
        dsr = dsr_from_stats(st["sharpe"], st["skew"], st["kurtosis"], st["n_obs"], claimed, tsd)
        tag = {1: " (naive under-report)", round(n_eff): " (~effective)", len(names): " (raw-N, over-deflates)"}.get(claimed, "")
        print(f"  claimed n_trials={claimed:>5}: DSR={dsr:.3f}{tag}")

    print(f"\n=== (4) SELECTION BIAS IS PURE BETA (top-by-IS-Sharpe persist OOS only via market exposure) ===")
    top = sorted(names, key=lambda k: -is_sh[k])[:20]
    mo = m[split:]
    var_m = mo.var()
    raw_sh, res_sh, betas = [], [], []
    for k in top:
        rs = zoo[k][split:]
        beta = np.cov(rs, mo)[0, 1] / var_m
        resid = rs - beta * mo
        raw_sh.append(sharpe_stats(rs)["sharpe"] * ANN)
        res_sh.append(sharpe_stats(resid)["sharpe"] * ANN)
        betas.append(beta)
    print(f"  top-20 by IS Sharpe -> OOS raw Sharpe mean={np.mean(raw_sh):+.2f} (beta mean={np.mean(betas):+.2f}); "
          f"BETA-NEUTRAL residual Sharpe mean={np.mean(res_sh):+.2f}")
    print(f"  => the OOS 'persistence' is market beta, not alpha; certified among top-20: "
          f"{sum(certify_family([zoo[k][split:]-np.cov(zoo[k][split:],mo)[0,1]/var_m*mo], n_trials=len(names))[1]['n_certified'] for k in top)}/20 (beta-neutral)")

    print(f"\n=== (5) CALIBRATION (power + false-positive control) ===")
    n_oos = len(m) - split
    vol = float(np.std(m[split:]))
    inj = {sh: (sh / ANN) * vol + np.random.default_rng(100 + int(sh * 10)).standard_normal(n_oos) * vol
           for sh in (0.8, 1.5, 2.5)}
    per_i, _ = certify_family([zoo[k][split:] for k in names] + list(inj.values()), n_trials=len(names) + len(inj))
    for i, sh in enumerate(inj):
        p = per_i[len(names) + i]
        print(f"  injected true Sharpe {sh}: realized={p['sharpe_ann']:.2f} dsr={p['dsr']:.3f} certified={p['certified']}")
    sr_star = expected_max_sharpe(len(names), tsd) * ANN
    print(f"  deflation LEVEL SR* (DSR=0.5) = {sr_star:.2f} ann; DSR>0.95 PASS bar ~ SR*+1.645*SE "
          f"(~{sr_star + 1.645*tsd*ANN/math.sqrt(1):.2f} for a clean series)")
    m_perm = m.copy()
    np.random.default_rng(1).shuffle(m_perm)
    zoo_null, _ = strategy_zoo(m_perm)
    _, fam_null = certify_family([v[split:] for v in zoo_null.values()], n_trials=len(zoo_null))
    print(f"  PERMUTATION null (returns shuffled): certified={fam_null['n_certified']}/{len(zoo_null)} "
          f"(near-vacuous -- the real zoo is also ~null; the informative control is the positive one)")

    print(f"\n=== (6) EFFECTIVE-TRIALS RESCUE (one genuine idea proposed as many CORRELATED variants) ===")

    def pinned_alpha(target_sharpe, seed):
        r = np.random.default_rng(seed).standard_normal(n_oos)
        r = (r - r.mean()) / r.std()
        return r * vol + (target_sharpe / ANN) * vol

    core = pinned_alpha(1.0, 500)
    variants = [core + np.random.default_rng(600 + i).standard_normal(n_oos) * vol * 0.10 for i in range(25)]
    fam6 = [zoo[k][split:] for k in names] + variants
    per_raw, f_raw = certify_family(fam6, n_trials=None)
    per_eff, f_eff = certify_family(fam6, n_trials="effective")
    v_sh = [per_eff[len(names) + i]["sharpe_ann"] for i in range(len(variants))]
    cert_raw = sum(per_raw[len(names) + i]["certified"] for i in range(len(variants)))
    cert_eff = sum(per_eff[len(names) + i]["certified"] for i in range(len(variants)))
    print(f"  25 correlated variants of ONE genuine idea (realized Sharpe {min(v_sh):.2f}-{max(v_sh):.2f}) in the "
          f"null zoo; family effective trials={f_eff['effective_trials']:.1f} vs raw {f_raw['n_trials']}")
    print(f"  deflation bar: raw-N ~{expected_max_sharpe(f_raw['n_trials'], f_raw['trial_sr_std'])*ANN:.2f} vs "
          f"effective-N ~{expected_max_sharpe(f_eff['n_trials'], f_eff['trial_sr_std'])*ANN:.2f} ann (SR* level)")
    print(f"  certified under RAW-N deflation:       {cert_raw}/25  (raw N over-deflates -> false-negative)")
    print(f"  certified under EFFECTIVE-N deflation: {cert_eff}/25  (correct: ~one idea, not {len(fam6)} trials)")

    summary = {"n_days": len(m), "zoo_size": len(names), "look_good_is": looks_is,
               "rescue_raw_certified": cert_raw, "rescue_eff_certified": cert_eff,
               "rescue_effective_trials": f_eff["effective_trials"],
               "oos_nominal_sig": fam["n_nominal_sig"], "null_expectation": exp_null,
               "n_certified": fam["n_certified"], "effective_trials": n_eff,
               "selection_raw_sharpe": float(np.mean(raw_sh)), "selection_residual_sharpe": float(np.mean(res_sh)),
               "sr_star_ann": sr_star, "calib_injected": {str(sh): per_i[len(names) + i]["certified"]
                                                          for i, sh in enumerate(inj)},
               "calib_permutation_certified": fam_null["n_certified"]}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/llm_alpha_gauntlet.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/llm_alpha_gauntlet.json")


if __name__ == "__main__":
    main()
