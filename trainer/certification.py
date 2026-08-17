"""The machine-discovered-alpha certification gauntlet (option 3: 'Deflated Sharpe for the LLM era'). Given a
FAMILY of out-of-sample strategy return series -- the kind an LLM agent or an automated search emits by the
thousand -- it composes the tested rigor primitives in sharpe.py into a single honest PASS/FAIL per strategy and
a family-level survival count, so 'looks profitable' can be separated from 'survives multiplicity + serial
correlation + an out-of-sample bound'. A strategy is CERTIFIED only if it clears ALL of:
  - ECONOMIC: a one-sided HAC (serial-correlation-robust) lower Sharpe bound above `sr_econ_ann` (not merely > 0),
  - MULTIPLICITY: a Deflated Sharpe Ratio > `dsr_threshold` against the expected-max Sharpe for the TRUE trials,
  - FDR: Benjamini-Yekutieli (the arbitrary-dependence bound) across the family.
`n_trials` is the honest multiplicity: pass the real search breadth (often >> len(family)); the default assumes
each supplied series was one trial. numpy + scipy only; no torch."""
import math

import numpy as np
from scipy.stats import norm

from trainer.effective_trials import effective_trials
from trainer.sharpe import (
    benjamini_hochberg,
    benjamini_yekutieli,
    dsr_from_stats,
    sharpe_standard_error_hac,
    sharpe_stats,
)


def certify_family(returns_list, n_trials=None, sr_econ_ann=0.5, alpha=0.05, periods_per_year=252.0,
                   dsr_threshold=0.95):
    """Certify a family of OOS strategy return series. Returns (per_strategy, family). `n_trials` sets the
    multiplicity used for DSR deflation: an int overrides it, None uses the raw strategy count, and the string
    'effective' uses the correlation-corrected effective number of independent trials (the honest choice for a
    redundant automated/LLM search -- raw-count deflation over-penalizes correlated families). The family dict
    always reports `effective_trials` regardless."""
    ann = math.sqrt(periods_per_year)
    series = [np.asarray(r, dtype=float) for r in returns_list]
    m = len(series)
    if m == 0:
        return [], {"n_strategies": 0, "n_trials": 0, "effective_trials": 0.0, "trial_sr_std": 0.0,
                    "n_nominal_sig": 0, "n_econ_pass": 0, "n_dsr_pass": 0, "n_bh_reject": 0, "n_by_reject": 0,
                    "n_certified": 0, "certified_rate": 0.0}
    stats = [sharpe_stats(r) for r in series]
    sharpes = np.array([s["sharpe"] for s in stats], dtype=float)
    trial_sr_std = float(np.std(sharpes, ddof=1)) if m > 1 else 0.0
    n_eff = effective_trials(series) if m > 1 else 1.0
    if n_trials == "effective":
        trials = max(1, int(round(n_eff)))
    elif n_trials is not None:
        trials = int(n_trials)
    else:
        trials = m
    sr_econ = sr_econ_ann / ann
    za = float(norm.ppf(1.0 - alpha))

    per = []
    for r, st in zip(series, stats):
        se = sharpe_standard_error_hac(r)
        if np.isfinite(se) and se > 0:
            hac_lower = st["sharpe"] - za * se
            p_one = float(norm.sf(st["sharpe"] / se))
        else:
            hac_lower = -math.inf
            p_one = 1.0
        dsr = dsr_from_stats(st["sharpe"], st["skew"], st["kurtosis"], st["n_obs"], trials, trial_sr_std)
        per.append({"sharpe_ann": st["sharpe"] * ann, "n_obs": st["n_obs"], "p_one": p_one, "dsr": dsr,
                    "hac_lower_ann": hac_lower * ann if np.isfinite(hac_lower) else -math.inf,
                    "nominal_sig": p_one < alpha})

    pvals = [p["p_one"] for p in per]
    bh = benjamini_hochberg(pvals, q=alpha)
    by = benjamini_yekutieli(pvals, q=alpha)
    for i, p in enumerate(per):
        p["bh"] = bool(bh[i])
        p["by"] = bool(by[i])
        p["dsr_pass"] = p["dsr"] > dsr_threshold
        p["econ_pass"] = bool(p["hac_lower_ann"] > sr_econ_ann)
        p["certified"] = bool(p["econ_pass"] and p["dsr_pass"] and p["by"])

    n_cert = sum(p["certified"] for p in per)
    family = {"n_strategies": m, "n_trials": trials, "effective_trials": n_eff, "trial_sr_std": trial_sr_std,
              "n_nominal_sig": sum(p["nominal_sig"] for p in per),
              "n_econ_pass": sum(p["econ_pass"] for p in per),
              "n_dsr_pass": sum(p["dsr_pass"] for p in per),
              "n_bh_reject": int(sum(bh)), "n_by_reject": int(sum(by)),
              "n_certified": n_cert, "certified_rate": n_cert / m}
    return per, family
