"""The unified ML-trading honesty gauntlet -- 'SPIVA / Replicating-Anomalies for ML-trading'. Given a FAMILY of
out-of-sample net-return series (one claimed strategy, or the whole zoo an automated/LLM search emits) it composes
every rigor primitive into ONE survive/fail verdict per strategy, so a headline Sharpe is separated from a
discovery. A strategy SURVIVES only if it clears ALL of:
  - ECONOMIC + MULTIPLICITY + FDR: the certify_family gauntlet (HAC lower Sharpe bound > sr_econ_ann; Deflated
    Sharpe > threshold against the TRUE trial count; Benjamini-Yekutieli across the family),
  - RANDOM-FORMULA NULL (optional): its Sharpe beats matched-complexity random formulas (empirical p < alpha),
  - POST-CUTOFF (optional): its edge persists strictly out-of-sample past a cutoff (HAC lower bound still > econ).
The null and post-cutoff gates auto-pass when their inputs are omitted, so with neither the verdict reduces exactly
to certify_family. Feed net-of-cost returns (see trading_costs); the null is built from net-of-cost random formulas
(see random_formula_null) so the whole verdict is charged for cost and search. numpy + scipy only."""
import math

import numpy as np
from scipy.stats import norm

from trainer.certification import certify_family
from trainer.random_formula_null import null_pvalue
from trainer.sharpe import sharpe_standard_error_hac, sharpe_stats


def _hac_lower_ann(returns, za, ann):
    r = np.asarray(returns, dtype=float)
    if r.size < 3:
        return -math.inf
    st = sharpe_stats(r)
    se = sharpe_standard_error_hac(r)
    if not (np.isfinite(se) and se > 0):
        return -math.inf
    return (st["sharpe"] - za * se) * ann


def run_gauntlet(returns_list, n_trials=None, null_sharpes=None, cutoff=None, sr_econ_ann=0.0, alpha=0.05,
                 periods_per_year=252.0, dsr_threshold=0.95):
    """Adjudicate a family of OOS net-return series. Returns (per_strategy, family). Each per-strategy dict extends
    certify_family's with `null_p`/`null_pass`, `postcutoff_sharpe_ann`/`postcutoff_pass`, and the combined
    `survives`. The family dict adds `n_survived` and `survival_rate`. `null_sharpes` is a matched-complexity
    random-formula null (annualised Sharpes); `cutoff` is the index at which the strictly-out-of-sample tail
    begins."""
    per, family = certify_family(returns_list, n_trials=n_trials, sr_econ_ann=sr_econ_ann, alpha=alpha,
                                 periods_per_year=periods_per_year, dsr_threshold=dsr_threshold)
    ann = math.sqrt(periods_per_year)
    za = float(norm.ppf(1.0 - alpha))
    series = [np.asarray(r, dtype=float) for r in returns_list]

    for p, r in zip(per, series):
        if null_sharpes is not None and np.asarray(null_sharpes).size > 0:
            p["null_p"] = null_pvalue(p["sharpe_ann"], null_sharpes)
            p["null_pass"] = bool(p["null_p"] < alpha)
        else:
            p["null_p"] = None
            p["null_pass"] = True

        if cutoff is not None:
            lower = _hac_lower_ann(r[int(cutoff):], za, ann)
            p["postcutoff_sharpe_ann"] = lower
            p["postcutoff_pass"] = bool(lower > sr_econ_ann)
        else:
            p["postcutoff_sharpe_ann"] = None
            p["postcutoff_pass"] = True

        p["survives"] = bool(p["certified"] and p["null_pass"] and p["postcutoff_pass"])

    n_surv = sum(p["survives"] for p in per)
    family["n_survived"] = n_surv
    family["survival_rate"] = (n_surv / len(per)) if per else 0.0
    return per, family
