"""REFEREE candidate #5 -- statistical deflation of fresh self-graded agentic-alpha headlines (re-analysis of the
papers' OWN reported numbers, verified from the abstracts). Agora ('AI Trading's Alpha Singularity', arXiv
2606.29194): holdout Sharpe +1.87 on a SINGLE 91-day sealed holdout, single-seed, no deflation (its own best
baseline is -0.755 cross-seed mean). AgonAlpha (arXiv 2608.11250): Sharpe 3.48 'SPECTACULAR-grade' across five
users x six backends on WorldQuant BRAIN, no multiple-testing correction reported. We apply the standard
finite-sample and expected-max-under-null arithmetic. Demonstration script."""
import math
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.sharpe import expected_max_sharpe  # noqa: E402


def t_of_sharpe(sr_ann, n_days, ppy=252.0):
    t = sr_ann * math.sqrt(n_days / ppy)
    return t, float(norm.sf(t))


def main():
    print("=== Agora (2606.29194): +1.87 Sharpe on a single 91-day holdout, single-seed ===")
    t, p = t_of_sharpe(1.87, 91)
    print(f"  finite-sample t-stat of an ANNUALISED Sharpe 1.87 over 91 trading days: t = {t:.2f}, one-sided p = {p:.3f}")
    print(f"  -> NOT significant (needs t>=1.65). A 91-day Sharpe of 1.87 is statistically indistinguishable from 0;")
    print(f"     the reported edge is within one holdout's noise, and the paper's own baseline is -0.755 cross-seed.\n")

    print("=== AgonAlpha (2608.11250): Sharpe 3.48, no trial count and no deflation, across >=30 deployment cells ===")
    print("  what search size K reaches E[max Sharpe]=3.48 under the null, by evaluation length?")
    print(f"  {'eval yrs':>8} {'null SR-SE':>11} {'K for 3.48':>12}")
    for yrs in (0.5, 1.0, 2.0, 5.0):
        sr_se = math.sqrt(1.0 / yrs)
        k = 2
        while k < 10 ** 9 and expected_max_sharpe(k, sr_se) < 3.48:
            k = int(k * 1.5) + 1
        shown = f"{k:.0e}" if k < 10 ** 9 else ">1e9"
        print(f"  {yrs:>8.1f} {sr_se:>11.2f} {shown:>12}")
    print("  -> on a SHORT evaluation (0.5-1 yr, common for freshly-mined alphas) a modest undeflated search")
    print("     reaches 3.48 by luck; on a long (5 yr) evaluation it does not. The paper reports NEITHER the")
    print("     evaluation length NOR the search size NOR a deflated Sharpe -- so 3.48 cannot be assessed as")
    print("     discovery vs expected-max-of-search. That non-disclosure IS the finding (the DSR standard exists")
    print("     precisely to discharge this burden).\n")

    print("VERDICT: Agora's +1.87 is decisively insignificant (91-day t~1.1); AgonAlpha's 3.48 is UNVERIFIABLE as")
    print("reported (no trials, no deflation, no eval length). Both are apparatus-hardening referee points, not")
    print("landmark refutations -- exactly the 'incremental' caliber the survey pre-registered.")


if __name__ == "__main__":
    main()
