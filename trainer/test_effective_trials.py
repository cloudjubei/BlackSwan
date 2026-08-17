"""Unit tests for the effective-number-of-trials estimators. The Deflated Sharpe Ratio deflates the observed
Sharpe by the expected MAXIMUM Sharpe across n INDEPENDENT trials; a machine/LLM search emits thousands of
CORRELATED strategies, so deflating by the raw count over-penalizes and false-negatives genuine alpha. The
effective count corrects this: independent trials -> n_eff ~ M; perfectly redundant -> n_eff ~ 1; k independent
blocks -> n_eff ~ k. Two standard estimators from the multiple-testing literature: eigenvalue participation ratio
and Li-Ji (2005). numpy only; no torch."""
import math

import numpy as np
import pytest

from trainer.effective_trials import (
    effective_trials,
    effective_trials_from_max_sharpe,
    effective_trials_liji,
    effective_trials_participation,
)
from trainer.sharpe import expected_max_sharpe


def _independent(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n) for _ in range(m)]


def _correlated_blocks(k, per_block, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        base = rng.standard_normal(n)
        for _ in range(per_block):
            out.append(base + rng.standard_normal(n) * 0.01)
    return out


def test_independent_recovers_m_participation():
    assert effective_trials(_independent(30, 6000, 1), method="participation") == pytest.approx(30, rel=0.15)


def test_independent_recovers_m_liji():
    assert effective_trials(_independent(30, 6000, 2), method="liji") == pytest.approx(30, rel=0.20)


def test_perfectly_redundant_is_one():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(4000)
    fam = [base + rng.standard_normal(4000) * 1e-6 for _ in range(25)]
    assert effective_trials(fam, method="participation") == pytest.approx(1.0, abs=0.1)
    # Li-Ji's fractional term is ill-behaved when one eigenvalue dominates (M_eff = 1 + frac(lam_max)); it stays
    # small/bounded but is not exactly 1 on non-integer dominant eigenvalues -> participation is the robust choice.
    assert effective_trials(fam, method="liji") <= 2.5


def test_block_structure_recovers_block_count():
    fam = _correlated_blocks(5, 8, 5000, seed=4)  # 5 near-independent blocks of 8 near-duplicates
    assert effective_trials(fam, method="participation") == pytest.approx(5, abs=1.0)


def test_participation_is_bounded_1_to_m():
    fam = _correlated_blocks(3, 5, 3000, seed=5)
    ne = effective_trials(fam, method="participation")
    assert 1.0 <= ne <= len(fam)


def test_more_correlation_lowers_neff():
    indep = effective_trials(_independent(20, 5000, 6), method="participation")
    blocks = effective_trials(_correlated_blocks(4, 5, 5000, seed=6), method="participation")
    assert blocks < indep


def test_single_series_is_one():
    assert effective_trials([np.random.default_rng(7).standard_normal(500)]) == pytest.approx(1.0)


def test_empty_is_zero():
    assert effective_trials([]) == 0.0


def test_participation_from_identity_corr_equals_m():
    assert effective_trials_participation(np.eye(12)) == pytest.approx(12.0, rel=1e-9)


def test_liji_from_identity_corr_equals_m():
    assert effective_trials_liji(np.eye(12)) == pytest.approx(12.0, rel=1e-9)


def test_estimators_from_rank_one_corr_equal_one():
    ones = np.ones((10, 10))
    assert effective_trials_participation(ones) == pytest.approx(1.0, abs=1e-6)
    assert effective_trials_liji(ones) == pytest.approx(1.0, abs=1e-6)


def test_from_max_sharpe_round_trips_expected_max():
    for k in (2, 10, 100, 1000, 50000):
        s = expected_max_sharpe(k, 0.5)
        assert effective_trials_from_max_sharpe(s, 0.5) == pytest.approx(k, rel=0.02)


def test_from_max_sharpe_is_monotone_in_observed():
    a = effective_trials_from_max_sharpe(0.5, 0.5)
    b = effective_trials_from_max_sharpe(1.0, 0.5)
    c = effective_trials_from_max_sharpe(2.0, 0.5)
    assert 1.0 <= a < b < c


def test_from_max_sharpe_below_two_trials_floors_at_one():
    tiny = expected_max_sharpe(2, 0.5) * 0.5
    assert effective_trials_from_max_sharpe(tiny, 0.5) == pytest.approx(1.0, abs=1e-6)


def test_from_max_sharpe_degenerate_std_is_one():
    assert effective_trials_from_max_sharpe(1.0, 0.0) == 1.0
    assert effective_trials_from_max_sharpe(0.0, 0.5) == 1.0


def test_from_max_sharpe_scales_up_for_large_observed():
    k = effective_trials_from_max_sharpe(3.0, 0.3)
    assert np.isfinite(k) and k > 1000


def test_effective_dof_recovers_linear_fit_dimension():
    from trainer.effective_trials import effective_dof_from_sharpe
    # a p-dim in-sample fit on null of length n has R^2 ~ p/n, annualized Sharpe ~ sqrt(ppy*R^2/(1-R^2))
    for p, n in ((10, 1000), (40, 2000), (5, 4000)):
        r2 = p / n
        s_ann = math.sqrt(252.0 * r2 / (1 - r2))
        assert effective_dof_from_sharpe(s_ann, n) == pytest.approx(p, rel=0.02)


def test_effective_dof_monotone_and_bounded():
    from trainer.effective_trials import effective_dof_from_sharpe
    assert effective_dof_from_sharpe(0.0, 2000) == 0.0
    assert effective_dof_from_sharpe(1.0, 2000) < effective_dof_from_sharpe(3.0, 2000)
    assert effective_dof_from_sharpe(1e6, 2000) <= 2000
