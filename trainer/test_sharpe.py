"""Unit tests for the pure Sharpe / Deflated-Sharpe-Ratio module (Bailey & Lopez de Prado).

These are the multiple-testing rigor primitives for the Wave-2 verdict layer: a per-run OOS Sharpe,
the Probabilistic Sharpe Ratio (PSR, prob the true SR exceeds a benchmark given skew/kurtosis/n), the
expected-max-SR deflation level across N trials, the Deflated Sharpe Ratio (DSR = PSR at that level),
and the minimum track-record length. No torch; numpy + scipy only."""
import math

import pytest

from trainer.sharpe import (
    sharpe_ratio,
    sharpe_stats,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    min_track_record_length,
)

# A deterministic positive-drift return series (no RNG) used across the PSR/DSR tests.
POS = [0.01, 0.02, -0.01, 0.015, 0.004, -0.006, 0.018, 0.009, -0.012, 0.011,
       0.007, -0.003, 0.014, 0.002, -0.008, 0.016, 0.006, -0.004, 0.013, 0.005] * 3


# --- sharpe_ratio -----------------------------------------------------------

def test_sharpe_ratio_known_value():
    # [1,2,3,4,5]: mean 3, sample std sqrt(2.5)=1.5811 -> 1.8974.
    assert sharpe_ratio([1, 2, 3, 4, 5]) == pytest.approx(3 / math.sqrt(2.5), rel=1e-9)


def test_sharpe_ratio_zero_mean_is_zero():
    assert sharpe_ratio([1, -1, 1, -1]) == 0.0


def test_sharpe_ratio_zero_variance_returns_zero_not_inf():
    # constant returns -> std 0 -> undefined; return 0.0 (no risk-adjusted info), never inf/nan.
    assert sharpe_ratio([0.01] * 10) == 0.0


def test_sharpe_ratio_too_few_points_is_zero():
    assert sharpe_ratio([0.05]) == 0.0
    assert sharpe_ratio([]) == 0.0


# --- sharpe_stats (the per-run bundle summary.py emits) ---------------------

def test_sharpe_stats_bundle_matches_components():
    s = sharpe_stats(POS)
    assert set(s) == {"sharpe", "skew", "kurtosis", "n_obs"}
    assert s["sharpe"] == pytest.approx(sharpe_ratio(POS), rel=1e-12)
    assert s["n_obs"] == len(POS)
    assert math.isfinite(s["skew"]) and math.isfinite(s["kurtosis"])
    assert s["kurtosis"] > 0  # non-excess (Pearson): normal == 3


def test_sharpe_stats_constant_series_is_safe():
    # A flat equity (no trades) -> constant returns -> defined, normal-shaped defaults, never nan.
    s = sharpe_stats([0.0] * 50)
    assert s == {"sharpe": 0.0, "skew": 0.0, "kurtosis": 3.0, "n_obs": 50}


def test_sharpe_stats_too_few_points_is_safe():
    s = sharpe_stats([0.01])
    assert s["sharpe"] == 0.0 and s["n_obs"] == 1 and math.isfinite(s["skew"])


# --- probabilistic_sharpe_ratio --------------------------------------------

def test_psr_is_half_when_sr_equals_benchmark():
    # symmetric zero-mean returns -> SR=0; PSR vs benchmark 0 -> Phi(0) = 0.5 exactly.
    assert probabilistic_sharpe_ratio([1, -1, 1, -1], sr_benchmark=0.0) == pytest.approx(0.5, abs=1e-9)


def test_psr_in_unit_interval_and_above_half_for_positive_sr():
    psr = probabilistic_sharpe_ratio(POS, sr_benchmark=0.0)
    assert 0.0 <= psr <= 1.0
    assert psr > 0.5  # positive observed Sharpe -> better-than-even that true SR > 0


def test_psr_decreases_as_benchmark_rises():
    low = probabilistic_sharpe_ratio(POS, sr_benchmark=0.0)
    high = probabilistic_sharpe_ratio(POS, sr_benchmark=0.3)
    assert high < low


# --- expected_max_sharpe (the deflation level) ------------------------------

def test_expected_max_sharpe_no_deflation_for_single_trial():
    # 1 trial = no multiple testing -> deflation level 0.
    assert expected_max_sharpe(1, 0.5) == 0.0
    assert expected_max_sharpe(0, 0.5) == 0.0


def test_expected_max_sharpe_zero_when_no_trial_spread():
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_expected_max_sharpe_increases_with_trials():
    e10 = expected_max_sharpe(10, 1.0)
    e100 = expected_max_sharpe(100, 1.0)
    assert 0.0 < e10 < e100  # more configs tried -> higher bar to clear


def test_expected_max_sharpe_scales_with_trial_std():
    assert expected_max_sharpe(50, 2.0) == pytest.approx(2 * expected_max_sharpe(50, 1.0), rel=1e-9)


# --- deflated_sharpe_ratio --------------------------------------------------

def test_dsr_equals_psr_when_single_trial():
    # N=1 -> deflation level 0 -> DSR == PSR(benchmark 0).
    assert deflated_sharpe_ratio(POS, n_trials=1, trial_sr_std=0.5) == pytest.approx(
        probabilistic_sharpe_ratio(POS, sr_benchmark=0.0), rel=1e-9)


def test_dsr_is_lower_than_psr_under_many_trials():
    psr = probabilistic_sharpe_ratio(POS, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(POS, n_trials=100, trial_sr_std=0.5)
    assert dsr < psr  # deflation for 100 trials must reduce confidence


# --- min_track_record_length -----------------------------------------------

def test_mintrl_infinite_when_sr_not_above_benchmark():
    assert min_track_record_length([1, -1, 1, -1], sr_benchmark=0.0) == math.inf


def test_mintrl_finite_and_positive_for_positive_sr():
    n = min_track_record_length(POS, sr_benchmark=0.0, target_prob=0.95)
    assert math.isfinite(n) and n > 0
