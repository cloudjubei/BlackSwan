"""Unit tests for the pure Sharpe / Deflated-Sharpe-Ratio module (Bailey & Lopez de Prado).

These are the multiple-testing rigor primitives for the Wave-2 verdict layer: a per-run OOS Sharpe,
the Probabilistic Sharpe Ratio (PSR, prob the true SR exceeds a benchmark given skew/kurtosis/n), the
expected-max-SR deflation level across N trials, the Deflated Sharpe Ratio (DSR = PSR at that level),
and the minimum track-record length. No torch; numpy + scipy only."""
import math
from functools import lru_cache

import numpy as np
import pytest

from trainer.sharpe import (
    sharpe_ratio,
    sharpe_stats,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    min_track_record_length,
    psr_from_stats,
    dsr_from_stats,
    min_track_record_length_from_stats,
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


# --- stats-input variants (A4.3): PSR/DSR/minTRL from a precomputed (sharpe,skew,kurtosis,n) bundle ---------
# These are the entry points the modeltrainer engine's TS Deflated-Sharpe port mirrors — it feeds the per-run
# oos_sharpe/oos_ret_skew/oos_ret_kurt/oos_n_obs it already stores, never a raw return array. Golden values
# below are the cross-language pin for that port.

def test_psr_from_stats_matches_the_array_form():
    s = sharpe_stats(POS)
    for bench in (0.0, 0.3):
        assert psr_from_stats(s["sharpe"], s["skew"], s["kurtosis"], s["n_obs"], sr_benchmark=bench) == pytest.approx(
            probabilistic_sharpe_ratio(POS, sr_benchmark=bench), rel=1e-12)


def test_dsr_from_stats_matches_the_array_form():
    s = sharpe_stats(POS)
    assert dsr_from_stats(
        s["sharpe"], s["skew"], s["kurtosis"], s["n_obs"], n_trials=100, trial_sr_std=0.5
    ) == pytest.approx(deflated_sharpe_ratio(POS, n_trials=100, trial_sr_std=0.5), rel=1e-12)


def test_mintrl_from_stats_matches_the_array_form():
    s = sharpe_stats(POS)
    assert min_track_record_length_from_stats(
        s["sharpe"], s["skew"], s["kurtosis"], s["n_obs"]
    ) == pytest.approx(min_track_record_length(POS), rel=1e-12)
    assert min_track_record_length_from_stats(0.0, 0.0, 3.0, 60) == math.inf  # SR not above benchmark
    assert psr_from_stats(0.5, 0.0, 3.0, 1) == 0.0  # n<2 undefined


def test_psr_from_stats_golden_vectors():
    # Exact closed-form pins (kurtosis NON-excess; PSR denom = 1 - g3·SR + (g4-1)/4·SR²). The engine's TS port
    # MUST reproduce these to 1e-9 — a drift in either language fails. Regenerate from this module if the
    # formula ever legitimately changes.
    assert psr_from_stats(0.0, 0.0, 3.0, 100) == pytest.approx(0.5, abs=1e-12)  # SR=benchmark ⇒ Phi(0)
    assert psr_from_stats(0.1, 0.0, 3.0, 101) == pytest.approx(0.8407413278013518, rel=1e-9)  # normal moments
    assert psr_from_stats(0.15, -0.5, 4.0, 200, sr_benchmark=0.05) == pytest.approx(0.911495153669269, rel=1e-9)


# --- ANTI-VACUITY (L3): the gate must REJECT a search that contains no real edge ----------------------------
# The leak register pins this acceptance test verbatim: "K null runs (true edge 0) -> top DSR ~= 0.5, gate
# rejects". Without it a gate that rubber-stamps every candidate is indistinguishable from one that works —
# right up until it certifies noise as a champion. Every generator below is explicitly seeded: these are
# correctness gates, and a flaky correctness gate gets deleted by whoever hits it on a bad day.

NULL_TRIALS = 500     # K configs in one simulated search
NULL_N_OBS = 252      # per-trial test window (a year of daily bars)
NULL_SIGMA = 0.01     # per-observation return vol; the level is irrelevant — Sharpe is scale-free
NULL_SEED = 20260806
NULL_CORPORA = 64     # independent searches replayed for the distribution-level claims
EDGE_SEED = 20260807
EDGE_MU = 0.006       # true per-observation SR = EDGE_MU / NULL_SIGMA = 0.6, clear of the ~0.30 bar below
DSR_GATE = 0.95       # the verdict layer's pass threshold
# EDGE_MU must sit well ABOVE the sample Sharpe the gate actually demands at NULL_TRIALS/NULL_N_OBS, which
# test_the_gate_demands_a_knowable_sample_sharpe pins at ~0.30/observation. A true SR only ~1.6 sampling
# errors above that bar (EDGE_MU = 0.004) fails the gate on 6% of equally valid draws — the survival arm then
# proves nothing but the seed. 0.6 is ~4.7 sampling errors clear, so survival is a property of the drift.


@lru_cache(maxsize=None)
def _null_search(seed):
    """One complete null SEARCH: NULL_TRIALS independent return series whose TRUE edge is exactly zero
    (loc=0.0), i.e. the eight-consecutive-nulls situation. Returns the moment bundle of the BEST trial —
    what a researcher reports — plus the cross-trial Sharpe std the deflation level is built from."""
    rng = np.random.default_rng(seed)
    trials = rng.normal(0.0, NULL_SIGMA, size=(NULL_TRIALS, NULL_N_OBS))
    trial_sharpes = np.array([sharpe_ratio(t) for t in trials])
    winner = sharpe_stats(trials[int(np.argmax(trial_sharpes))])
    return winner, float(trial_sharpes.std(ddof=1))


def _dsr_of(bundle, n_trials, trial_sr_std):
    return dsr_from_stats(
        bundle["sharpe"], bundle["skew"], bundle["kurtosis"], bundle["n_obs"],
        n_trials=n_trials, trial_sr_std=trial_sr_std,
    )


def _psr_of(bundle, sr_benchmark=0.0):
    return psr_from_stats(
        bundle["sharpe"], bundle["skew"], bundle["kurtosis"], bundle["n_obs"], sr_benchmark=sr_benchmark
    )


def test_null_search_winner_looks_like_a_find():
    # Precondition for the whole section: the null winner is NOT a weak candidate. Best-of-500 under the
    # null lands a healthy positive Sharpe — this is exactly the number a search reports as a lead.
    winner, _ = _null_search(NULL_SEED)
    assert winner["sharpe"] > 0.15
    assert winner["n_obs"] == NULL_N_OBS


def test_null_search_top_dsr_is_a_coin_flip_and_the_gate_rejects_it():
    # THE headline property. True edge zero everywhere ⇒ the best of K trials is worth a coin flip, and the
    # gate must say so.
    winner, trial_sr_std = _null_search(NULL_SEED)
    dsr = _dsr_of(winner, NULL_TRIALS, trial_sr_std)
    # Band, not a knife edge: SR* is the EXPECTED max of K null Sharpes while the REALISED max scatters
    # around it (asymptotically Gumbel, sd ≈ (π/√6)/√(2·ln K) ≈ 0.36 in the z-units DSR = Phi(z) consumes).
    # So a single null search lands near 0.5 with roughly that spread; ±0.35 is about one such sd and stays
    # far from the gate. The mean over NULL_CORPORA searches pins the 0.5 centre tightly.
    assert abs(dsr - 0.5) <= 0.35
    assert dsr < DSR_GATE  # gate rejects: DSR >= 0.95 is the pass condition


def test_null_search_top_dsr_centres_on_half_and_the_gate_rejects_every_replication():
    # Distribution-level version of the headline: NULL_CORPORA independent searches, all pure noise.
    tops = np.array([
        _dsr_of(winner, NULL_TRIALS, trial_sr_std)
        for winner, trial_sr_std in (_null_search(NULL_SEED + i) for i in range(NULL_CORPORA))
    ])
    assert tops.mean() == pytest.approx(0.5, abs=0.05)  # theory says exactly 0.5; SEM over 64 searches ≈ 0.017
    assert tops.max() < DSR_GATE  # not one null search in NULL_CORPORA is certified


def test_deflation_alone_flips_the_null_winner_from_pass_to_reject():
    # THE CRUX. The rejection above must come from the multiple-testing correction, not from the candidate
    # being weak. Judged as a single hypothesis the very same winner sails through the same threshold.
    winner, trial_sr_std = _null_search(NULL_SEED)
    undeflated = _psr_of(winner, sr_benchmark=0.0)
    assert undeflated >= DSR_GATE  # plain PSR vs benchmark 0: PASSES

    # Two independent ways to switch the deflation off, both landing on that same passing number: one trial
    # (no multiple testing) and zero cross-trial spread. Only the SR* term differs from the rejecting call.
    assert _dsr_of(winner, 1, trial_sr_std) == pytest.approx(undeflated, rel=1e-12)
    assert _dsr_of(winner, NULL_TRIALS, 0.0) == pytest.approx(undeflated, rel=1e-12)
    assert _dsr_of(winner, 1, trial_sr_std) >= DSR_GATE

    deflated = _dsr_of(winner, NULL_TRIALS, trial_sr_std)
    assert deflated < DSR_GATE
    # and the deflation level is the only thing that moved: a strictly positive SR* is the whole difference
    # between the passing calls above and this rejecting one.
    assert expected_max_sharpe(NULL_TRIALS, trial_sr_std) > 0
    assert deflated < undeflated


def test_every_null_replication_passes_undeflated_and_fails_deflated():
    # The flip is not a property of one lucky seed: NULL_CORPORA out of NULL_CORPORA searches pass without
    # deflation and fail with it.
    passed_undeflated = 0
    rejected_deflated = 0
    for i in range(NULL_CORPORA):
        winner, trial_sr_std = _null_search(NULL_SEED + i)
        passed_undeflated += _psr_of(winner) >= DSR_GATE
        rejected_deflated += _dsr_of(winner, NULL_TRIALS, trial_sr_std) < DSR_GATE
    assert passed_undeflated == NULL_CORPORA
    assert rejected_deflated == NULL_CORPORA


def test_null_winner_has_no_track_record_long_enough_to_clear_the_deflation_level():
    # The minTRL reading of the same verdict: to certify the null winner against SR* you would need more
    # observations than it has (inf when its Sharpe never clears SR* at all).
    winner, trial_sr_std = _null_search(NULL_SEED)
    sr_star = expected_max_sharpe(NULL_TRIALS, trial_sr_std)
    need = min_track_record_length_from_stats(
        winner["sharpe"], winner["skew"], winner["kurtosis"], winner["n_obs"], sr_benchmark=sr_star
    )
    assert need > winner["n_obs"]


def test_a_genuine_edge_survives_the_same_deflation():
    # A gate that rejects everything is as useless as one that accepts everything. Same K, same trial_sr_std,
    # same window length as the null search — only the true drift differs.
    _, trial_sr_std = _null_search(NULL_SEED)
    edge = sharpe_stats(np.random.default_rng(EDGE_SEED).normal(EDGE_MU, NULL_SIGMA, size=NULL_N_OBS))
    assert _dsr_of(edge, NULL_TRIALS, trial_sr_std) >= DSR_GATE
    # and it clears the deflation level with observations to spare, unlike the null winner.
    assert min_track_record_length_from_stats(
        edge["sharpe"], edge["skew"], edge["kurtosis"], edge["n_obs"],
        sr_benchmark=expected_max_sharpe(NULL_TRIALS, trial_sr_std),
    ) <= edge["n_obs"]


def test_a_genuine_edge_survives_the_same_deflation_in_every_replication():
    # Survival must not rest on one lucky draw any more than rejection does: NULL_CORPORA independent series
    # carrying the same true drift all clear the gate, each with a track record long enough to prove it.
    _, trial_sr_std = _null_search(NULL_SEED)
    sr_star = expected_max_sharpe(NULL_TRIALS, trial_sr_std)
    survived = 0
    for i in range(NULL_CORPORA):
        edge = sharpe_stats(np.random.default_rng(EDGE_SEED + i).normal(EDGE_MU, NULL_SIGMA, size=NULL_N_OBS))
        survived += (
            _dsr_of(edge, NULL_TRIALS, trial_sr_std) >= DSR_GATE
            and min_track_record_length_from_stats(
                edge["sharpe"], edge["skew"], edge["kurtosis"], edge["n_obs"], sr_benchmark=sr_star
            ) <= edge["n_obs"]
        )
    assert survived == NULL_CORPORA


def test_the_gate_demands_a_knowable_sample_sharpe():
    # What the deflation actually costs a candidate, stated rather than implied: at NULL_TRIALS configs over
    # NULL_N_OBS observations the gate passes nothing below ~0.30 per-observation Sharpe (~4.8 annualised) and
    # passes normal-shaped returns above it. Bracketed loosely — SR* moves with the search's own trial spread
    # (observed bar 0.284-0.318 over 300 null searches) — but tightly enough to keep the two arms honest: the
    # null winner (~0.17) sits below the bar and EDGE_MU / NULL_SIGMA sits above it.
    winner, trial_sr_std = _null_search(NULL_SEED)
    sr_star = expected_max_sharpe(NULL_TRIALS, trial_sr_std)
    assert psr_from_stats(0.25, 0.0, 3.0, NULL_N_OBS, sr_benchmark=sr_star) < DSR_GATE
    assert psr_from_stats(0.36, 0.0, 3.0, NULL_N_OBS, sr_benchmark=sr_star) >= DSR_GATE
    assert winner["sharpe"] < 0.25 < 0.36 < EDGE_MU / NULL_SIGMA


def test_dsr_is_non_increasing_in_n_trials():
    # Searching more configs makes any given Sharpe less impressive — never more.
    winner, trial_sr_std = _null_search(NULL_SEED)
    grid = [1, 2, 5, 10, 25, 100, 500, 2000, 10000, 100000]
    vals = [_dsr_of(winner, n, trial_sr_std) for n in grid]
    assert all(b <= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] < vals[0]  # not vacuously constant
    assert _dsr_of(winner, 25, trial_sr_std) < _dsr_of(winner, 5, trial_sr_std)


def test_expected_max_sharpe_is_non_decreasing_in_trials_and_in_trial_std():
    trials = [1, 2, 5, 10, 25, 100, 500, 2000, 10000, 100000]
    by_trials = [expected_max_sharpe(n, 0.5) for n in trials]
    assert all(a <= b for a, b in zip(by_trials, by_trials[1:]))
    assert by_trials[0] < by_trials[-1]  # not vacuously constant

    stds = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    by_std = [expected_max_sharpe(100, s) for s in stds]
    assert all(a <= b for a, b in zip(by_std, by_std[1:]))
    assert by_std[0] < by_std[-1]


def test_no_deflation_for_degenerate_trial_counts_or_spreads():
    # <2 trials or a non-positive cross-trial spread means there is nothing to correct for: SR* = 0, so DSR
    # collapses onto the undeflated PSR rather than silently rejecting.
    winner, _ = _null_search(NULL_SEED)
    undeflated = _psr_of(winner, sr_benchmark=0.0)
    assert expected_max_sharpe(-5, 0.5) == 0.0
    assert expected_max_sharpe(NULL_TRIALS, -0.5) == 0.0
    assert _dsr_of(winner, -5, 0.5) == pytest.approx(undeflated, rel=1e-12)
    assert _dsr_of(winner, NULL_TRIALS, -0.5) == pytest.approx(undeflated, rel=1e-12)


def test_dsr_is_undefined_below_two_observations():
    assert dsr_from_stats(0.5, 0.0, 3.0, 1, n_trials=NULL_TRIALS, trial_sr_std=0.2) == 0.0
    assert dsr_from_stats(0.5, 0.0, 3.0, 0, n_trials=NULL_TRIALS, trial_sr_std=0.2) == 0.0
    assert deflated_sharpe_ratio([0.01], n_trials=NULL_TRIALS, trial_sr_std=0.2) == 0.0
    assert min_track_record_length([0.01]) == math.inf


# --- powered-null primitives (Sharpe SE/CI, MDE, power, FDR, per-cell verdict) -------------------------------
from scipy.stats import norm  # noqa: E402

from trainer.sharpe import (  # noqa: E402
    sharpe_standard_error,
    sharpe_confidence_interval,
    minimum_detectable_sharpe,
    sharpe_power,
    benjamini_hochberg,
    powered_null_verdict,
    newey_west_inflation,
    sharpe_standard_error_hac,
    benjamini_yekutieli,
)


def test_newey_west_inflation_known_values():
    assert newey_west_inflation([0.0, 0.0, 0.0], 3) == pytest.approx(1.0)
    # rho_1 = 0.5, q = 1: eta = 1 + 2*(1 - 1/2)*0.5 = 1.5
    assert newey_west_inflation([0.5], 1) == pytest.approx(1.5)
    # q < 1 -> no adjustment; strong negative autocorr floors the factor, never inverts it
    assert newey_west_inflation([0.9], 0) == 1.0
    assert newey_west_inflation([-1.0], 1) == pytest.approx(1e-6)


def test_sharpe_standard_error_hac_matches_iid_on_white_noise_and_grows_under_autocorr():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(4000) + 0.03
    st = sharpe_stats(white)
    se_iid = sharpe_standard_error(st["sharpe"], st["skew"], st["kurtosis"], st["n_obs"])
    # white noise: HAC ~ iid (small-sample autocorr only)
    assert sharpe_standard_error_hac(white) == pytest.approx(se_iid, rel=0.15)
    # a smoothed (positively autocorrelated) series: HAC SE strictly larger than iid
    smooth = np.convolve(white, np.ones(5) / 5, mode="same")
    st2 = sharpe_stats(smooth)
    se_iid2 = sharpe_standard_error(st2["sharpe"], st2["skew"], st2["kurtosis"], st2["n_obs"])
    assert sharpe_standard_error_hac(smooth) > se_iid2


def test_benjamini_yekutieli_is_more_conservative_than_bh():
    p = [0.001, 0.02, 0.5, 0.6]
    bh = benjamini_hochberg(p, q=0.05)
    by = benjamini_yekutieli(p, q=0.05)
    assert sum(by) <= sum(bh)  # BY rejects no more than BH
    # m=4 -> H_m = 1+1/2+1/3+1/4 = 2.0833; BY threshold at k=1 is 0.05/2.0833/4 = 0.006
    assert by[0] is True and by == [True, False, False, False]


def test_sharpe_standard_error_normal_is_lo_variance():
    # normal (skew 0, non-excess kurt 3): SE(SR) = sqrt((1 + 0.5*SR^2)/(n-1))
    assert sharpe_standard_error(0.1, 0.0, 3.0, 101) == pytest.approx(
        math.sqrt((1 + 0.5 * 0.1 ** 2) / 100), rel=1e-12
    )


def test_sharpe_standard_error_undefined_is_inf():
    assert sharpe_standard_error(0.1, 0.0, 3.0, 1) == math.inf
    # a skew/kurt combination that drives the PSR denominator non-positive
    assert sharpe_standard_error(5.0, 3.0, 3.0, 100) == math.inf


def test_sharpe_confidence_interval_symmetric_two_sided():
    lo, hi = sharpe_confidence_interval(0.2, 0.0, 3.0, 401, alpha=0.05)
    se = sharpe_standard_error(0.2, 0.0, 3.0, 401)
    z = float(norm.ppf(0.975))
    assert lo == pytest.approx(0.2 - z * se, rel=1e-9)
    assert hi == pytest.approx(0.2 + z * se, rel=1e-9)


def test_power_at_the_mde_equals_the_target():
    n, alpha, power = 500, 0.05, 0.8
    mde = minimum_detectable_sharpe(n, alpha=alpha, power=power)
    assert sharpe_power(mde, n, alpha=alpha) == pytest.approx(power, abs=2e-3)


def test_mde_shrinks_with_sample_size():
    assert minimum_detectable_sharpe(2000) < minimum_detectable_sharpe(200)


def test_sharpe_power_is_monotone_in_effect():
    assert sharpe_power(0.20, 300) > sharpe_power(0.05, 300)
    assert sharpe_power(0.0, 300) == pytest.approx(0.05, abs=1e-9)  # power at the null == alpha


def test_benjamini_hochberg_step_up_and_order_invariance():
    assert benjamini_hochberg([0.001, 0.04, 0.5, 0.5], q=0.05) == [True, False, False, False]
    # step-up pulls in the earlier (larger) p once a later one clears; order must not matter
    assert benjamini_hochberg([0.03, 0.012], q=0.05) == [True, True]
    assert benjamini_hochberg([0.9, 0.8], q=0.05) == [False, False]
    assert benjamini_hochberg([], q=0.05) == []


def test_powered_null_verdict_disproves_a_tight_zero():
    v = powered_null_verdict(0.0, 0.0, 3.0, 100000, sr_econ=0.05, alpha=0.05)
    assert v["verdict"] == "powered-null"
    assert v["upper_bound"] < 0.05


def test_powered_null_verdict_inconclusive_when_underpowered():
    v = powered_null_verdict(0.0, 0.0, 3.0, 30, sr_econ=0.05, alpha=0.05)
    assert v["verdict"] == "inconclusive"
    assert v["upper_bound"] > 0.05


def test_powered_null_verdict_flags_a_survivor():
    v = powered_null_verdict(0.3, 0.0, 3.0, 500, sr_econ=0.05, alpha=0.05)
    assert v["verdict"] == "survivor"
    assert v["lower_bound"] > 0.0


# --- HAC-corrected PSR (Lo-2002 serial-correlation adjustment the standard PSR omits) ---

def _ar1(n, phi, sigma, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * sigma
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_psr_hac_matches_iid_when_no_autocorrelation():
    from trainer.sharpe import probabilistic_sharpe_ratio_hac
    x = _ar1(4000, 0.0, 0.01, 1) + 0.001   # iid with positive drift
    assert probabilistic_sharpe_ratio_hac(x) == pytest.approx(probabilistic_sharpe_ratio(x), abs=0.03)


def test_psr_hac_is_more_conservative_under_positive_autocorrelation():
    from trainer.sharpe import probabilistic_sharpe_ratio_hac
    x = _ar1(3000, 0.5, 0.01, 2) + 0.0015   # positively autocorrelated, positive drift
    assert probabilistic_sharpe_ratio_hac(x) < probabilistic_sharpe_ratio(x)


def test_psr_hac_degenerate_inputs():
    from trainer.sharpe import probabilistic_sharpe_ratio_hac
    assert probabilistic_sharpe_ratio_hac([0.01]) == 0.0
    assert 0.0 <= probabilistic_sharpe_ratio_hac([0.01, -0.01, 0.01, -0.01]) <= 1.0


def test_psr_hac_bounded_probability():
    from trainer.sharpe import probabilistic_sharpe_ratio_hac
    p = probabilistic_sharpe_ratio_hac(_ar1(2000, 0.3, 0.01, 3) + 0.002)
    assert 0.0 <= p <= 1.0
