"""Unit tests for the machine-discovered-alpha certification gauntlet (option 3, 'Deflated Sharpe for the LLM
era'). The gauntlet composes the already-tested rigor primitives in sharpe.py (HAC one-sided bound + multiplicity-
deflated Sharpe + BH/BY FDR + powered-null verdict) into a single per-strategy PASS/FAIL and a family-level
survival count. These tests pin the load-bearing behaviour: a pure-null family certifies ~nobody even though
several look nominally significant; a genuinely-strong strategy survives while nulls around it do not;
certification is a strict subset of nominal significance; and MORE (claimed) trials deflate certification
monotonically. numpy + scipy only; no torch."""
import numpy as np
import pytest

from trainer.certification import certify_family


def _null_family(m, n, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n) * 0.01 for _ in range(m)]


def _alpha_series(n, sharpe_ann, seed, ann=np.sqrt(252.0)):
    rng = np.random.default_rng(seed)
    mu = (sharpe_ann / ann) * 0.01
    return rng.standard_normal(n) * 0.01 + mu


def test_pure_null_family_certifies_nobody_though_some_look_significant():
    fam = _null_family(200, 1500, seed=1)
    per, family = certify_family(fam)
    assert family["n_certified"] == 0
    assert family["n_nominal_sig"] >= 1


def test_true_alpha_survives_and_nulls_do_not():
    strong = _alpha_series(3000, sharpe_ann=2.6, seed=7)
    fam = [strong] + _null_family(40, 3000, seed=8)
    per, family = certify_family(fam)
    assert per[0]["certified"] is True
    assert sum(p["certified"] for p in per[1:]) == 0
    assert family["n_certified"] == 1


def test_certified_is_strict_subset_of_nominal_significance():
    fam = [_alpha_series(2500, 1.8, seed=3)] + _null_family(30, 2500, seed=4)
    per, _ = certify_family(fam)
    for p in per:
        if p["certified"]:
            assert p["p_one"] < 0.05


def test_more_claimed_trials_deflate_certification_monotonically():
    fam = [_alpha_series(2500, 1.6, seed=11), _alpha_series(2500, 1.5, seed=12)] + _null_family(10, 2500, seed=13)
    _, few = certify_family(fam, n_trials=5)
    _, many = certify_family(fam, n_trials=5000)
    assert many["n_certified"] <= few["n_certified"]
    assert many["n_dsr_pass"] <= few["n_dsr_pass"]


def test_dsr_falls_as_trials_rise_for_fixed_strategy():
    fam = [_alpha_series(2500, 1.6, seed=21)] + _null_family(6, 2500, seed=22)
    per_few, _ = certify_family(fam, n_trials=3)
    per_many, _ = certify_family(fam, n_trials=10000)
    assert per_many[0]["dsr"] < per_few[0]["dsr"]


def test_family_counts_are_ordered_nominal_ge_certified():
    fam = [_alpha_series(2500, 2.0, seed=31)] + _null_family(50, 2500, seed=32)
    _, family = certify_family(fam)
    assert family["n_nominal_sig"] >= family["n_certified"]
    assert family["n_strategies"] == 51


def test_degenerate_inputs_do_not_crash():
    per, family = certify_family([np.zeros(500), np.ones(500) * 0.01])
    assert family["n_certified"] == 0
    assert all(np.isfinite(p["sharpe_ann"]) for p in per)


def test_empty_family_is_empty():
    per, family = certify_family([])
    assert per == []
    assert family["n_strategies"] == 0 and family["n_certified"] == 0


def test_effective_trials_mode_deflates_less_than_raw_on_redundant_family():
    rng = np.random.default_rng(41)
    base = rng.standard_normal(2500) * 0.01
    redundant = [base + rng.standard_normal(2500) * 0.001 for _ in range(60)]  # ~1 effective trial
    _, raw = certify_family(redundant, n_trials=None)
    _, eff = certify_family(redundant, n_trials="effective")
    assert eff["effective_trials"] < 5.0 < raw["n_trials"]
    assert eff["n_trials"] <= 3
    per_eff, _ = certify_family(redundant, n_trials="effective")
    for p in per_eff:
        assert p["dsr"] >= 0.0
