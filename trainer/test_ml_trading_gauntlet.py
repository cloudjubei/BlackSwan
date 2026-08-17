import numpy as np
import pytest

from trainer.ml_trading_gauntlet import run_gauntlet


def _good(rng, t=750, mean=0.0011, sd=0.008):
    return rng.standard_normal(t) * sd + mean


def _noise(rng, t=750, sd=0.008):
    return rng.standard_normal(t) * sd


def test_noise_family_none_survive():
    rng = np.random.default_rng(0)
    fam = [_noise(rng) for _ in range(40)]
    per, family = run_gauntlet(fam, n_trials=40)
    assert family["n_survived"] == 0
    assert family["survival_rate"] == 0.0


def test_strong_strategy_survives_all_gates():
    rng = np.random.default_rng(1)
    per, family = run_gauntlet([_good(rng)], n_trials=1)
    assert per[0]["survives"] is True
    assert family["n_survived"] == 1


def test_random_null_gate_bites():
    rng = np.random.default_rng(2)
    strong = _good(rng)
    null_above = np.full(200, 50.0)
    per, family = run_gauntlet([strong], n_trials=1, null_sharpes=null_above)
    assert per[0]["econ_pass"] is True
    assert per[0]["dsr_pass"] is True
    assert per[0]["null_pass"] is False
    assert per[0]["survives"] is False


def test_random_null_gate_passes_when_strategy_beats_null():
    rng = np.random.default_rng(3)
    strong = _good(rng)
    null_below = np.full(200, -50.0)
    per, _ = run_gauntlet([strong], n_trials=1, null_sharpes=null_below)
    assert per[0]["null_pass"] is True
    assert per[0]["survives"] is True


def test_postcutoff_gate_bites():
    rng = np.random.default_rng(4)
    series = np.concatenate([_good(rng, t=450), _noise(rng, t=450)])
    per, _ = run_gauntlet([series], n_trials=1, cutoff=450)
    assert per[0]["postcutoff_pass"] is False
    assert per[0]["survives"] is False


def test_postcutoff_gate_passes_when_edge_persists():
    rng = np.random.default_rng(5)
    series = _good(rng, t=900)
    per, _ = run_gauntlet([series], n_trials=1, cutoff=450)
    assert per[0]["postcutoff_pass"] is True
    assert per[0]["survives"] is True


def test_multiplicity_bites_on_noise_family():
    rng = np.random.default_rng(6)
    fam = [_noise(rng) for _ in range(200)]
    per, family = run_gauntlet(fam, n_trials=200)
    assert family["n_nominal_sig"] >= 1
    assert family["n_survived"] == 0


def test_gates_default_pass_matches_certify_core():
    rng = np.random.default_rng(7)
    fam = [_good(rng), _noise(rng)]
    per, _ = run_gauntlet(fam, n_trials=2)
    for p in per:
        assert p["survives"] == bool(p["econ_pass"] and p["dsr_pass"] and p["by"])


def test_survival_is_strict_and_of_all_gates():
    rng = np.random.default_rng(8)
    strong = _good(rng)
    per, _ = run_gauntlet([strong], n_trials=1, null_sharpes=np.full(50, 50.0), cutoff=375)
    assert per[0]["survives"] is False


def test_empty_family():
    per, family = run_gauntlet([], n_trials=0)
    assert per == []
    assert family["n_survived"] == 0
    assert family["survival_rate"] == 0.0
