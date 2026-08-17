"""Unit tests for the faithful Kelly-Malamud-Zhou complexity-ladder primitive (random Fourier features + ridge
with shrinkage z on the SAMPLE second moment, dual-solved so complexity c=P/N can exceed 1). The v1 proof-seed
failed verification precisely because its ridge was inert and its features unnormalized; these tests pin the
correct behaviour: dual==primal, proper sqrt(2/P) scaling, ridgeless interpolation past P=N, and equivalence to
sklearn Ridge with alpha=z*T. numpy + scipy + sklearn only; no torch."""
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from trainer.complexity_ladder import kmz_ridge_fit_predict, random_fourier_features


def _bank(rng, d, p, gamma=1.0):
    return rng.standard_normal((d, p)) * np.sqrt(gamma), rng.uniform(0, 2 * np.pi, p)


def test_rff_shape_and_unit_scale():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 5))
    W, b = _bank(rng, 5, 400)
    Z = random_fourier_features(X, W, b)
    assert Z.shape == (2000, 400)
    mean_sq_norm = float(np.mean(np.sum(Z ** 2, axis=1)))
    assert mean_sq_norm == pytest.approx(1.0, abs=0.15)


def test_primal_dual_agree_underparam():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 4))
    W, b = _bank(rng, 4, 80)
    Ztr = random_fourier_features(X, W, b)
    ytr = rng.standard_normal(300)
    Zte = random_fourier_features(rng.standard_normal((100, 4)), W, b)
    _, pd = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=0.05, force="dual")
    _, pp = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=0.05, force="primal")
    assert np.allclose(pd, pp, atol=1e-8)


def test_primal_dual_agree_overparam():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((150, 4))
    W, b = _bank(rng, 4, 900)
    Ztr = random_fourier_features(X, W, b)
    ytr = rng.standard_normal(150)
    Zte = random_fourier_features(rng.standard_normal((100, 4)), W, b)
    _, pd = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=0.02, force="dual")
    _, pp = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=0.02, force="primal")
    assert np.allclose(pd, pp, atol=1e-7)


def test_ridgeless_interpolates_when_overparam():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 4))
    W, b = _bank(rng, 4, 1200)
    Ztr = random_fourier_features(X, W, b)
    ytr = rng.standard_normal(120)
    pred_tr, _ = kmz_ridge_fit_predict(Ztr, ytr, Ztr, z=1e-10)
    assert np.max(np.abs(pred_tr - ytr)) < 1e-4


def test_recovers_sklearn_ridge_alpha_zT():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((400, 5))
    W, b = _bank(rng, 5, 120)
    Ztr = random_fourier_features(X, W, b)
    ytr = rng.standard_normal(400)
    Zte = random_fourier_features(rng.standard_normal((60, 5)), W, b)
    z = 0.03
    _, pred = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=z)
    sk = Ridge(alpha=z * Ztr.shape[0], fit_intercept=False).fit(Ztr, ytr)
    assert np.allclose(pred, sk.predict(Zte), atol=1e-6)


def test_auto_matches_forced_both_regimes():
    rng = np.random.default_rng(5)
    for n, p in ((200, 60), (80, 500)):
        X = rng.standard_normal((n, 3))
        W, b = _bank(rng, 3, p)
        Ztr = random_fourier_features(X, W, b)
        ytr = rng.standard_normal(n)
        Zte = random_fourier_features(rng.standard_normal((40, 3)), W, b)
        _, pa = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=0.04)
        _, pf = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=0.04, force="primal" if p < n else "dual")
        assert np.allclose(pa, pf, atol=1e-7)


def test_larger_z_shrinks_weights():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((200, 4))
    W, b = _bank(rng, 4, 150)
    Ztr = random_fourier_features(X, W, b)
    ytr = X[:, 0] + rng.standard_normal(200) * 0.1
    Zte = random_fourier_features(rng.standard_normal((50, 4)), W, b)
    _, low = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=1e-4)
    _, high = kmz_ridge_fit_predict(Ztr, ytr, Zte, z=10.0)
    assert np.std(high) < np.std(low)
