"""Faithful Kelly-Malamud-Zhou complexity ladder: random Fourier features + ridge with shrinkage z on the SAMPLE
second moment, solved in whichever of the primal (P<=N) / dual (P>N) forms is efficient so complexity c=P/N can
run past interpolation. This is the corrected primitive after the v1 proof-seed failed verification for using an
inert ridge (lambda*I on Z'Z of scale ~N) and unnormalized features.

Objective (KMZ units): w minimises (1/T)||y - Z w||^2 + z ||w||^2, i.e. (Z'Z/T + z I) w = Z'y/T, so z is a
shrinkage on the sample covariance Z'Z/T (equivalently sklearn Ridge with alpha = z*T). Features are scaled
sqrt(2/P) so E||phi(x)||^2 ~ 1 and z is in meaningful units across the whole P sweep. numpy only at runtime."""
import numpy as np


def random_fourier_features(X, W, b):
    """cos(X W + b) * sqrt(2/P): P random Fourier features approximating a shift-invariant kernel, normalized so
    the mean squared feature-vector norm is ~1 (kept constant as P grows, unlike a bare sqrt(2))."""
    X = np.asarray(X, dtype=float)
    p = W.shape[1]
    return np.cos(X @ W + b) * np.sqrt(2.0 / p)


def kmz_ridge_fit_predict(Ztr, ytr, Zte, z, force=None):
    """Fit ridge (Z'Z/T + z I) w = Z'y/T on (Ztr, ytr) and return (train_pred, test_pred). Uses the dual form
    w = Z'(ZZ'/T + z I_T)^{-1} y/T when P > T (or force='dual'), else the primal; both are algebraically equal.
    `z` is the shrinkage on the sample second moment (sklearn Ridge alpha = z*T)."""
    Ztr = np.asarray(Ztr, dtype=float)
    Zte = np.asarray(Zte, dtype=float)
    ytr = np.asarray(ytr, dtype=float).ravel()
    t, p = Ztr.shape
    use_dual = (force == "dual") or (force is None and p > t)
    if use_dual:
        g = Ztr @ Ztr.T / t
        g[np.diag_indices_from(g)] += z
        alpha = np.linalg.solve(g, ytr / t)
        return Ztr @ (Ztr.T @ alpha), Zte @ (Ztr.T @ alpha)
    a = Ztr.T @ Ztr / t
    a[np.diag_indices_from(a)] += z
    w = np.linalg.solve(a, Ztr.T @ ytr / t)
    return Ztr @ w, Zte @ w
