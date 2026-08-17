"""Effective degrees of freedom of a gradient-fit strategy on pure noise, in a Sharpe-deflation metric -- HONEST
FOOTNOTE, not a landmark (adversarial verification recordable=false; see below). It restates KNOWN effective-model-
complexity / generalized-degrees-of-freedom facts through a trading-Sharpe inversion; it is NOT an open result.

Known context (do NOT claim novelty here): early-stopped gradient descent ~ ridge with lambda~1/t
(Ali-Kolter-Tibshirani 2019); search / generalized DoF can exceed the parameter count (R. Tibshirani 2015; Efron;
Ye); Effective Model Complexity grows with BOTH width and training epochs (Nakkiran 2019); neural EDF growth-during-
training and saturation near the parameter count is measured directly in arXiv 2602.13442. The exact in-sample
overfit result arXiv 2501.03938 (Jan 2025) is LINEAR-ONLY and does NOT address the neural case (an earlier draft
here fabricated that it 'names the neural case as open' -- it does not; that clause was removed).

What this measures, honestly: on PURE NOISE (features independent of target, true edge = 0) the in-sample timing
Sharpe S of a fitted model is pure overfit; p_eff = T*S^2/(252+S^2) reads it back as effective degrees of freedom.
(1) CALIBRATION: for RFF-ridge, measured p_eff tracks the nominal capacity P (mildly downward-biased, ~0.55-1.10;
near 1 only for P>=10) -- validating the HARNESS against textbook R^2~p/n -> Sharpe -> DoF, not that p_eff is 'the
right DoF'. (2) The gradient MLP's p_eff is a small setup-specific multiple (~1.1-1.5x) of its param count that
grows with width AND training epochs and saturates -- CONSISTENT WITH the known EDF results above. (3) The one
useful residual is DSR-pipeline plumbing: a NULL-CALIBRATED, held-out in-sample overfit haircut for gradient-fit
strategies that needs no config enumeration -- demonstrated with proper train/test separation (calibrate the
haircut on one set of null seeds, evaluate on FRESH ones): naive DSR (n_trials=1) certifies fresh pure-noise fits,
the calibrated haircut does not. CAVEAT: the p_eff-form haircut sqrt(252*p_eff/T) itself UNDER-deflates the
high-Sharpe neural regime (it drops the 1/(1-R^2) factor), so the trustworthy haircut is the directly-measured
mean overfit, not the p_eff transform. Not a law, not an open regime, not alpha, not a landmark."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.effective_trials import effective_dof_from_sharpe  # noqa: E402
from trainer.sharpe import psr_from_stats, sharpe_stats  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ANN = math.sqrt(252.0)
D = 10
SEEDS = 40
torch.set_num_threads(4)


def gen_null(rng, t, d=D):
    return rng.standard_normal((t, d)).astype(np.float32), rng.standard_normal(t).astype(np.float32)


def insample_timing_sharpe(pred, y):
    p = pred - pred.mean()
    sd = p.std()
    if sd == 0:
        return 0.0
    pos = np.clip(p / sd, -3.0, 3.0)
    return sharpe_stats(pos * y)["sharpe"] * ANN


def rff_ridge_insample(X, y, p, gamma=0.5, lam=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((X.shape[1], p)) * math.sqrt(gamma)
    b = rng.uniform(0, 2 * math.pi, p)
    Z = np.cos(X @ W + b) * math.sqrt(2.0 / p)
    yc = y - y.mean()
    A = Z.T @ Z + lam * np.eye(p)
    w = np.linalg.solve(A, Z.T @ yc)
    return insample_timing_sharpe(Z @ w, y)


def mlp_insample(X, y, width, epochs, depth=1, seed=0):
    torch.manual_seed(seed)
    layers = [nn.Linear(X.shape[1], width), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, 1)]
    net = nn.Sequential(*layers)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(net(xt), yt).backward()
        opt.step()
    with torch.no_grad():
        pred = net(xt).numpy().ravel()
    return insample_timing_sharpe(pred, y)


def measure(fn, t, seed0=1000, seeds=SEEDS, **kw):
    vals = [fn(*gen_null(np.random.default_rng(seed0 + s), t), seed=s, **kw) for s in range(seeds)]
    s_is = float(np.mean(vals))
    return s_is, effective_dof_from_sharpe(s_is, t), vals


def params_mlp(w, depth=1):
    return w * (D + 1) + (depth - 1) * (w * w + w) + (w + 1)


def main():
    print("Effective DoF of a gradient fit on PURE NOISE, in a Sharpe metric (HONEST FOOTNOTE, not a landmark). "
          f"D={D}, seeds={SEEDS}\n")

    print("=== (1) CALIBRATION -- RFF-ridge (linear): p_eff tracks nominal capacity P (mildly downward-biased) ===")
    print(f"{'T':>6}{'P':>5}{'S_is':>8}{'p_eff':>8}{'p_eff/P':>9}")
    ridge_rows = []
    for t in (1000, 2000, 4000):
        for p in (2, 5, 10, 20, 40):
            s_is, pe, _ = measure(rff_ridge_insample, t, p=p)
            ridge_rows.append({"T": t, "P": p, "S_is": s_is, "p_eff": pe})
            print(f"{t:>6}{p:>5}{s_is:>8.2f}{pe:>8.1f}{pe / p:>9.2f}")

    print("\n=== (2) GRADIENT MLP: p_eff ~ a small multiple of params, grows with width AND epochs (KNOWN EDF) ===")
    print(f"{'width':>6}{'params':>7}{'epochs':>8}{'S_is':>8}{'p_eff':>8}{'p_eff/params':>13}")
    mlp_rows = []
    T = 2000
    for width in (4, 8, 16):
        pm = params_mlp(width)
        for epochs in (50, 200, 800, 3200):
            s_is, pe, _ = measure(mlp_insample, T, width=width, epochs=epochs)
            mlp_rows.append({"width": width, "params": pm, "epochs": epochs, "S_is": s_is, "p_eff": pe})
            print(f"{width:>6}{pm:>7}{epochs:>8}{s_is:>8.2f}{pe:>8.1f}{pe / pm:>13.2f}")

    print("\n=== (3) HELD-OUT anti-conservatism: naive DSR certifies FRESH pure-noise fits; calibrated haircut does not ===")
    cw, ce = 8, 200
    s_cal, _, _ = measure(mlp_insample, T, seed0=1000, seeds=SEEDS, width=cw, epochs=ce)  # haircut from calib seeds
    n_fresh = 120
    fresh = [mlp_insample(*gen_null(np.random.default_rng(90000 + s), T), width=cw, epochs=ce, seed=90000 + s)
             for s in range(n_fresh)]
    naive_cert = sum(1 for s in fresh if psr_from_stats(s / ANN, 0.0, 3.0, T, sr_benchmark=0.0) > 0.95)
    hair_cert = sum(1 for s in fresh if psr_from_stats(s / ANN, 0.0, 3.0, T, sr_benchmark=s_cal / ANN) > 0.95)
    print(f"  calibrated overfit haircut (MLP w={cw}, epochs={ce}) = {s_cal:.2f} ann Sharpe, from {SEEDS} calib seeds")
    print(f"  on {n_fresh} FRESH pure-noise fits (unseen seeds): mean in-sample Sharpe {np.mean(fresh):.2f}")
    print(f"  naive DSR (n_trials=1) certifies:   {naive_cert}/{n_fresh} ({100*naive_cert/n_fresh:.0f}%)  <- anti-conservative on pure noise")
    print(f"  null-calibrated haircut certifies:  {hair_cert}/{n_fresh} ({100*hair_cert/n_fresh:.0f}%)  <- correctly rejects overfit")

    summary = {"D": D, "seeds": SEEDS, "ridge": ridge_rows, "mlp": mlp_rows,
               "heldout": {"calib_haircut": s_cal, "n_fresh": n_fresh, "fresh_mean": float(np.mean(fresh)),
                           "naive_certified": naive_cert, "haircut_certified": hair_cert}}
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(summary, open("experiments/results/overfitting_deflation_law.json", "w"), indent=1, default=str)
    print("\nwrote experiments/results/overfitting_deflation_law.json")


if __name__ == "__main__":
    main()
