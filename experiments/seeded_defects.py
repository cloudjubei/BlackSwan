"""SWING-i: SEEDED-DEFECT validation of the apparatus — the methods contribution's core evidence. Turn the
"the harness caught 2 of its own bugs" anecdote into a measured benchmark: inject a catalogue of KNOWN
statistical artifacts into strategies and measure whether the apparatus's detectors flag each (recall) without
flagging clean strategies (false-alarm). Each defect is a concrete construction; each detector a concrete test;
both run over many random instantiations on real equity returns for a catch-rate / false-alarm confusion matrix.

Defects (each makes a TRUE-NULL strategy look like an edge, the way a naive analyst would report it):
  leakage        — signal peeks at next-bar return (lookahead)             detector: lag-sensitivity placebo
  is_as_oos      — fit best on TRAIN, report the TRAIN Sharpe as if OOS    detector: train->test Sharpe gap
  trial_undercount — best of N random signals, reported undeflated         detector: Deflated Sharpe (expected-max)
  cost_omission  — high-turnover book reported GROSS                       detector: gross->net Sharpe gap
  survivorship   — drop the instruments that did badly OOS                 detector: full vs restricted universe
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universe_battery import load_universe  # noqa: E402
from trainer.sharpe import sharpe_stats, expected_max_sharpe, deflated_sharpe_ratio, sharpe_standard_error_hac  # noqa: E402

ANN = math.sqrt(252.0)
FEE = 0.0005
N_RUNS = 15
N_IS = 80      # best-of-N for the is_as_oos defect (needs enough overfit for the train->test gap to show)
N_TRIAL = 60   # best-of-N for the trial_undercount defect


def sr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 30 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std())


def book(PX, RET, score, k=3, reb=21):
    """k-long/k-short book, rebalanced every `reb` bars. Vectorized: full weight rows are written ONLY at
    rebalance bars (NaN elsewhere) then forward-filled to hold between rebalances."""
    idx = PX.index; cols = list(PX.columns)
    W = pd.DataFrame(np.nan, index=idx, columns=cols)
    scv = score.to_numpy()
    for i in range(0, len(idx), reb):
        row_scores = scv[i]
        ok = np.where(np.isfinite(row_scores))[0]
        if len(ok) < 2 * k:
            W.iloc[i] = 0.0
            continue
        order = ok[np.argsort(row_scores[ok])]
        row = np.zeros(len(cols))
        row[order[:k]] = -1.0 / k
        row[order[-k:]] = 1.0 / k
        W.iloc[i] = row
    W = W.ffill().fillna(0.0)
    gross = (W.shift(1).fillna(0.0) * RET).sum(axis=1)
    net = gross - FEE * W.diff().abs().sum(axis=1)
    return W, gross, net


def random_score(PX, rng):
    return pd.DataFrame(rng.standard_normal(PX.shape), index=PX.index, columns=PX.columns)


# ---- detectors (each returns True = FLAG the defect) -------------------------------------------------------
def detect_leakage(clean_series, lagged_series):
    """A leaked signal's edge collapses when you add a one-bar lag (it was using contemporaneous/future info)."""
    return sr(clean_series) - sr(lagged_series) > 0.05  # per-obs Sharpe drop when properly lagged


def detect_is_as_oos(train_series, test_series):
    return (sr(train_series) - sr(test_series)) > 0.015


def detect_trial_undercount(best_oos, n_trials, trial_std, n_obs):
    """Flag ONLY when a NOMINALLY-significant result fails the Deflated Sharpe — i.e. the significance is
    multiple-testing luck. A non-significant clean null (raw t < 1.645) is not a trial-undercount defect."""
    raw = sr(best_oos)
    raw_significant = raw * math.sqrt(max(n_obs, 1)) > 1.645
    dsr = deflated_sharpe_ratio(best_oos, n_trials, trial_std)
    return raw_significant and dsr < 0.95


def detect_cost_omission(gross_series, net_series):
    return (sr(gross_series) - sr(net_series)) > 0.03


def detect_survivorship(full_sharpe, restricted_sharpe):
    return (restricted_sharpe - full_sharpe) > 0.02


def main():
    PX, _ = load_universe("equity")
    RET = PX.pct_change(fill_method=None)
    n = len(PX); split = int(n * 0.55); tr, te = PX.index[:split], PX.index[split:]
    print(f"substrate: equity universe {PX.shape[1]} names x {len(PX)} days ({len(PX)/252:.0f}y)")

    cm = {d: {"caught": 0, "total": 0} for d in ["leakage", "is_as_oos", "trial_undercount", "cost_omission", "survivorship"]}
    false_alarm = {d: 0 for d in cm}
    clean_total = 0

    for run in range(N_RUNS):
        rng = np.random.default_rng(1000 + run)

        # CLEAN NULL: a random cross-sectional signal (true edge ~0), properly OOS + net + single trial
        sc = random_score(PX, rng)
        _, g, nt = book(PX, RET, sc)
        clean_train, clean_test = nt.loc[tr].dropna(), nt.loc[te].dropna()
        # false-alarm: do detectors fire on the clean, honestly-reported null?
        clean_total += 1
        if detect_is_as_oos(clean_train, clean_test):
            false_alarm["is_as_oos"] += 1
        if detect_cost_omission(g.loc[te].dropna(), nt.loc[te].dropna()):
            false_alarm["cost_omission"] += 1
        # clean single-trial DSR (n_trials=1 -> no deflation -> should NOT flag as multiple-testing)
        if detect_trial_undercount(clean_test.to_numpy(), 1, 0.0, len(clean_test)):
            false_alarm["trial_undercount"] += 1
        # clean leakage placebo: lagging a legitimate signal by one more bar barely changes it
        lagged = (book(PX, RET, sc.shift(1))[2]).loc[te].dropna()
        if detect_leakage(clean_test, lagged):
            false_alarm["leakage"] += 1
        # clean survivorship: full long-only vs a RANDOM half (not outcome-selected) -> no inflation
        rand_sub = list(rng.choice(PX.columns, size=PX.shape[1] // 2, replace=False))
        full_lo = RET.loc[te].mean(axis=1).dropna()
        sub_lo = RET[rand_sub].loc[te].mean(axis=1).dropna()
        if detect_survivorship(sr(full_lo), sr(sub_lo)):
            false_alarm["survivorship"] += 1

        # DEFECT: LEAKAGE — signal peeks at next-bar return
        leak_score = RET.shift(-1)  # tomorrow's return as today's signal
        _, _, leak_net = book(PX, RET, leak_score)
        lk_test = leak_net.loc[te].dropna()
        lk_lagged = book(PX, RET, leak_score.shift(1))[2].loc[te].dropna()  # remove the peek
        cm["leakage"]["total"] += 1
        if detect_leakage(lk_test, lk_lagged):
            cm["leakage"]["caught"] += 1

        # DEFECT: IS-as-OOS — fit best random signal on train, report the train Sharpe
        best_tr, best_series = -9, None
        for _ in range(N_IS):
            s2 = random_score(PX, rng)
            _, _, nt2 = book(PX, RET, s2)
            v = sr(nt2.loc[tr].dropna())
            if v > best_tr:
                best_tr, best_series = v, nt2
        cm["is_as_oos"]["total"] += 1
        if detect_is_as_oos(best_series.loc[tr].dropna(), best_series.loc[te].dropna()):
            cm["is_as_oos"]["caught"] += 1

        # DEFECT: TRIAL UNDERCOUNT — best of N=50 random signals on OOS, reported undeflated
        trials = []
        for _ in range(N_TRIAL):
            s3 = random_score(PX, rng)
            _, _, nt3 = book(PX, RET, s3)
            trials.append(nt3.loc[te].dropna())
        sharpes = [sr(t) for t in trials]
        bi = int(np.argmax(sharpes))
        cm["trial_undercount"]["total"] += 1
        if detect_trial_undercount(trials[bi].to_numpy(), N_TRIAL, float(np.std(sharpes, ddof=1)), len(trials[bi])):
            cm["trial_undercount"]["caught"] += 1

        # DEFECT: COST OMISSION — a high-turnover (5-day) book reported GROSS
        sc_ht = random_score(PX, rng)
        _, g_ht, n_ht = book(PX, RET, sc_ht, reb=3)
        cm["cost_omission"]["total"] += 1
        if detect_cost_omission(g_ht.loc[te].dropna(), n_ht.loc[te].dropna()):
            cm["cost_omission"]["caught"] += 1

        # DEFECT: SURVIVORSHIP — report only the names that SURVIVED (top-half by OOS total return) vs full
        oos_by_name = RET.loc[te].sum()
        winners = list(oos_by_name.sort_values(ascending=False).index[: PX.shape[1] // 2])
        full_lo2 = RET.loc[te].mean(axis=1).dropna()
        win_lo = RET[winners].loc[te].mean(axis=1).dropna()
        cm["survivorship"]["total"] += 1
        if detect_survivorship(sr(full_lo2), sr(win_lo)):
            cm["survivorship"]["caught"] += 1

    print("\n" + "=" * 74)
    print("SEEDED-DEFECT CONFUSION MATRIX — apparatus detector performance")
    print("=" * 74)
    print(f"{'defect':18s}{'recall (caught)':>18s}{'false-alarm':>14s}")
    for d in cm:
        rec = cm[d]["caught"] / max(cm[d]["total"], 1)
        fa = false_alarm[d] / max(clean_total, 1)
        print(f"{d:18s}{cm[d]['caught']:>6d}/{cm[d]['total']:<3d} ={rec:>5.0%}     {false_alarm[d]:>3d}/{clean_total:<3d} ={fa:>5.0%}")
    macro_recall = np.mean([cm[d]["caught"] / max(cm[d]["total"], 1) for d in cm])
    macro_fa = np.mean([false_alarm[d] / max(clean_total, 1) for d in cm])
    print(f"\n  macro recall {macro_recall:.0%}  macro false-alarm {macro_fa:.0%}  over {N_RUNS} runs")
    import json
    json.dump({"n_runs": N_RUNS, "confusion": {d: {"recall": cm[d]["caught"]/max(cm[d]["total"],1),
              "false_alarm": false_alarm[d]/max(clean_total,1)} for d in cm},
              "macro_recall": macro_recall, "macro_false_alarm": macro_fa},
              open(os.path.join(os.path.dirname(__file__), "seeded_defects_results.json"), "w"), indent=2, default=float)
    print("  wrote seeded_defects_results.json")


if __name__ == "__main__":
    main()
