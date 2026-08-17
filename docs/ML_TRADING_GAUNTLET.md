# The ML-Trading Honesty Gauntlet — "SPIVA / Replicating-Anomalies for ML-trading"

> A unified, powered, net-of-cost apparatus that adjudicates ML-trading claims, plus the first five claims run
> through it. Built to answer the broadened BlackSwan question: *is there any ML training that genuinely works in a
> trading setting, and is the return-prediction literature refutable?* Nothing here is committed.

## The apparatus (built, TDD)

The gauntlet composes the existing rigor primitives (Deflated/HAC Sharpe, effective-trials, BH/BY FDR, mutual
information) with three new pieces, all in `trainer/`:

- **`trading_costs.py`** — the canonical net-of-cost transform (turnover = Σ|Δw|, gross = Σ(w·r), net = gross −
  fee·turnover), extracted from the pattern each strategy module duplicated inline. *12 tests.*
- **`random_formula_null.py`** — the centrepiece. Matched-complexity random reverse-Polish formulas over a feature
  panel → dollar-neutral, one-period-**lagged** book (no look-ahead) → net-of-cost annualised Sharpe → an empirical
  null distribution + continuity-corrected p-value. A mined alpha only counts if it beats random formulas of its own
  structural complexity. *12 tests.*
- **`ml_trading_gauntlet.py`** — `run_gauntlet` composes `certify_family` (economic HAC lower bound + Deflated Sharpe
  vs the true trial count + Benjamini-Yekutieli across the family) with a **random-formula-null gate** and a
  **post-cutoff out-of-sample gate**. A strategy `survives` only as the AND of all gates. *10 tests.*

Mirrored to modeltrainer (`src/tradingCosts.ts`, `src/randomFormulaNull.ts`) — deterministic scorers golden-pinned to
Python (`formulaSharpe = −2.716916557618524`, etc.); random generation ported with an injected RNG. *13 TS tests.*
**Total 113 Python + 13 TS tests green.**

## The organizing principle (already published — do not claim as novel)

*ML training pays in trading iff the target's information content exceeds a function of the search multiplicity.* This
is **Financial Epiplexity** (Noguer i Alonso, arXiv 2607.02695 — bounds Sharpe/IC/breadth by "structural bits per
period") + **Catt** (arXiv 2603.27074 — MI = max achievable log-loss reduction), both verified to exist. We do not
publish the boundary as new theory; we operationalise it empirically, which the theory papers explicitly decline to do
for neural/RL classes.

## Positive control (the boundary, on real data)

`experiments/gauntlet_positive_control.py`, 40 Binance perps × 2,164 days:
- **Information side:** MI(past vol → next vol) = 0.89 nats vs MI(past return → next return) = 0.017 nats — a **52×
  ratio**. Risk is predictable; direction is not.
- **Economic side:** cross-sectional 1-day reversal is **+0.46 gross → −0.31 net** and does not beat the random-formula
  null → SURVIVES = False.

## The five claims

| # | Claim | Data | Verdict |
|---|---|---|---|
| 1 | NCO (López de Prado) beats 1/N & plain shrinkage | crypto + cross-asset | **Refutation FAILS → honest finding** |
| 6 | Crypto XS momentum is tradeable alpha | perps + funding | **Refutation stands** (dies net-of-cost OOS) |
| 3 | Formulaic-alpha-mining zoo mines real alpha | crypto | **Refutation stands** (gross-real, net cost artifact) |
| 2 | Meta-labeling adds efficacy | crypto | **Inconclusive** (decayed primary) |
| 5 | Self-graded agentic Sharpe 1.87 / 3.48 | published numbers | **Insignificant / unverifiable-as-reported** |

**#1 NCO** (`nco_gauntlet.py`, `nco_robustness.py`). The pre-registered refutation *failed* — and the discipline
recorded the honest opposite. **Long-only** min-variance robustly beats 1/N net-of-cost in the high-dispersion crypto
cross-section (HAC-lower +0.44…+0.62, DSR ≈ 1, survives post-cutoff across windows — *not* a short-funding artifact),
but ties 1/N on the lower-dispersion cross-asset set. So the finding is **dispersion-conditional** (a refinement of
DeMiguel's "1/N is unbeatable"), and **NCO's clustering adds nothing over plain Ledoit-Wolf GMV**.

**#6 crypto momentum** (`crypto_overclaim_gauntlet.py`). Charged taker + slippage + **funding** (the perp-specific cost
papers omit), cross-sectional momentum SURVIVES = False everywhere. The liquid half looks positive in-sample (+0.69
net, beats the null p = 0.013) but **fails post-cutoff (−0.02)** — a decay kill, not a "nothing there."

**#3 alpha-mining zoo** (`alpha_mining_gauntlet.py`). Best-of-K random formulas capture *weak persistent gross
structure* (top-20 gross OOS +0.48), but +1.67 Sharpe of turnover cost destroys it; net-selected formulas collapse and
**0 / 400 survive** their own search multiplicity (400 formulas = 7 effective trials). The zoo's gross IC/Sharpe
headline is real-but-untradeable.

**#2 meta-labeling** (`metalabeling_gauntlet.py`). Inconclusive: the TSMOM primary decays out-of-sample (gross +0.55 IS
→ −0.24 OOS) and the meta-model finds near-zero discriminating signal (sizing std 0.02 → constant), so the
orthogonal-vs-same-feature boundary cannot be demonstrated without a primary that retains OOS edge. Deferred.

**#5 agentic headlines** (`agentic_headline_deflation.py`; sources verified). Agora (2606.29194) +1.87 on a single
91-day, single-seed holdout has t = 1.12 (p = 0.13) — statistically indistinguishable from zero, and its own baseline
is −0.755 cross-seed. AgonAlpha (2608.11250) Sharpe 3.48 is reachable by an undeflated search on a short evaluation
and the paper discloses neither the trial count, evaluation length, nor a deflated Sharpe — unverifiable as reported.

## Postscript — pushing #1 to publishable (and why it doesn't clear the bar)

`experiments/dispersion_law.py` turned #1's two-cross-section observation into a continuum: 90 random long-only
15-perp baskets, **Spearman(cross-sectional vol dispersion, min-var-minus-1/N net Sharpe) = +0.70 (p < 1e-4)**,
tercile advantage rising +0.21 → +0.42 → +0.98 with HAC-significant share 17% → 30% → 70%, NCO − LW-GMV ≈ −0.02.
The relationship is real in-sample. But a prior-art gate (verified against primary sources) shows it is a **modest
refinement, not a discovery**, and must not be written up as a law:

- The *question and method* — a cross-sectional volatility statistic conditioning mean-variance-vs-1/N — are owned by
  **Horses for Courses (Platanakis-Sutcliffe-Ye, EJOR 2020)**, which keys on idiosyncratic-vol *level*.
- The *sign* is near-mechanical (Choueifaty: equal vols ⇒ min-var = 1/N, so "min-var beats 1/N as vols disperse" is
  definitional gross of estimation error).
- The *mechanism* (low-vol tilt, not covariance cleverness) is owned by Scherer 2011 / Clarke-de Silva-Thorley; the
  *NCO-clustering-null* by Trucios 2026 and "Beyond De Prado and Cotton" 2026; *covariance-beats-1/N-net-of-cost* by
  Kirby-Ostdiek 2012 / Curran 2020.
- Decisively, the crypto premise is **era-dependent and contested**: **Brauneis-Mestel 2018** found the *opposite* on
  2015–2017 crypto (1/N beat >75% of mean-variance portfolios on Sharpe/CEQ net-of-cost). My 2022–2026 window merely
  favored the low-vol tilt — an era effect, not a robust law.

**Verdict: #1 is not a publishable novel finding.** Recording it as one would have been an overclaim; the gate caught it.

## The real deliverable

The gauntlet is **honest in both directions** — it kills false alpha (reversal, momentum decay, mining cost artifact)
*and* false refutations (NCO's pre-registered kill fired, so we recorded the opposite honest finding rather than the
refutation we set out to make). Verify-before-record caught mis-framed verdicts twice (the alpha-mining correlation was
cost-persistence, not alpha; the meta-labeling result was degenerate, not a boundary). The honest ceiling holds a fifth
time: this is referee / law / apparatus work — decisive on data in hand, but not a free-data landmark and not a
money-printer.
