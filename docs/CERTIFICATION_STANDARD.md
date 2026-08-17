# Certifying machine-discovered alpha: the effective-trials standard

A practical certification protocol for strategies proposed by an automated search or an LLM agent, and the one
non-trivial correction it needs. This is rigor **infrastructure**, not a new theorem; the primitives are standard
(Lo 2002 HAC Sharpe SE; Bailey–López de Prado Deflated Sharpe; Benjamini–Yekutieli FDR). Code: `trainer/
certification.py`, `trainer/effective_trials.py` (both TDD); demonstration `experiments/llm_alpha_gauntlet.py`.

## The gauntlet

A strategy's out-of-sample return series is **certified** only if it clears all three gates:

1. **Economic** — the one-sided HAC (serial-correlation-robust) lower Sharpe bound exceeds an economic bar
   `sr_econ_ann` (default 0.5), not merely 0.
2. **Multiplicity** — its Deflated Sharpe Ratio exceeds 0.95 against the expected-max Sharpe for the trial count.
3. **FDR** — it survives Benjamini–Yekutieli (the arbitrary-dependence bound) across the family.

```python
per, family = certify_family(oos_returns, n_trials="effective")
```

## The correction: use EFFECTIVE, not raw, trials

The Deflated Sharpe Ratio deflates the observed Sharpe by the expected **maximum** Sharpe across *n independent*
trials. An automated/LLM search emits thousands of **highly correlated** strategies (variants of a few ideas), so
plugging the raw search count over-deflates and **false-negatives genuine alpha**.

`trainer/effective_trials.py` estimates the effective number of independent trials from the family's return
correlation matrix, in `[1, M]`:
- **participation ratio** `(Σλ)² / Σλ²` of the correlation eigenvalues — robust; **use this** (independent → M,
  one dominant factor → 1);
- **Li-Ji (2005)** — a cross-check, but its fractional term is ill-behaved when one eigenvalue dominates, so it is
  not exactly 1 under heavy redundancy. Participation ratio is the recommended estimator.

`certify_family(..., n_trials="effective")` deflates by this count; the family dict always reports
`effective_trials` regardless of the mode.

## Why it matters (demonstrated)

On a 2,031-strategy market-timing zoo over a survivorship-biased 64-year equal-weight equity basket, embed **25
correlated variants of one genuine Sharpe-1.0 idea**. The family's effective trials is **3.6**, not 2,056:

| Deflation | SR\* deflation level | genuine idea certified |
|---|---|---|
| **Raw-N** (2,056) | ~0.85 ann | **0 / 25** (false-negative — the real idea is silently killed) |
| **Effective-N** (≈4) | ~0.26 ann | **25 / 25** (correctly recovered) |

The bare zoo (no injected alpha) certifies **0** under *both* deflations, and clean injected alpha of realized
Sharpe ≥ ~1.3 certifies while ~0.8 does not — so the harness has power and controls false positives. The
correction only relaxes the multiplicity gate to the honest number of independent bets; it does **not** create
false positives on a genuine null.

## Honest scope

- This is **not** an LLM-specific or novel result. "Under-reporting the trial count inflates DSR" **is** the
  Deflated-Sharpe thesis; a strategy zoo dying under multiplicity is established (Harvey–Liu–Zhu 2016;
  McLean–Pontiff 2016). The genuine contribution here is the **effective-trials correction** to the deflation.
- The estimator assumes the supplied series are equal-length and reasonably stationary; on non-stationary or
  regime-shifting data the correlation (and thus `n_eff`) drifts — treat it as an estimate, not a constant.
- The gauntlet certifies *statistical + economic* survival, not tradeability net of realistic execution; pair it
  with a cost model. An actual LLM discovery loop (certifying real LLM proposals end-to-end) is future work.
