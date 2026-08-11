# Per-anomaly conclusions — why the published edges were believed, and why they don't survive

For each reproduced claim: **the claim**, **why it should work** (the economic story), **why the original
paper concluded it works** (sample + method), **what went wrong** (the methodological hole our discipline
exposed), and **current status** (our verdict + the broader literature). All our tests use the free,
survivorship-free daily panel (7 commodities + SPY/TLT/IEF/UUP, 2006–), 17 out-of-sample windows 2008–2024,
realistic 5 bps/side cost, mutation-proven leakage guards, Deflated-Sharpe / best-of-N multiplicity
correction, and independent adversarial verification.

The three papers that frame *all* of these: **McLean & Pontiff (2016)** — published predictors decay ~58%
out-of-sample; **Harvey, Liu & Zhu (2016)** — with the true multiple-testing burden most anomaly t-stats need
to clear ≈3.0, not 2.0; **Bailey & López de Prado (2014)** — the Deflated Sharpe Ratio, a single backtest's
Sharpe must be discounted for the number of trials that produced it. Nearly every hole below is one of these.

---

## 1. Mou (2011), "Front-Running the Goldman Roll" — **DISPROVED** (scope: energy)

- **Claim.** The GSCI's mechanical monthly roll (sell the front contract, buy the next) is so large and
  calendar-predictable that front-running it — shorting the M1–M2 calendar spread into the roll window —
  earns ≈3–4%/yr.
- **Why it should work.** Index funds are price-*insensitive*, forced, pre-announced traders; limits to
  arbitrage mean few players can absorb the flow, so the predictable price pressure is capturable.
- **Why the paper concluded it.** ~2000–2010 sample, **pooled across ~24 GSCI commodities**, large pooled
  t-statistics.
- **What went wrong.** (i) *Selection / no cost-and-multiplicity discipline*: the tempting +3%/yr is a
  max-Sharpe **selected-cell illusion** — the unbiased cross-config mean is ≈+0.5%/yr (~6× smaller), tail-
  driven by the 2009 GFC / 2020 COVID super-contango, and never DSR-significant even against a *zero* bar.
  (ii) *Decay*: after ~2005 the roll became crowded and index providers diversified their roll schedules
  ("roll enhancement"), so the pressure dispersed. Our energy-basket reproduction is *negative* (demand/
  storage seasonality opposes the roll).
- **Current status.** Largely arbitraged away. **Scope caveat:** we tested WTI + the energy complex only;
  Mou's uncorrelated metals/ags/livestock legs need paid term-structure data and remain *untested* — "no
  evidence for," not a refutation of, the full pooled basket.

## 2. Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum" — **DISPROVED** (this panel)

- **Claim.** The *sign* of an asset's own trailing 12-month return predicts its next-month return; a
  vol-scaled, diversified 58-instrument trend portfolio earns a Sharpe well above 1.
- **Why it should work.** Behavioral under-reaction to news followed by delayed over-reaction; and a risk
  premium for taking the other side of hedgers' demand.
- **Why the paper concluded it.** **1985–2009** sample, ~58 futures across 4 asset classes, strong in-sample
  t-stats robust across instruments.
- **What went wrong.** *Post-publication decay*, textbook McLean-Pontiff. On our free panel the edge is
  present **exactly in their sample era** (pre-2012 annualized Sharpe **+0.51**, and our backtest reproduces
  the canonical 2008/2010 trend years — a positive control proving the test *can* see trend) and **vanishes
  after** (2012–2024: **−0.18**); the full-sample t is ≈0. It is a *gross* null (gross ≈ net), not a cost
  kill. The managed-futures industry lived the same 2011–2019 drawdown.
- **Current status.** Contested. Our verdict is scoped: DISPROVED for a *meaningful, cost-surviving* edge on
  this 11-asset panel; 17 windows cannot exclude a *tiny* sub-0.2–0.36 Sharpe, and a 58-instrument levered
  book is a different (untested) object. The honest reading: the headline magnitude does not survive OOS on
  freely-obtainable data.

## 3. Jegadeesh & Titman (1993), "Buying Winners and Selling Losers" — **DISPROVED** (inverts here)

- **Claim.** Stocks that outperformed over the past 3–12 months keep outperforming; long-winners /
  short-losers earns ≈1%/month.
- **Why it should work.** Under-reaction to firm-specific information; gradual news diffusion; the disposition
  effect (investors sell winners too early).
- **Why the paper concluded it.** **1965–1989 US equity** cross-section, strong t-stats.
- **What went wrong (for the *generalisation*).** Cross-sectional momentum is an *equity single-stock*
  phenomenon. On a commodity-heavy macro basket it **inverts**: the published long-winner/short-loser book is
  significantly *negative* (t_win −2.26, −12%/yr), because commodities *mean-revert* at these horizons where
  equities trend. So the free-data result refutes the *generalisation of the rule*, not equity momentum
  itself (the survivorship-free equity single-stock universe is data-gated and untested here).
- **Current status.** Equity momentum is among the more durable anomalies but decayed post-2000, suffers
  violent crashes (2009), and Harvey-Liu-Zhu flag its multiple-testing context. Our contribution: it does
  **not** transfer to free macro data — the tradeable direction there is *reversal* (see §9).

## 4. Frazzini & Pedersen (2014), "Betting Against Beta" / Baker-Haugen low-vol — **DISPROVED** (this panel)

- **Claim.** Low-beta/low-risk assets earn higher risk-adjusted returns; a beta-neutral BAB factor (long
  leveraged low-beta, short de-leveraged high-beta) earns significant alpha across assets and decades.
- **Why it should work.** Leverage- and margin-constrained investors bid *up* high-beta assets (they "reach
  for beta" instead of levering), so low-beta is systematically underpriced.
- **Why the paper concluded it.** Broad multi-asset, multi-decade sample; large, consistent alphas.
- **What went wrong.** (i) *Construction sensitivity* — Novy-Marx & Velikov and others show BAB's alpha is
  fragile to the rank-weighting, microcaps, and the non-standard beta estimator. (ii) On our free panel the
  long-low/short-high book is a **coinflip** (t≈0, negative net) at every formation window (beta- *and*
  volatility-ranked), and the single edge-like cell is *entirely* the 2008 GFC long-Treasuries/short-oil
  trade — a macro crisis bet, not a low-risk premium.
- **Current status.** Contested. **Scope:** our equal-weight book (realized β ≈ −2.1) is *not* a faithful
  beta-neutralized FP-BAB, and the equity single-stock cross-section where low-vol is defined is data-gated.
  Low-vol persists as a practical *equity* factor; the BAB-specific alpha is debated. On free macro data:
  no low-risk premium.

## 5. Ariel (1987) / Lakonishok & Smidt (1988), turn-of-the-month — **DISPROVED**

- **Claim.** Equity returns concentrate in a 4-day window around the month boundary (last trading day + first
  ~3); the rest of the month earns ≈0.
- **Why it should work.** Month-end/start cash-flow timing — salary and pension contributions, fund flows,
  window dressing — creates recurring buying pressure.
- **Why the paper concluded it.** Long US index samples (LS: ~90 years of the Dow), where the window's
  average return dwarfs the rest.
- **What went wrong.** Publicised → arbitraged; and as a bounded calendar rule it is a multiple-testing
  magnet. On the *exposure-balanced* (market-drift-neutral) spread — which strips the confound that the
  window is simply "long a rising market" — it is **null-to-negative** (gross t_win −0.53, net −1.74).
- **Current status.** Decayed. (Some studies find a residual in a narrower window in specific samples; it is
  not a cost-surviving edge here.)

## 6. Bouman & Jacobsen (2002), "Sell in May / the Halloween Indicator" — **DISPROVED-MARGINAL**

- **Claim.** November–April returns systematically exceed May–October in most equity markets.
- **Why it should work.** Seasonal risk-aversion / liquidity cycles — summer holidays, seasonal-affective
  mood, seasonal fund flows.
- **Why the paper concluded it.** **37 markets, 1970–1998**, positive and often significant in most.
- **What went wrong.** *Multiple testing across 37 markets* and data-snooping (Maberly & Pierce 2004: a
  couple of outliers — 1987, 1998 — drive the US result). On our panel it is the *least-dead* calendar rule
  — a genuine, cost-robust positive *tilt* (only ~2 trades/yr) — **but it fails every bar**: window t_win
  ≈1.9 (< 2, far below the best-of-6 DSR ~2.5), the honest **pooled daily t is only 0.85**, breadth 10/17 =
  59% is a coinflip (binomial p ≈ 0.63), and it is **carried by ~2 of 17 years** (drop 2019+2023 → t_win
  1.10).
- **Current status.** Contested — Zhang & Jacobsen (2021) argue it persists globally; skeptics call it
  data-mining. Our verdict: a real but sub-significant, non-bankable tilt — *disproved-marginal*, explicitly
  weaker than our two mean-reversion open threads.

## 7. French (1980), the Monday / weekend effect — **DISPROVED (decayed to zero)**

- **Claim.** Monday returns are systematically negative; the rest of the week positive.
- **Why it should work.** Information accumulates over the closed weekend; settlement/clearing timing;
  bad-news release patterns.
- **Why the paper concluded it.** **S&P 1953–1977**, a robustly negative Monday mean.
- **What went wrong.** The purest *decay* case in the battery: on our panel the **gross** Monday spread is
  already **null (t_win +0.32)** — the effect is simply *gone* — and trading it only bleeds the high daily
  turnover (net t_win −2.92).
- **Current status.** Dead. A textbook example of an anomaly that disappeared (and partly reversed) after
  publication.

## 8. Gatev, Goetzmann & Rouwenhorst (2006), "Pairs Trading" — **INCONCLUSIVE** (open thread)

- **Claim.** Distance pairs trading — form the closest-moving pairs, fade divergences beyond 2σ, close on
  convergence — earned ≈11%/yr, market-neutral, 1962–2002.
- **Why it should work.** Close substitutes are tied by the Law of One Price; temporary relative mispricings
  revert (relative-value arbitrage).
- **Why the paper concluded it.** **1962–2002 US equities**, large closest-pair universe, robust profits.
- **What went wrong / what we found.** Do & Faff (2010) document a sharp post-1990s decline (arbitraged;
  spreads tightened; decimalisation). On our free macro basket the mean-reversion rule is a **weak,
  persistent, cost-surviving tilt** (annualized ≈+0.35, 12/17 windows, robust to 40 bps) that **does not
  clear multiplicity** (per-config t ≈1.5). Its inverse (chase divergence) is disproved-negative.
- **Current status.** Decayed on equities. On free data it is one of our two **open threads** — see §9.

## 9. What actually survived (weakly): commodity mean-reversion — **INCONCLUSIVE**

The two threads we could **not** cleanly kill — cross-sectional *reversal* (§3's mirror) and *pairs* (§8) —
are the **same weak effect** seen two ways. The "try harder" decomposition localizes it:

- **It is a *commodity* effect.** Commodities-only reversal t_win **+3.90** / pairs **+1.91**; the
  financials-only (SPY/TLT/IEF/UUP) versions are **null** (−0.49 / +0.10). The equity-rate-dollar cross-
  section carries no mean-reversion edge.
- **Same family, not one factor** (daily-P&L correlation +0.38), and **grain/energy-concentrated** (dropping
  wheat cuts reversal t_win 2.00→0.60; dropping the financials *raises* it).
- This is the documented **commodity value / mean-reversion** effect (Asness, Moskowitz & Pedersen 2013,
  "Value and Momentum Everywhere"; commodities mean-revert where equities trend).
- **Why it stays inconclusive, not a survivor:** the commodities-only strength rests on *post-hoc*
  sub-universe selection, a small 7-asset cross-section, and heavy concentration in ~2 grain markets — a
  small-capacity, hard-to-scale residual, not a bankable edge.

---

## The crypto + positioning families (our own pre-registered hypotheses, not single-paper reproductions)

The §B trail (price at every frequency, perp funding, taker order-flow, scheduled events, liquidation
cascades, macro-regime timing, Wikipedia attention, the commodity index roll, macro world-models for
gold/silver/copper, and the entire CFTC COT positioning class across level / Williams-index / flow /
cross-sectional / cross-asset-class) is **all disproved** under the same discipline. Their recurring failure
signature: every tempting single-window cell is **beta** (long a rising asset — ~zero alpha over hold),
**best-of-N multiplicity** (t at the expected maximum under the null), or a **mechanically-coupled mirror**
(the inverse arm carries no independent power); the winning *arm* tracks the year's price regime, i.e. it is
price-timing, already null.

## Meta-conclusion

Across 65 pre-registered probes / 18 families / 1,818 backtested cells, **63 are disproved and 2 inconclusive
— none is a cost-surviving, multiplicity-corrected, out-of-sample edge on freely-obtainable data.** The two
survivors-in-name are a single weak, small-capacity commodity mean-reversion effect. The published edges fail
for a small number of recurring reasons — **post-publication decay** (TSMOM, Monday, turn-of-month, pairs),
**in-sample selection / no multiplicity correction** (Mou, sell-in-May), **construction sensitivity** (BAB),
and **non-generalisation** (equity momentum inverts on commodities) — exactly the pathologies McLean-Pontiff,
Harvey-Liu-Zhu, and the Deflated Sharpe Ratio were written to catch.
