# BlackSwan research plan — the two bets to a landmark (data or method)

> Plan of record. Supersedes the earlier scratch note under `experiments/`.
> Two go-forward bets are in **§ SELECTED BETS**; tracks A/B/C below are the search space they were chosen from.


Verified conclusion (3 prior-art checks, one per avenue): **no bombshell exists in the free-data / factor-zoo
directions.** Each candidate headline is already owned or refuted:
- Avenue 1 ("retail can't win"): settled (Sharpe arithmetic + Barber-Odean + Taiwan/Brazil); strong form FALSE
  (Boehmer et al 2021: retail order flow predicts returns); "who wins" numbers need non-free data.
- Avenue 2 ("info-theoretic equilibrium / capacity saturation"): PREEMPTED ~2 months ago (Noguer i Alonso,
  arXiv 2606.08209 / 2606.27100, Jun 2026); "efficiency = incompressibility" is an econophysics subfield
  (Hasanhodzic-Lo-Viola 2011; Zenil-Delahaye); the OPPOSITE has a JF stamp (Kelly-Malamud-Zhou "Virtue of
  Complexity", JF 2024).
- Avenue 3 ("minimum viable horizon"): headline CONTRADICTED (Anarkulova-Cederburg JFE 2022: ~12% chance a
  30-yr investor loses real); long-horizon inference un-powerable (Valkanov/BRW overlap; ~10 decade-obs).

**A landmark therefore requires a NEW ASSET: either (A) a data edge free data cannot supply, or (B) a genuinely
new method/theorem.** This plan puts BOTH in scope and runs deep research to FIND the specific bets.

---

## TRACK A — DATA EDGE (the only route to a "we found something real" bombshell)

The bet: an edge that is real precisely because the data is exclusive / faster / alternative / underexploited.
Sub-tracks to research + rank by (edge plausibility x accessibility/cost x novelty):

- **A1 — "Who wins" microstructure (Avenue 1's provable core).** Non-free data: Nasdaq TotalView-ITCH / LSE
  Rebuild-Order-Book / CME MDP (participant-tagged message data); CFTC / SEC-CAT audit trails (restricted);
  SEC Rule 605/606 (public, underused); TAQ + OPRA (paid); exchange 10-K colocation/market-data rents (free
  proxy). Deliverable it unlocks: an original "who captures the spread / latency tax" decomposition. Accessible
  wedge = Rule 605/606 + TAQ + 10-K rents.
- **A2 — Alternative data with documented, less-arbitraged, still-live edge** affordable to a small team
  (satellite, card/transaction, web/app, sentiment/NLP, supply-chain, ESG). Which survive McLean-Pontiff decay?
- **A3 — On-chain / crypto-native (the free-ish rich frontier we barely touched).** On-chain flows, DEX,
  mempool, whale wallets, stablecoin flows, exchange in/outflows, liquidations, funding term structure. Where is
  the documented, less-arbitraged edge? (Crypto is where cheap alt-data is richest and least efficient.)
- **A4 — Order-book microstructure + options.** L2/L3 depth, order-flow imbalance, options gamma/dealer
  positioning, 0DTE flow. Edge / access / cost; retail-accessible slices.

## TRACK B — METHOD / THEOREM EDGE (landmark-capable, buildable from what we have)

- **B1 — Refute the "no-model-works" crowd (Avenue 2, a MODELTRAINER program).** Kelly-Malamud-Zhou say Sharpe
  rises with complexity; Nagel/Buncic say it's an artifact; Noguer says architecture doesn't matter past the MI
  ceiling. Find the asset / regime / horizon / data-tier where model COMPLEXITY demonstrably adds NET-OF-COST
  value — i.e. SHOW cases where it works. Our unique asset: ~20k modeltrainer runs across the full ladder
  (linear -> GBM -> LSTM -> transformer -> TSFM -> deep RL), the RL rung empirically unclaimed.
- **B2 — New theory.** Resource-bounded predictability bounds (extend Hasanhodzic-Lo-Viola); a VALID-inference
  estimator for near-zero predictive-information floors (the regime where MI estimators are least trustworthy);
  algorithmic-randomness finance. Landmark-capable, needs no new data.
- **B3 — Regime-adaptive / causal / structural methods.** AMH says predictability is time-varying, never
  permanently dead -> a method that DETECTS and exploits WHEN it is exploitable (regime-switch, causal-structure,
  meta-learning) rather than a static predictor. Where is the frontier with real edge potential?

## TRACK C (side deliverable) — Avenue 3 as a COMPREHENSIVE LONG-TERM-INVESTING GUIDE

Not a novel academic paper — a rigorous, powered, honest GUIDE: the sweet spots, the pitfalls, what to watch
for. Grounded in Anarkulova-Cederburg (long-horizon loss is real), Bessembinder (premium is skew, diversify or
lose), valuation-conditioning (CAPE, honestly), cost/turnover drag, time-diversification truth vs fallacy.
Deliverable: a definitive reference (book chapter / FAJ / long-form), not a JF submission.

---

## SELECTED BETS (from the 8-researcher deep-research workflow `find-the-bombshell`)

Strategic filter: a candidate earns the flagship slot only if our un-copyable assets (powered-null apparatus +
~20k-run complexity ladder + net-of-cost engine) make us DECISIVE, not incidental. Two independent data
frontiers (A1 "who wins" + A3 on-chain) CONVERGED on the same wedge — the strongest signal in the survey.

### DATA BET → Hyperliquid leveraged-positioning wedge (A3 spine + A1 attribution layer)
The ONLY place a small team gets CAT-grade, participant-(wallet)-attributable, real-time microdata FOR FREE —
structurally impossible in every other asset class. Edge is MECHANICAL (forced liquidations MUST transact), not
informational, so it resists the arbitrage that killed funding carry. Rehabilitates our disproved COT
positioning result as a data-resolution artifact; bombshell layer = wallet-attributed "who captures the
forced-flow rent" (what TAQ/ITCH/CAT can never produce).
- FIRST PULL (this week): free `POST api.hyperliquid.xyz/info`; reconstruct aggregate net-leveraged-positioning
  + liquidation-density map (hourly, full history) for top ~10 perps; fills+funding for returns. Two signals:
  (a) positioning-extreme mean-reversion, (b) fade-the-overshoot around dense liquidation clusters. Run BOTH
  through the powered-null (Sharpe CI / MDE / FDR across assets×horizons / HAC) charging honest slippage inside
  the thin liquidation window; control for contemporaneous price to prove an EX-ANTE lead. Tag fills to
  MM/arb-bot/retail wallet cohorts for the "who captures" layer.
- KILL: net-of-cost Sharpe CI includes zero, OR fully explained by contemporaneous price (no ex-ante lead), OR
  MDE > realized effect. (The exact funding-carry / COT failure mode — do not nurse it. The wallet-attribution
  microstructure paper can survive even if the edge dies.)

### METHOD BET → MI-conditioned complexity frontier (the "Virtue of Complexity" reconciliation LAW)
Turns the Kelly-Malamud-Zhou (complexity helps) vs Nagel/Buncic (artifact) vs Noguer (MI-ceiling,
architecture-invariant) three-way war into ONE falsifiable law: the mutual-information threshold (nats) at which
model complexity's NET-OF-COST value crosses zero. Uniquely enabled by our ladder+null; constructive/positive;
every camp must cite it.
- FIRST EXPERIMENT (this week): reuse the ~20k-run ladder. Estimate per-horizon MI budget (nats) with several
  estimators for targets spanning the axis — daily index return (~0.005 nats, low), FX daily (low), REALIZED
  VOL (~0.3–0.7 nats, high), cross-sectional dispersion (mid-high), crypto-perp short-horizon (mid). Plot each
  target's net-of-cost Sharpe by rung (linear→GBM→LSTM→transformer→TSFM→RL) with powered-null CIs. PROOF SEED:
  exhibit ONE high-MI target (realized vol) where rung-ordering STRICTLY predicts net Sharpe with NON-OVERLAPPING
  CIs, vs a low-MI daily-index target where it collapses.
- KILL: can't produce even one high-MI strict-ordering vs low-MI collapse pair, OR the MI rank-order isn't
  robust across entropy estimators → no LAW, only scatter; downgrade to a robustness note, drop landmark framing.

Run both in parallel (zero shared dependencies).

### Tier-1 alternate (hold): "Deflated Sharpe for the LLM era"
Make deflated-Sharpe/FDR/capacity the SEARCH OBJECTIVE + certify LLM-discovered signals against
memorization/look-ahead. Our apparatus is decisive vs the hottest, sloppiest 2026 subfield. Likely a landmark
NEGATIVE + mandatory infrastructure.

### HONEST LANDMARK CEILING (un-inflated, from the research director)
The seminal results here are **LAWS and NULLS, not a durable money-printing edge.** MI-complexity law is the
cleanest path to seminal (ends a famous war with a number) IF the CIs separate; the Hyperliquid wedge is
seminal-capable as the first attributed decomposition + positioning-factor rehabilitation, CONDITIONAL on
surviving the "toy-market" external-validity attack. A durable tradeable free-data edge that is ALSO seminal
remains UNPROVEN — the honest expectation is a law or a rigorous negative that EXTENDS our no-free-edge thesis.
Do not promise a money-printer. Pre-commit that a rigorous negative is a publishable outcome.

Biggest risks: (1) net-of-cost mortality (the null repeating — both bets can die like funding carry); (2)
landmark inflation (toy-market dismissal; wide-CI "scatter not law"); (3) crowding/scoop + mechanism drift
(Rule 605 fixed Sept-2026 drop; Hyperliquid post-JELLY rule changes alter liquidation mechanics mid-sample →
walk-forward across regimes, don't pool).

## Decision criteria for the go-forward bet(s)
edge plausibility (is there really uncaptured alpha / a real theorem?) x accessibility (can WE get the data /
prove the theorem affordably?) x novelty (is it unclaimed / does it beat the named adversary on the axis that
matters — usually NET-OF-COST?). The deep-research workflow ranks every candidate on these and returns the top
data bet(s) + top method bet(s) + first experiments.
