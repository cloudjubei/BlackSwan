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

#### PHASE 1 RESULT (free data) — DONE, gate closed on the free layer
The free `info` endpoint gives ONLY funding+premium history + candles; **historical OI, observed liquidations
and wallet-attributable fills are NOT free** — they live in the requester-pays S3 archive (`hyperliquid-archive`
/ `hl-mainnet-node-data`; the user's AWS creds are present here). So Phase 1 tested the cheap positioning-
PRESSURE proxy (premium = mark−oracle) as the gate.
- HL-12 cut (`scripts/fetch_hyperliquid.py`, `experiments/hyperliquid_positioning.py`+`_intraday.py`): fade of
  positioning extremes = powered-null/negative, cost-robust. An **adversarial verification workflow
  (`verify-hyperliquid-negative`, verdict NOT-trustworthy) caught an overclaim** — only the FADE sign was tested;
  the underpowered 12-coin sample makes either sign unreliable. Redirect: POWER the question on a broad panel.
- DECISIVE broad panel (`scripts/fetch_cex_perps.py`→`cexperps/`, `experiments/positioning_factor_powered.py`):
  Binance **467 USDT perps × 6.6y** (premium-index = unclamped positioning pressure, free). **Panel cross-
  sectional IC** (N=coins×days): full-sample premium→next-day IC **−0.0114, t=−4.65** (orthogonal to price
  momentum) ⇒ **REVERSION** (fade crowded longs) has real forward content, not continuation. **By era:**
  pre-2021 −0.029 (t−2.63), 2021-2023 −0.021 (**t−4.47**), **2023-2026 −0.002 (t−0.84, GONE)**. Tradeable fade
  factor: +0.5…+1.5 gross → **+0.06…+0.39 inconclusive at 4.5 bps taker, 0 survivors, 0 BH/BY**; era: **2021-23
  +1.08 CI[+0.04,+2.13] SURVIVOR → 2023-26 −0.96 powered-null**. MDE_ann≈0.98 (set by YEARS, not breadth).
- **VERDICT:** free crypto-perp positioning-premium **reversion was real & significant 2019-2023, DECAYED to a
  statistical null by 2023-2026 (McLean-Pontiff crowding), and was never net-of-cost robust.** A clean LAW/NULL
  on a novel free dataset — the honest ceiling, not a printer. (Open caveat: panel IC may be partly a 1-day
  microcap bounce — liquid-only + skip-a-day robustness not yet run.)
- **S3 GATE (the genuine cost/strategy decision):** the free layer shows positioning edges get arbitraged away as
  the venue matures → **lowers the prior** on a durable forced-liquidation printer. Pursue the paid S3 layer ONLY
  as the wallet-attributed "who captures the forced-flow rent" MICROSTRUCTURE paper (survives even if the edge is
  dead), NOT a money-printer hunt. Tooling: `hac_verdict` now `periods_per_year`-parameterized (root, non-breaking).

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

#### PROOF-SEED RESULT (synthetic) — FAILED verification, landmark framing DEAD as-is
Built `trainer/mutual_information.py` (TDD, 14 tests, 4 estimators in nats — SURVIVES as reusable tooling) and a
synthetic KMZ RFF-ridge proof-seed `experiments/complexity_mi_law.py`. Adversarial verification
(`verify-mi-complexity-law`) returned **recordable=false**:
- **Only survivor:** gross OOS timing skill is monotonically MI-gated and leakage-clean (shuffle test collapses
  OOS IC to ~0). A sanity gate, not a law.
- **The reconciliation was a REGIME ARTIFACT.** A faithful-KMZ rebuild (flat coefficient spectrum, genuine ridge
  z on the sample covariance, RFF normalized √(2/P), dual solve to c=20) **REPRODUCES KMZ's out-of-sample virtue**
  — with z=100 the most-complex model is globally optimal (OOS Sharpe → ~3.08 at c=20). My "moderate-c optimal /
  no double-descent" came from an inert ridge (all three λ were the same near-ridgeless fit), unnormalized
  features, c capped at 5, and a mis-specified double-descent statistic. **So we do NOT refute KMZ — they
  reproduce in their regime.** (KMZ is an OOS claim; I'd mislabeled it in-sample.)
- Cost axis inert on i.i.d. data (turnover complexity-invariant → the "net-of-cost" figures were gross); MI\* is
  not a constant (comparison/power-dependent); the MI axis is a Gaussian upper bound.
- **Honest path if continued (more MODEST, not a bombshell):** faithful-KMZ regime + time-series PERSISTENCE
  (AR(1), so cost can bite) + REAL financial targets (realized-vol high-MI/high-persistence vs daily-return
  low-MI), because MI and persistence/cost are confounded in reality — the actual empirical question. **DECISION
  POINT:** is the war-ending landmark worth chasing when KMZ reproduce, or reframe to the modest empirical study?
- META: 3rd overclaim caught by adversarial verification this session — verification before recording is now a
  hard rule for any BlackSwan "landmark".

#### OPTION 1 (corrected reframe) — EXECUTED → a MODEST honest result, not a landmark
New TDD tooling: `trainer/complexity_ladder.py` (7 tests — faithful RFF + genuine ridge z on the sample
covariance, dual/primal auto-solve, ridgeless-interpolation & sklearn-equivalence pinned) + the earlier
`trainer/mutual_information.py` (14 tests). Two experiments:
- **De-confounding synthetic** (`experiments/complexity_mi_persistence.py`) dials MI × persistence(φ)
  independently. With genuine ridge, KMZ's OOS virtue reproduces (OOS IC → max complexity). The net-of-cost
  complexity premium is **positive & significant at every (MI×φ) cell, rising with MI, flat across persistence**;
  the shrunk complex model doesn't churn more. ⇒ in a **stationary** world there is **no MI\* threshold and no
  persistence/cost gate** — complexity just pays. So the "complexity fails" regime must come from
  **non-stationarity**, not MI or cost.
- **Real data** (`experiments/complexity_real_targets.py`; 88 assets equity+crypto+commodity, up to 46y,
  faithful ladder, temporal 60/40, 5 bps; leakage-guard passed — shuffled-target IC ~0): the complexity premium
  in **OOS IC scales with the target's real MI** — ret1 (next-day) +0.000 flat / ret5 +0.010\* / **rvol5
  (realized vol) +0.176\*** (IC 0.15→0.33). But the **net-of-cost timing Sharpe on tradeable return targets is
  negative at every complexity** (ret1 −0.34…−0.46, ret5 −0.21…−0.37) — cost kills them, complexity doesn't
  rescue.
- **Honest reconciliation (defensible, modest):** complexity's value is MI-gated (synthetic + real agree); on
  real data **MI and tradeability are confounded** — high-MI targets (volatility) reward complexity in predictive
  skill (**KMZ right**) but aren't directional-timing-tradeable, while the tradeable return targets are low-MI
  where complexity's gross edge is tiny and net-of-cost negative (**Nagel/Buncic right**). The "war" dissolves —
  the camps test different MI regimes/target types. A clean empirical characterization, **not** a war-ender.
- **VERDICT:** option 1 yields a modest, honest empirical paper; the durable asset is the TDD tooling. **Decision
  point:** whether to invest further here or swing at the seminal-ceiling **option 3** (Deflated Sharpe for the
  LLM era — our apparatus as the certification standard for machine-discovered alpha).

Run both in parallel (zero shared dependencies).

### Option 3 "Deflated Sharpe for the LLM era" — EXECUTED → INFRASTRUCTURE, not a landmark
TDD `trainer/certification.py` (8 tests): `certify_family` = a 3-gate PASS/FAIL (ECONOMIC HAC-lower-bound >
sr_econ + MULTIPLICITY Deflated-Sharpe + BY FDR) composing the already-tested sharpe.py primitives. Demo
`experiments/llm_alpha_gauntlet.py` (2031-strategy timing zoo, survivorship-biased 64y equal-weight equity): the
zoo is an **effective null OOS** (5.1% nominal-sig ≈ the 5% false-positive rate) → **0 certified**; the gauntlet
has **power** (injected Sharpe 1.5/2.5 certify, 0.8 doesn't) and **FP control** (permutation null 0/2031).
- **Adversarial verification (`verify-llm-gauntlet`) demoted it — recordable only as honest infrastructure.** Both
  advertised "LLM-specific failure modes" were WITHDRAWN: "hidden multiplicity" **is** the Deflated-Sharpe thesis
  (DSR catches it: 0.998@n=1 → 0.305@raw-N), and "pretraining-cutoff look-ahead" was pure market **beta** (top-20
  by IS Sharpe: OOS raw +0.25 via beta +0.49, **beta-neutral residual −0.13** — the picks never had alpha). No LLM
  is modeled. The negative (timing zoos die under multiplicity) is established (Harvey-Liu-Zhu 2016;
  McLean-Pontiff 2016) — known statistics correctly composed, not a new method.
- **The one genuine contribution — EFFECTIVE TRIALS — is now BUILT (#1 done).** `trainer/effective_trials.py`
  (11 tests): effective # independent trials in [1,M] via the participation ratio `(Σλ)²/Σλ² = trace(C)²/ΣC_ij²`
  (no eigendecomposition; Li-Ji as a fragile cross-check). Wired into `certify_family(..., n_trials="effective")`.
  **Rescue demo:** 25 correlated variants of one genuine Sharpe-1.0 idea → effective trials 3.6 vs raw 2056;
  certified **0/25 under raw-N (false-negative) vs 25/25 under effective-N**. Mirrored to modeltrainer
  `deflatedSharpe.ts` (8 golden-pinned tests); standard note `docs/CERTIFICATION_STANDARD.md`. 89 rigor tests green.
- **Track B (actual LLM discovery loop) — DONE → honest NEGATIVE.** Real Claude agents proposed 40 executable
  strategies = the published anomaly canon (Faber, TSMOM, RSI-2, Bollinger), honest expected Sharpe 0.42.
  Backtest+certify (`experiments/track_b_llm_discovery.py`, effective-trials): 2/40 certify, ~4.8 effective ideas.
  The apparent "LLM adds value" (+0.19 vs random zoo −0.03, t=7.35) is an **unfair-baseline artifact** — vs
  random params of the *same types* it's +0.192 vs +0.155, **t=1.19 (n.s.)**; the LLM edge is entirely *picking
  good types* (recalling the canon), not design. And it's **decayed** (beta-neutral Sharpe 2005-15 +0.28 →
  post-2015 +0.06). An LLM proposing strategies just recalls decayed published anomalies — **no live LLM alpha.**
- **Next:** deep-research-again for a genuinely novel direction we missed; Avenue-3 guide is last (per plan).

### DEEP-RESEARCH v2 — the honest re-survey (7 frontiers + adversarial novelty judge, 129 prior-art searches)
Blunt verdict: **for ALPHA on free data the field is genuinely picked-over for a small team.** Everything that
survives is a **law / null / benchmark / measurement "referee" contribution — not a bombshell**, and most likely
outcomes skew deflationary. Prior-art hits: the information-theoretic predictability ceiling is now *owned*
(arXiv 2606.27100, 2026); **our Hyperliquid liquidation edge was partly scooped** — Garcia Seuma (arXiv
2607.27070 + 2608.03616) measured the cascade branching ratio on the *identical* S3 archive and found it
subcritical, so the gated S3 wallet-attribution bet is deflated and the on-chain cluster is a **race, not a moat**.
- **TOP PICK → the Overfitting-Deflation-Law for the neural/RL regime.** Verified-open: the one exact
  in-sample-overfit result (arXiv 2501.03938, Jan 2025) is *linear-only* and explicitly names neural/RL +
  "effective trials for general gradient-based search" as unsolved. Fit `E[max spurious Sharpe]=f(capacity,
  compute,T)`; deliver a plug-in **effective-DoF haircut reported without counting trials** + a demo that
  DSR/PBO/CPCV are anti-conservative for gradient-discovered strategies. **Directly extends today's
  effective-trials work** (enumerated→gradient search) and uses our un-copyable ~20k-run ladder. Law/infra, not
  alpha. Unlocks a search-deflated re-adjudication of Gu-Kelly-Xiu ("does ML really beat linear once charged for
  its effective trials?").
- **Runners-up:** CausalArb (causal-discovery benchmark with arbitrage-certain ground-truth; no data moat);
  Identified-Set-of-Dealer-Gamma (partial-ID bounds replacing the GEX point-sign heuristic; crypto ground-truth);
  Detectability-Horizon (a decaying edge has convergent lifetime evidence → a provable un-certifiable region; a
  decay-aware DSR that formalizes *why* the alpha hunt keeps failing).

### TOP-PICK PURSUED (Overfitting-Deflation-Law) — BUILT, verified, → FOOTNOTE not a landmark
The deep-research #1 survivor was implemented (`trainer/effective_trials.py` gained `effective_dof_from_sharpe` +
`effective_trials_from_max_sharpe`, TDD; `experiments/overfitting_deflation_law.py`) and adversarially verified →
**recordable=false.** All three headline pillars were wrong: (1) the "PSR 0.500 anti-conservatism" was a
**tautology** (benchmark set to the observed Sharpe → PSR=Φ(0)=0.5 for any input); (2) a **fabricated citation**
— the docstring claimed arXiv 2501.03938 "names the neural case as open"; it does not (linear-only, zero mention
of neural/gradient) — I'd propagated the research agent's unverified claim as fact; (3) "compute inflates
effective DoF" is **textbook** (Ali-Kolter-Tibshirani 2019 early-stopped-GD≈ridge; R. Tibshirani 2015 search-DoF;
Nakkiran 2019 EMC; arXiv 2602.13442), not open. Honest residual: a leakage-clean harness + a null-calibrated,
held-out overfit haircut for gradient strategies — modest DSR-plumbing, not a law. Experiment rewritten honestly
(fabrication removed, tautology → held-out demo, real citations added). The tested DoF primitives survive.

### FINAL TALLY — all three bombshell swings + BOTH deep-research picks tested; NONE is a landmark
- **DATA bet** (positioning): free reversion real 2019-23, **decayed to null** by 2023-26 — a clean null.
- **METHOD bet** (MI-complexity): a **modest** reconciliation; KMZ reproduce in their regime — no war-ender.
- **Option 3** (certification): **infrastructure** + a correct null-zoo negative — not novel, not LLM-specific.
The research-director honest ceiling ("laws and nulls, mandatory infrastructure, not a money-printer") is fully
confirmed. **The most robust finding of the whole effort is meta:** adversarial verification caught an overclaim
on *every* swing (4×) — the durable asset is the rigor apparatus + the discipline of running it against our own
claims, not any single result. Remaining live options: pursue the effective-trials correction as a real
methods contribution; run Track B; or the gated S3 wallet-attribution paper. Do NOT record any of the swings as
a landmark.

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
