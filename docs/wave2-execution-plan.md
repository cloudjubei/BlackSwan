# Wave-2 Execution Plan — Replicate + Falsify Trading-ML Methods Under Real Costs (BlackSwan)

Wave 2 = replicate published trading-ML methods inside BlackSwan, under REAL 0.1%/trade costs, on
out-of-sample walk-forward windows, and record **claimed-vs-measured**. The papers are catalogued in
`BlackSwan/.factory/trainer.json` `papers[]` with draft `replicateConfig`s. This plan was validated
against the live code by a per-paper audit (each finding adversarially re-verified at file:line), a
rigor-gate design pass, and a completeness critic. It sequences the launchable runs, names the shared
infra to build, and fixes the verdict criteria.

> Status of this pass: configs audited + corrected; the 5 no-infra replications are launchable today;
> the rigor/verdict layer (enablingBuild #1) is NOT yet built, so runs produce raw claimed-vs-measured
> rows, not graded verdicts, until it lands.

## Results (live, raw — formal grade awaits the DSR/verdict layer)

**Moskowitz TSMom (#7) — REFUTED on single-asset BTC spot.** First completed replication (2026-06-27):
the deterministic `momentum` model, long/flat, fixed sizing, 0.1% fees, swept over lookback 30/90/252 ×
walk-forward 2022/2023/2024. `return_vs_hold_pct` per cell:

| lookback | 2022 (bear, hold −65%) | 2023 (bull, hold +153%) | 2024 (hold +38%) | windows > hold |
|---|---|---|---|---|
| 30  | **+19.5%** | −70.6% | **+6.6%** | 2/3 |
| 90  | **+28.2%** | −145.9% | −41.5% | 1/3 |
| 252 | +65.3% (0 trades, window-limited) | −93.0% (1 trade) | 0 trades | degenerate |

No lookback beats buy-and-hold in ALL three windows → fails criterion 1 (no DSR needed). The signature is
the textbook one: momentum is **downside protection** (wins the 2022 bear by going to cash) but
**trend-lags in bulls** (loses 2023/2024), and against a long-biased B&H benchmark it nets out negative.
Note: the 1d walk-forward windows are only ~181–364 bars, so lookback 252 is degenerate (rarely/never
trades) — meaningful momentum lookbacks here are ≤ ~60. Verdict matches the pre-registered thesis.

**Bysik & Slepaczuk (#1) — REFUTED as a strategy; the skeptical thesis HOLDS UP.** `supervised-gbm`,
1h+1d, 0.1% fees, swept `prob_threshold` 0.5/0.6/0.7 × the three windows. `return_vs_hold_pct` / trades:

| `prob_threshold` | 2022 (bear) | 2023 (bull) | 2024 | avg trades | windows > hold |
|---|---|---|---|---|---|
| 0.5 | −262% (1592) | −377% (1593) | −210% (998) | 1394 | 0/3 |
| 0.6 | −58% (432) | −190% (658) | −107% (390) | 493 | 0/3 |
| 0.7 | +56% (5) | −78% (7) | −37% (4) | 5 | 1/3 |

The central Wave-2 falsification, confirmed. It's a U-shape: low threshold → ~1400 trades → turnover
annihilates returns (−210% to −377% vs hold); raising the threshold cuts turnover (1394 → 493 → **5**
trades) and "restores" returns **only by converging to cash** (thr 0.7 ≈ 5 trades, ~0% return). It beats
hold in exactly one cell — 2022, where being flat beats a −57% market (not skill, just not trading).
**No threshold yields a tradeable edge over buy-and-hold.** The paper's skeptical claim (sign-based ML is
fluff once 0.1% fees apply, because turnover dominates) replicated decisively; its constructive
"cost-aware filter restores it" half does not transfer to beating B&H on single-asset spot.

## The BAR (what "survives" means)
A replication SURVIVES only if it: beats buy-and-hold **out-of-sample net of 0.1%/trade fees**, with
profit **NOT concentrated in up-regimes** (genuine timing, not beta), **stable across seeds AND across
the 2022/2023/2024 walk-forward windows**, and clears the **Deflated Sharpe Ratio** multiple-testing
gate at trialsN=102. Objective = `traded_return` (direction max), per `trainer/summary.py`. There is
currently NO risk-adjusted/Sharpe reward facet (old `differential_sharpe` removed); Sharpe-objective
papers proxy it with `combo_direct=1` (a documented deviation that caps them at PARTIAL until
enablingBuild #2 lands).

## Per-paper claimed-vs-(predicted)

| # | Paper | Model lever | Claimed (paper) | Predicted under BAR | Runs | New infra to LAUNCH |
|---|-------|-------------|-----------------|---------------------|------|---------------------|
| 1 | Bysik & Slepaczuk 2026 (ML BTC under costs) | supervised-gbm | Sharpe ~1.09 / ~65.4% net 10bps, marginally > B&H | likely-refuted (naive sign collapses under cost; 2024 win = up-regime beta) | 36 | none |
| 2 | Théate & Ernst 2021 (TDQN) | dqn | Sharpe 0.404 vs B&H 0.369 | likely-refuted (no Sharpe objective, no augmentation, 1-bar memory, BTC long-beta) | 9 | none |
| 3 | Moody & Saffell 2001 (RRL) | reppo-custom | DSR > profit-max for consistency | likely-refuted; PARTIAL-capped (DSR objective absent) | 9 | none to launch |
| 4 | Lim et al 2019 (Deep Momentum) | duel-dqn-custom-lstm | >2x Sharpe vs classic momentum, survives to ~2-3bps | likely-refuted (tested at ~4x the cost cliff, single-asset, 2-unit LSTM) | 9 | none |
| 5 | Borrageiro et al 2022 (RRL crypto) | reppo-custom | IR 1.46 / 350% | likely-refuted; PARTIAL-capped (71% is funding carry, absent on spot) | 9 | none to launch |
| 6 | Zhang/Zohren/Roberts 2020 (Deep RL) | dqn/a2c/ppo-custom | Sharpe 1.288 / 1.05 | likely-refuted (diversification-driven Sharpe can't transfer to single BTC) | 27 | none after draft fixes |
| 7 | Moskowitz/Ooi/Pedersen 2012 (TSMom) | momentum (BUILT) | ~1.2 Sharpe, 58-instrument | likely-refuted (single-asset spot, long/flat, net fees) | 9 | none — `momentum` model + `momentum_lookback` lever shipped |

**Direction (2026-06-26):**
- **#7 Moskowitz — BUILD the momentum model (active).** `hodl` is plain buy-and-hold, so the draft would
  tautologically tie the B&H baseline TSMom must beat. Rather than park it, we build a real
  time-series-momentum model lane (enablingBuild #3, now promoted to active) — an interesting baseline in
  its own right and a reusable non-RL control. Moskowitz's stored config retargets the new `momentum`
  model once it lands.
- **#6 Zhang — DEFER as a multi-asset precursor (out of scope now).** Vol-target / variable position
  sizing (letting the *same* asset be held at different sizes over time — scaling in/out) is the real
  content of Zhang's vol-scaling, and it is the natural stepping stone INTO the multi-asset portfolio
  project (per-asset weights = variable sizing generalized across assets). It is out of scope as a
  single-asset in-place Wave-2 run; moved to "Deferred to Wave 3 (precursors)". No config fix needed now.

## Launch config corrections
- **combo_direct must be isolated.** A bare `combo_direct:1` under `combo_unified` composes ADDITIVELY
  with the inherited combo_all2 shaping (combo_sell=1000, combo_positionprofitpercentage=10, etc —
  `config_builder.py:186-188` only overrides keys present in cfg; `base_crypto_env.py:512-558` adds
  `direct` on top of shaping). Every RL replication must explicitly zero
  combo_sell/buy/positionprofitpercentage/wrongaction (and combo_noaction where relevant). *(Build/apply
  per-config; the cleanest root fix is to make `combo_direct` imply isolation in `config_builder`.)*
- **lookback is inert at 1d.** `fidelity.py:64` hardcodes lookback=1 for 1d; only the 1h path honors
  `cfg.lookback_window` (`config_builder.py:80`). TDQN drops the inert lever; daily-cadence papers that
  need real memory must move to 1h or accept a 1-bar+indicator state. *(Latent bug — a `lookback_window`
  set at 1d silently does nothing. Either honor it on the 1d branch or reject it at validation.)*
- **vol_target range is [0.005, 0.1].** Zhang's draft 0.15 is out-of-range (would throw) → clamped to
  0.03 so vol-scaling actually engages (0.1 effectively clamps to all-in).
- **walk_forward_window choices are strings** ('2022'/'2023'/'2024') to satisfy choice validation.
- **Supervised line ignores reward levers** — Bysik drops reward_model/combo_direct (inert, misleading)
  and adds the `prob_threshold` sweep (tests the paper's cost-aware confidence filter).
- **Silent stop_loss** — Borrageiro sets `stop_loss:null` (`config_builder.py:126` injects 0.02 otherwise).

## Run budget
**Total = 102 runs**, matching the program-level trialsN for the DSR gate: 36 (Bysik) + 9 (TDQN) +
9 (Moody) + 9 (Lim) + 9 (Borrageiro) + 27 (Zhang) + 3 (Moskowitz). Orders 1–6 (99 runs) are launchable
on existing infra; order 7 (3 runs) is gated on the momentum lane.

Single run (CLI): `.venv/bin/python -m trainer.run --config-json <configPath> --summary-out <out.json>`
(the manifest's `run` command). The hub expands a `replicateConfig` into the window×seed×sweep matrix.

## Enabling builds (priority order — shared infra collapsed)
1. **DSR gate + Wave-2 verdict layer (M, rigor keystone, shared by ALL 7).** `trainer/sharpe.py` (pure:
   `sharpe_ratio`, `deflated_sharpe_ratio` per Bailey & López de Prado, `min_backtest_length`) + emit
   `oos_sharpe`/`n_obs`/`ret_skew`/`ret_kurt` from `summary.py`; then `src/wave2Verdict.ts` — per-window
   MEDIAN aggregation (replacing the `viewer/hypothesis.js` `max()` cherry-pick), DSR deflation at
   `WAVE2_TRIALS_N=102`, the up-regime/beta gate, seed/window stability, and `deriveWave2Status →
   replicated|partial|refuted`. Persist via an `evaluateWave2Replication` tool + `wave2Status`/measured
   block on `TrainingPaperRecord`. NOTE: `summary.py` currently emits NO per-run Sharpe
   (`test_summary.py:347-357` asserts risk metrics are dropped) — the build must ADD it, not just consume it.
2. **Risk-adjusted (differential-Sharpe) reward facet (L, shared keystone across the Sharpe cluster).**
   A new `combo_sharpe` term in the `combo_unified` branch of `base_crypto_env.py` + lever wiring. Until
   it lands, the 5 Sharpe-objective papers are capped at PARTIAL (they cannot test their TRUE objective).
3. **Time-series-momentum baseline model lane (M, ACTIVE — greenlit 2026-06-26).** A reachable `momentum`
   model (sign of trailing-N return → long/short, periodically re-evaluated + re-sized) + a `config_builder`
   route (today everything non-hodl/non-supervised falls through to RL and errors) + a `momentum_lookback`
   lever. Unblocks Moskowitz and gives a reusable non-RL deterministic control for every future run.
   Building this next.

## The rigor gate (verdict layer)
Map the OOS run-distribution (all window×seed runs) to status in `deriveWave2Status(runs, trialsN=102)`.
Each window must have ≥2 seeds and the replication must cover all 3 windows (else status=untested). Use
per-window MEDIAN across seeds — never max().

- **REPLICATED** — ALL of: (1) median `return_vs_hold_pct` > 0 in EVERY window; (2) DSR > 0.95 at
  trialsN=102 on the best window AND test window meets min-backtest-length; (3) upRegimeShare < 0.70 with
  non-trivial flat/down realized_pnl; (4) seedStable (no single seed flips the sign); (5) health 'ok',
  clears the trade gate. PLUS no BLOCKING fidelity gap (a Sharpe-objective paper on combo_direct, or a
  method needing a non-existent lever, caps at PARTIAL).
- **PARTIAL** — beats B&H in ≥2/3 windows but fails the DSR gate, the up-regime/beta gate, or seed/window
  stability; OR all four met but a recorded fidelity DEVIATION exists (e.g. Sharpe objective proxied by
  combo_direct).
- **REFUTED** — median `return_vs_hold_pct` ≤ 0 in ≥2/3 windows, OR profit is up-regime-only and vanishes
  in flat/down windows, OR the run is degenerate/under-traded.

The pre-registered skeptical `verdictNote` attaches to the claimed-vs-measured row in every case.

### Up-regime attribution — two views, not a replacement
The existing up-regime data (`summary.py` `_regime_trend`) buckets each trade's P&L by the regime of its
**entry bar**. A position opened in a flat/down bar that then rides a bull leg books its profit in the
flat/down bucket — so `upRegimeShare = up_pnl / total_pnl` **UNDER-counts beta** exactly for the
buy-and-hold-like policies Wave 2 most needs to catch, and can be gamed by entering in flat bars and
holding through the rally. **Decision: keep the entry-bar view AND add an alternative view** rather than
replacing it — the two together are the signal (a large gap between them is itself diagnostic). The
alternative view attributes P&L by **exposure / holding period** (which bars the position was actually
open across), or, simpler and more robust, regresses strategy returns on market returns and reports the
**beta / up-vs-down capture ratio**. The verdict gate reads the exposure-weighted view; the UI surfaces
both side by side. Until the second view exists, treat the `< 0.70` entry-bar threshold as advisory only.

## Recommended first run
**Bysik & Slepaczuk 2026 — supervised-gbm (36 runs, no new infra).** Cheapest, highest-constructive
value, no RL loop (sklearn `HistGradientBoostingClassifier`), the central falsification, and it exercises
every existing summary/benchmark/regime code path. Record claimed (Sharpe ~1.09 / ~65.4% net) beside
per-window+per-seed measured; aggregate per-window MEDIAN (not max). The graded replicated|partial|refuted
STATUS waits on enablingBuild #1 — until then it is a recorded claimed-vs-measured row.

## Deferred to Wave 3 (structurally untransferable to single-asset BTC spot)
- **Variable / vol-target position sizing (Zhang's vol-scaling) — the multi-asset precursor.** Letting the
  same asset be held at different sizes over time (scaling in/out) is the single-asset shadow of per-asset
  portfolio weighting. It is the natural stepping stone INTO the multi-asset project, NOT a single-asset
  Wave-2 run — so Zhang (#6) is deferred here as that precursor. (`position_sizing:vol_target` exists today
  as an all-or-vol-clamped sizing; true fractional scale-in/out is the bridge to build first in Wave 3.)
- **Jiang et al 2017 (EIIE/PVM portfolio).** A 12-coin portfolio-vector-memory allocator; single-asset
  `ppo-custom` cannot realize cross-asset allocation at all. Orphaned in the prior draft — deferred here.
- Cross-sectional portfolio aggregation (Zhang 50-contract, Lim 88-contract diversified Sharpe),
  perpetual-swap funding carry (Borrageiro ~71% of headline), and artificial-trajectory augmentation (full
  TDQN). All untransferable to single-asset BTC spot; their single-asset directional kernels still run in
  Wave 2 as falsification tests.
