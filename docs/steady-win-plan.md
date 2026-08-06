# BlackSwan — steady-win RL plan

**Goal:** use BlackSwan (the executor) + thefactory-modeltrainer (the engine) to build the best RL model that
delivers a **steady win** — a robust, positive, net-of-cost, out-of-sample edge — and harden both tools as we go.
**Remaining work only**; shipped history lives in git + memory. Contract: modeltrainer `docs/model-training-standard.md`.
Prior map: `docs/wave2-execution-plan.md` + modeltrainer `docs/blackswan-pipeline-map.md`. Engine/loop work
(automation, guardrails) lives in modeltrainer `docs/implementation-plan.md` (§A2/A3/**A4**).

## North star + what "steady win" means

1. **Steady win = the SCORECARD, decided out-of-sample, honest-by-construction** — NOT the reward. A champion must
   beat buy-and-hold net-of-fee in **EVERY** walk-forward window (median across seeds), clear a **deflated-Sharpe**
   threshold (multiple-testing-corrected — ~20,888 configs already searched), pass a **beta / up-vs-down-capture**
   gate (not a closet-long), stay drawdown-bounded, and clear a `trades_per_day` floor. **Steer on reward, decide
   on the scorecard.** The composite "declare champion" verdict (modeltrainer A4.3) is the loop's stop condition.
2. **The bottleneck is upstream, not the algorithm.** Owner-validated (`blackswan-pipeline-map.md`): the
   model/algo/reward/action search is well-explored; small-nets-win/big-overfit is the signature of a
   low-information signal. The three under-explored, highest-value directions are **(1) evaluation rigor,
   (2) problem formulation, (3) systematic process** — plus **asset selection** and **breadth**. This plan spends
   effort there; it does **not** permute more RL algos.
3. **Selection axis = exploitable structure net of cost, not low volatility.** BTC failed on direction-timing, not
   vol (vol is dialable via `position_sizing=vol_target`). Off-BTC targets are chosen for surviving-net-of-cost
   structure a single-asset directional model can capture (see Stage 0).

## Correctness bar — ZERO data-leakage (non-negotiable; governs EVERY stage)

Trading backtests lie when the model can see the future; **we have hit this before, so no result is trusted, and
no stage advances, until leakage is provably absent by a GREEN test suite.** A read-only audit (2026-08) found the
feature math largely causal-by-construction (trailing `rolling`/`ewm` + `shift(1)` only — no
`center=`/`shift(-k)`/`bfill`/`ffill`/`interpolate` in the feature path; separate train/test providers; train-only
supervised scaler; point-in-time macro fusion; base + coarsest layers guarded by `top=index-1`) — but it also found
ONE **live** leak and several unguarded methodological holes. The register is the backlog; the discipline below it
is standing law.

### Leak register (fix + pin each before it can bias a result)

| # | Sev | Leak | Where | Fix + test |
| --- | --- | --- | --- | --- |
| **L1** | ✅ **FIXED** | ~~Middle-layer higher-TF aggregation selects a FUTURE resample row~~ (was leaking up to 141 days ahead). FIXED via `MultiTimelineDataProvider._layer_trim()` subtracting each resampled layer's per-layer `fidelity_offset` trim in `_compute_values`/`_precompute_values`/`get_feature` (+ the oracle). Guarded by an implementation-agnostic **time-prefix causal invariance test** + the 27-combo I2 matrix now covering middle layers; the inverted flag test removed. **327 passed.** Remaining: the 2 `1m`-base **sub-step** combos still raise (need `process_fidelity`, not index math) — excluded, not silently observed. |
| **L2** | 🔴 HIGH | **Selection-on-test** — best-of-N + thresholds chosen on the same window that is reported; no never-touched lockbox, no held-out asset. | ranks on `total_return_pct` over the TEST window (`summary.py:601`); `run.py:_run_one` builds only train+test | Lockbox = a final window id + ≥1 held-out ASSET, forbidden to the selector; scored once at sign-off. Test: ranker never reads lockbox metrics; reported best re-scored on a fresh window + asset. |
| **L3** | 🔴 HIGH | **Multiple-testing** — ~20,888 configs, no deflation; Deflated/Probabilistic-Sharpe math exists but is DEAD CODE. | `trainer/sharpe.py:63-81` unused; selection never feeds `n_trials` | Wire `deflated_sharpe_ratio(returns, n_trials=#configs, trial_sr_std)`; accept only **DSR ≥ 0.95** + `min_track_record_length ≤ oos_n_obs`. Test: K null runs (true edge 0) → top DSR≈0.5, gate rejects. (= modeltrainer **A4.3**.) |
| **L4** | ✅ **FIXED** | ~~Silent train-window truncation~~. `require_data_present` (wired `run.py:120`) now asserts EVERY requested `(year,month)` resolves for a FIXED window (`len(present)==len(requested)` per resolved timeframe), raising with the shortfall; open-ended `oos-*` windows (`meta.test_to=='latest'`) are exempt (intentionally truncated to disk). Tests: partial-fixed → raises; partial-open-ended → allowed. |
| **L5** | ✅ **FIXED** | ~~No reproducibility fingerprint~~. `summary._provenance_fingerprint` stamps `{gitCommit, gitDirty, configHash, dataVersion=hash(resolved file basenames+sizes), dataFiles, libVersions, trainFrom/To, testFrom/To}` into every run's `provenance` (best-effort, never breaks a run). Test: configHash stable across identical cfgs + changes on any lever + span recorded. |
| **L6** | ✅ **FIXED** | ~~Same-bar fill~~. `EnvConfig.fill_mode` lever: `"close"` (DEFAULT — historical same-bar, corpus untouched) or `"next_open"` (decide at close, EXECUTE at next bar's OPEN via provider `get_open()`). Owner chose: next-open as a lever, use it for Stage 0. Tests: env selects close vs next-open + terminal fallback; Single/Multi `get_open` align with the raw open of the same bar. |
| **L7** | ✅ **FIXED** | ~~No purge/embargo at the boundary~~. Clean-by-construction: train/test are SEPARATE, time-disjoint providers (disjointness pinned `test_walk_forward.py`), and the supervised label loop `range(0, timesteps-horizon+1)` keeps `step+horizon` INSIDE the train provider. Added a **mutation-proven** guard (`test_forward_horizon_label_stays_inside_train_provider`) pinning the tail-purge. Full purged+embargo **CV** is a prerequisite for future *in-sample CV* (not built) — deferred, not needed by the single split. |
| **L8** | ✅ **FIXED** | ~~Constant-0 `z_score` / blanket `fillna(0)`~~. Root cause was TWO defects, both fixed WITHOUT changing the obs shape (no column added/removed): (a) **dtype-fragile bar-span inference** — `pd.to_datetime` on epoch-**ms** integers read them as ns, collapsing `minutes_per_bar` to 1 so the 1d/1m/1y horizons never scaled and blanked every long window; now `_infer_minutes_per_bar` accepts datetime64 **or** epoch-ms. (b) **horizon longer than history** → all-NaN → constant fill; now `_rolling` carries `min_periods=min(window, MIN_ROLLING_OBS=20)` so the feature degrades to a real *trailing* (still backward-only) estimate. Blanket `fillna(0)` refactored to fill each family at its OWN neutral — ratios at **parity 1.0** (a 0 asserted "price is 0× its average", an extreme false reading at every warm-up bar); the trailing `fillna(0)` stays as the safety net for mean-centred families, and the `[-1,1]` clamp is untouched. Proof on REAL data (GOLD 2022 daily, 251 rows): `price_z_score_1y` / `price_to_avg_1y` / `price_to_max_1y` went constant-0 → 233/233/221 distinct values. Named indicators (Pi_Cycle 111/350/471) deliberately keep their published bar windows and stay flagged `dead_features` when a window can't support them — degrading them would fake the indicator. Tests: bar-span inference (datetime + ms + single-bar), horizon degradation, ratio warm-up parity, and a **causality guard** (corrupting the future leaves every earlier row byte-identical). `pipelineVersion` **6.0 → 7.0** (feature semantics changed ⇒ prior runs are incomparable by the engine's major-version rule). |

### Standing discipline (law, not backlog)

1. **Future-corruption invariance is the gold-standard detector** — a provider-level property test across every
   layer-set / fidelity / lookback / context panel: pick random decision steps `t`, overwrite EVERY raw bar
   strictly after `t`'s anchor with noise (and separately reverse/shuffle the future), rebuild, assert
   `get_values(t)` is **byte-identical**. Any change proves leakage. This is the one test every new lever must pass
   (it catches leaks a bug-specific regression test misses).
2. **Mutation-test every guard** (no vacuous passes) — each guard ships with a paired mutation that flips it off and
   MUST turn its test red (`top=index-1→index`; `embargo=0`; `n_trials=1`), per the existing
   `test_fidelity_lookahead_coarse_only.py` pattern.
3. **Purged + embargoed CV** (López de Prado ch.7) for any in-sample fold; **Combinatorial Purged CV** (ch.12) for
   architecture/hyperparameter selection — a distribution of OOS Sharpe, not one lucky path, decides.
4. **Deflated-Sharpe selection gate** — no champion without DSR ≥ 0.95 at `n_trials` = the real config count.
5. **Train/validation/test hygiene + a lockbox** — select on validation only; the headline is scored exactly once,
   at sign-off, on a NEVER-touched window AND a held-out asset, provenance-stamped.
6. **Point-in-time everything** — macro/context fused by release instant (keep `pit_fusion.asof_join`);
   survivorship-frozen universe (no asset whose first bar postdates the cutoff); no back-adjusted continuous futures
   (as-of roll only) — each with its own invariance test.
7. **Provenance on every run** (L5) so any result is reproducible and auditable.

### The gate (hard, CI-blocking)

- **The leakage suite is CI-blocking. No stage advances, and no result enters the record, while it is red.**
- **L1 is fixed + pinned BEFORE any multi-layer run** (Stages 1–2). Single-layer `1d` Stage-0 baselines are
  unaffected by L1 but still bound by L2–L6.
- **Every new feature/lever (Stage 1) is TDD red-green with its future-corruption invariance test written first.**
- **Champion sign-off (Stage 3)** requires the full discipline green: purge+embargo, DSR ≥ 0.95, lockbox
  window + asset, provenance — the composite "declare champion" verdict (A4.3) ANDs them.
- **Ownership (A2–A4 shipped; only A5 app-move pending).** The engine (modeltrainer **A4.3**) now provides the
  deflated-Sharpe verdict, multi-window AND-of-medians gate, beta gate, and composite champion verdict — so **L2/L3
  and the champion stop-condition are ENGINE-gated**. BlackSwan's Stage-0.0 job is therefore (a) fix the
  executor-side vectors **L1, L4–L8**, and (b) **EMIT the inputs** the engine gates on (`oos_sharpe`, `oos_n_obs`,
  per-window returns, beta / up-vs-down-capture). A5 (BlackSwan-as-its-own-app) does not block this campaign.

## Operating model (repo split — where work lands)

| Concern | Repo |
| --- | --- |
| One `cfg → RunSummary` executor; the manifest, features, scorecard metrics (DSR/beta emission) | **BlackSwan** (this repo) |
| Planner / autopilot / cross-test / verdict / viewer; the automation loop (A2/A3/A4) + honesty guardrails | **thefactory-modeltrainer** |
| Generic infra: activity engine, cross-project execution, agent/panel spawning, scheduler | **thefactory-tools / thefactory-backend** |

**Track separability (A2–A5 owned by a separate process).** This campaign (Stages 0–3) is a DIFFERENT TRACK from
the modeltrainer platform work (A2/A3/A4/A5) and does **not block** on it — decoupled by the CLI contract: every
stage runs on the shipped executor (`python -m trainer.run …`) driven by the shipped manual-approve loop, and by
the repo-split (all BlackSwan-domain items live here).
- **Non-blocking accelerators.** A2/A3/A4.1/A4.2 only cut human clicks / enable overnight self-driving; the
  `explore` autopilot only automates a matrix I can script by hand; A5 (template/library packaging) is irrelevant
  to edge-search. Track B (A2–A5) **accelerates + hardens + packages** Track A; it never gates it.
- **One soft coupling.** Stage-3's champion DECLARATION wants A4.3's composite verdict. Mitigated in-repo:
  BlackSwan emits the DSR + beta/up-vs-down-capture metrics (a BlackSwan-domain item here), and until the engine's
  composite tool lands the AND-of-gates is computed manually from those metrics — Stage 3 **degrades, never blocks**.
- **Self-owned here regardless of who owns A2–A5:** the `asset` lever, the `calendar_features`/`regime` levers,
  metric emission, the `windows_supported` truncation guard, checkpoints-on.

**Experiment governance (framework is the ENGINE's, not BlackSwan's).** RL MODEL runs go to the `blackswan-run`
store ONLY (kept apples-to-apples pure). DIAGNOSTICS that produce no model (baseline scans, breadth analyses,
ablations, correctness probes) are **side-experiments** — a GENERIC engine capability (modeltrainer **A4**:
`-experiment` recordType + campaign + multi-source hypothesis evidence), reusable by every project, NOT
re-implemented in BlackSwan. There is ONE thesis/hypothesis concept, fed by evidence from BOTH stores (a thesis
may require both); the hypothesis aggregates across them. BlackSwan contributes only (a) the runs via the CLI
contract and (b) an optional thin DOMAIN reducer for a side-experiment's trading-specific `aggregate` (e.g.
equity-curve portfolio/breadth math) — added when the engine framework lands, never as standalone framework here.

---

## Stage -1. Operating model — semi-automate the loop (the answer to "how does an AI drive this?")

**The loop.** The AI ORIENTS (auto, no approval: `getTrainerState` → `diagnoseSearch` → `getRunData`/`getRunXAI` →
`queryProjectData` over run records/scorecards) → PROPOSES experiments (`recommendTrainingExperiments`,
approval-gated, spec-validated by `expandExperimentMatrix`, launches `train`), records its reasoning
(`createProjectRecord`/`updateProjectRecord`: hypotheses, alternate scorecards), and FILES code improvements
(Stories) → human APPROVES (the one gate) → activities / the `explore` autopilot run **detached** (local or a paired
remote runner), the queue drains → results land as **scorecard-stamped records** → the AI READS + diagnoses +
proposes the next batch. A long approval-gated conversation: **human = approve + strategic calls; AI =
research/propose/run/read/iterate; autopilot = the unattended inner search.**

**Runnable TODAY (all shipped):** the four auto read tools; `diagnoseSearch` (already a chat tool, scorecard-ranked
— the modeltrainer plan's A3.1 "promote it" line is stale); `recommendTrainingExperiments` (validates + launches);
the approval-gated `startProjectActivity`/`createProjectRecord`/`updateProjectRecord`; the `explore` autopilot
(one approval → many `train` children, staged calibrate→screen→global→local→converge); paired remote runners; the
records substrate + WS finished badge. This already gives the propose→approve→run→read→propose loop **inside the
BlackSwan (or trainer) project chat**.

**What keeps it SEMI- not FULLY-automated → the modeltrainer A4 builds** (cross-reference, don't duplicate):
A4.1 cross-project launch/monitor + event→turn re-invocation + overnight scheduler; A4.2 self-improvement
EXECUTION path (`spawnPanel` so the AI runs the Stories it files); A4.3 honesty guardrails + the composite
"declare champion" stop-condition. Until A4.1 lands, a central/overseer agent can READ BlackSwan cross-project but
must drive campaigns from inside the BlackSwan chat; until A4.3 lands, "are we done?" is assembled ad hoc.

**This stage's build:** nothing new in BlackSwan — it already conforms. The autonomy upgrades are the modeltrainer
A4 items, sequenced there. Here we ADOPT the loop and file the A4 work as the side-goal's first tickets.

---

## Stage 0. Rigorous asset-edge scan — deterministic baselines (evaluation rigor first)

Cheapest possible signal, no training, on data already on disk. Establishes WHICH assets have prima-facie
structure before any RL compute.

1. **Pre-register the gate** (write it as a scorecard/hypothesis record BEFORE running): an asset has capturable
   structure only if a deterministic baseline **beats buy-and-hold net of realistic per-asset fee**
   (`return_vs_hold_pct > 0`) in **≥3 walk-forward windows**, with **deflated-Sharpe positive** and Calmar/Sortino
   tail-checked. Benchmark = buy-and-hold, never cash.
2. **Universe (timeframe=1d — non-crypto `1h`/`1m` hard-fails, no intraday on disk):** GOLD, SPY, UUP
   (runnable today only via direct `--config-json` — not yet in `levers.asset`), + on-disk crypto (BTC + 8 alts) +
   10 stocks. **Per-asset fee override** (`transaction_fee` ~0.0002 equities/gold, ~0.001 crypto). **Pick windows
   via `walk_forward.windows_supported`** to avoid silent train-window truncation (`config_builder.py:45,52,60`):
   `stk-*` for 2018+ GOLD/SPY, `alt-*` for 2022+ alts.
3. **Baselines** (deterministic, no training): `hodl` / `momentum` / `ma_crossover` / `breakout` / `technical`(RSI)
   / `weekday` / `time`. Validated command shape:
   ```
   cd /Users/cloud/Documents/Work/BlackSwan && printf '%s' \
   '{"model_name":"momentum","asset":"GOLD","timeframe":"1d","fidelity_set":"1d","walk_forward_window":"2024","transaction_fee":0.0002,"momentum_lookback":30,"seed":0}' \
   > /tmp/c.json && .venv/bin/python -m trainer.run --config-json /tmp/c.json --summary-out /tmp/c.summary.json
   ```
4. **Read + rank:** per summary — `total_return_pct`, `return_vs_hold_pct` (the key field), `oos_sharpe`,
   `max_drawdown_pct`, `signal_expectancy`. Rank assets by the fraction of (window × model) cells clearing the gate.
5. **Output:** the shortlist of assets with structure. **Side-goal tickets filed:** add GOLD/SPY/UUP to
   `levers.asset` (so the autopilot fans out); a `windows_supported` truncation guard.

## Stage 1. Feature reformulation on survivors — the RIGHT few features (problem formulation)

Owner-validated: MORE indicators did NOT help (`process_df_simple` is the deliberate result). So this is **feature
SELECTION**, not accumulation — the cheap, structurally-different channels the models have NEVER seen, added as
**toggleable levers, tested one-at-a-time** to isolate attribution:

1. **`calendar_features` lever** — day-of-week / day-of-month / month + an **FOMC-window** flag, derived from the
   bar timestamp (extension point `config_builder.py:158` daily-timestamp block + `src/data/abstract_dataprovider.py:453-467`;
   FOMC needs only a static meeting-date table). These are exactly the seasonality/regime edges Stage 0 targets and
   are invisible on the daily path today.
2. **`regime` channel lever** — a standalone realized-vol / trend-slope channel decoupled from the
   `with_indicators` bundle (today `volRegime10`/`trendSlope10` are gated on the whole bundle,
   `abstract_dataprovider.py:157-158`), so the search can isolate the volatility-regime lens.
3. **Fix broken features at the root:** `z_score_1m/1y` = constant 0 (QW1); wire **taker order-flow** where it is
   real (crypto only — neutral-fill on gold/stocks, so never read a flat volume feature as "no edge").
4. **TDD** (mandatory): mirror `src/data/test_abstract_dataprovider.py` + `trainer/test_config_builder.py` — failing
   test first for each new lever/column.
5. **Test "features > asset swap":** ~~re-run Stage-0 baselines with each channel~~ — **corrected protocol**: the
   Stage-0 baselines (`momentum`/`ma_crossover`/`breakout`) read `env.get_price()` and NEVER read the observation
   (`obs` is an unused parameter), so they cannot test a feature channel at all. A channel can only be measured by
   an **obs-consuming** model; `supervised-gbm` is the cheap probe (~6 s/cell on daily), RL is the expensive one.
   Stage 0 also produced no survivors, so the probe runs on its top-ranked universe (GOLD, SPY).

### Stage 1 PRE-REGISTERED gate (written BEFORE any Stage-1 run — do not edit after results exist)

- **Probe:** `supervised-gbm`, daily, `fill_mode=next_open`, `transaction_fee=0.0002`, pipelineVersion 7.0.
- **Baseline projection is `standard` (lean), NOT `with_indicators`** — the curated bundle ALREADY emits
  `volRegime10`/`trendSlope10` (`_add_curated_indicators`), so testing a `regime` channel on top of it would be a
  guaranteed no-op. Starting lean and adding ONE channel at a time is what makes the attribution real (and matches
  "feature SELECTION, not accumulation").
- **Corpus:** arms {`baseline`(standard), `+calendar`, `+regime`, `+both`} × assets {GOLD, SPY} × windows
  {stk-2022, stk-2023, stk-2024, stk-oos-2024} × seeds {0, 1, 2} = **96 gated cells**, plus a non-gated
  **reference arm** `+indicators` (`projection=with_indicators`, 24 cells) to re-anchor the owner-validated
  "more indicators did not help" finding under the post-L8 feature layer. 120 cells total.
- **Primary metric:** `return_vs_hold_pct` per cell (the Stage-0 key field).
- **A channel PASSES iff**, paired against the identical baseline cell (same asset × window × seed), ALL hold:
  1. the mean paired improvement in `return_vs_hold_pct` is **> 0**, AND
  2. it **increases** the number of cells clearing `return_vs_hold_pct > 0`, AND
  3. condition 1 holds in **≥3 of the 4 windows** (the Stage-0 ≥3-window discipline).
- **Decision rule:** escalate to RL only if ≥1 channel passes. If none passes, Stage 1 is a **NULL** and we take the
  plan's honest fork (B1 cross-sectional / B2 signal model / broaden data) with four nulls behind us.
- **Discard rule:** any run whose leakage suite is not green, or that lacks a provenance stamp, is discarded rather
  than interpreted. This outranks the gate.

### Stage 1 RESULT: NULL — neither channel passed (Aug 2026)

Both levers BUILT (TDD, default-OFF so an existing observation is byte-identical): `calendar_features`
(day-of-week / day-of-month / month / turn-of-month, derived purely from the bar's own close timestamp) and
`regime` (realized-vol + deviation-from-trend WITHOUT the indicator bundle). 120 cells, 0 failures, ~3 min, run
through the engine's side-experiment framework; arms persisted as `blackswan-run-experiment` records
(`baseline 6df95f543c3a`, `calendar 333657f1b52a`, `regime fb93bb311e7f`, `both 381f045da9c3`,
`indicators 6a44b957966a`) with the two gated channels registered as hypotheses (`10b6380703f9`, `dec5b562bf41`).
Baseline: 6/24 cells beat buy-and-hold.

| arm | mean paired Δ vs-hold | cells clearing (vs 6) | windows Δ>0 | gate |
| --- | --- | --- | --- | --- |
| `calendar` | −3.76 | 0 | 1/4 | **FAIL** (all three conditions) |
| `regime` | **+1.97** | 6 (no increase) | 3/4 | **FAIL** (condition 2) |
| `both` (ref) | −1.28 | 0 | 1/4 | — |
| `indicators` (ref) | −0.43 | 0 | 2/4 | — |

**Decision (per the pre-registration, goalposts unmoved): Stage 1 is a NULL → take the honest fork.**

Notes worth carrying forward:
- **`regime` is a genuine near-miss, not a pass.** It improved the mean (+1.97) in 3 of 4 windows, but converted
  **zero additional cells** into buy-and-hold beaters — the mean is carried by one window (stk-oos-2024, +11.45).
  That is precisely the "moves the average without broadening the win" pattern condition 2 exists to reject.
- **The `indicators` reference arm re-confirms the owner-validated "more indicators did NOT help"** under the
  post-L8 feature layer: 0/24 clearing vs the lean baseline's 6/24. The lean projection is not the bottleneck.
- **Deviation from the plan's wording:** the FOMC-window flag was NOT built. No meeting-date table exists in the
  repo, hand-writing ~72 historical dates from memory would inject unverified data into a measured experiment, and
  deriving "a policy change happens tomorrow" from the on-disk `DFEDTARU` series would be **lookahead**. Shipped
  turn-of-month instead (a documented calendar anomaly that is exactly derivable and causal). A real FOMC channel
  needs a sourced meeting-date table first.
- **Two silent-void bugs were found and fixed by the empirical check** (all four arms initially returned identical
  numbers): the feature-cache key omitted the new levers (every arm reused the baseline frame), and the channels
  were wired into `process_df_simple` only while a default daily run goes through `SingleDataProvider`/`process_df`.
  Both are now pinned by tests — including a cache-key guard, since a lever missing from the key voids any future
  feature experiment the same way.

## Stage 2. RL + diversified-trend on survivors (problem formulation, not more algos)

Only on assets/features that passed. RL/autopilot (`reppo-custom` / PPO / GRU / S4D), `save_checkpoint=true` on
keepers (default `SAVE_CHECKPOINTS=False`, `config_builder.py:237`).

1. **Separate entry-signal from position-sizing** — `position_sizing=vol_target` exists; test a signal head vs the
   position manager (the Case-1/Case-2 split; ties to modeltrainer B2).
2. **Meta-labeling / triple-barrier** (López de Prado) as a supervised overlay on the baseline signal — a genuinely
   under-explored formulation, not another algo.
3. **Regime/state conditioning** using the Stage-1 regime channel.
4. **Diversified-trend recovery** — ONE trend policy across {gold, rates (once acquired), SPY, energy}, aggregate
   equity curves via the cross-asset **checkpoint-replay** harness (`--evaluate` + swapped asset/window). This is
   the only path to trend's diversification-sourced Sharpe without the multi-asset env (modeltrainer B1). Trend's
   ~1.0 Sharpe is a breadth result; a single market yields ~0.3–0.5, lumpy.
5. **Read** via `diagnoseSearch`: robust-split verdict + reward-vs-scorecard alignment (BlackSwan's is near-zero →
   decide on the scorecard, never the raw reward).

### Long/SHORT screen — RESULT: thesis REFUTED (do not spend RL compute here)

Stage 2 named long-only as "the biggest structural gap" and shorting as the highest-EV next lever. **Screened and
killed** (Aug 2026) — 144 deterministic cells (2 arms × 6 assets {GOLD,SPY,UUP,TLT,IEF,SHY} × 4 windows
{stk-2022/23/24, stk-oos-2024} × 3 published rules {momentum, ma_crossover, breakout}), daily, next-open fill,
2 bps, **pipelineVersion 7.0** (post-L8), 0 failures, ~4 min wall clock. Run through the ENGINE's side-experiment
framework (`runSideExperimentCampaign`), persisted as two `blackswan-run-experiment` records
(`b5379a2d63a9` long/SHORT, `29b290d04668` long-only) linked to hypothesis `346822171328` — **no RL run records
created**, so the run store stays apples-to-apples pure.

| window | L/S beats hold | long-only beats hold | mean vs-hold L/S | mean vs-hold LO | L/S beats LO |
| --- | --- | --- | --- | --- | --- |
| stk-2022 (bear) | 11/18 | **17/18** | **+12.56** | +9.16 | 10/18 |
| stk-2023 (bull) | 0/18 | 5/18 | −10.53 | **−4.54** | 3/18 |
| stk-2024 (bull) | 0/18 | 6/18 | −4.79 | **−5.59** | 9/18 |
| stk-oos-2024 | 1/18 | 1/18 | −12.10 | **−12.81** | 8/18 |
| **all** | **12/72** | **29/72** | −3.72 | −3.44 | **30/72** |

**Verdict `disproved`** (auto, from experiment evidence alone — the hypothesis flipped `untested → disproved` with
`transition.sources: ['experiment']` and ZERO RL runs, the A4 multi-source path proven live). Reading: shorting
does **amplify** the bear window (mean vs-hold +12.56 vs +9.16) but with far higher variance (it beats hold in
FEWER cells there, 11/18 vs 17/18), and it is destroyed in the bull windows (0/18 and 0/18). Net it is a coin-flip
against its own baseline (30/72) and *worse* on both mean return (6.10 vs 6.37) and mean vs-hold (−3.72 vs −3.44).
Enabling shorts does NOT convert the defensive-in-bear/lose-in-bull profile into an edge — it doubles down on the
same beta-timing bet in both directions. Next lever must come from **breadth (more low-correlation markets) or
risk-parity weighting**, not from the action space.

**Registry hygiene finding (unfixed, needs a decision):** all 6 pre-existing hypotheses pin `use_indicators`, which
is NOT a declared lever any more — the config builder resolves it to `projection: 'with_indicators'`. Those specs
therefore match only PRE-rename runs and can never gather new evidence (the dead-pin pathology `hypothesisHygiene`
exists to surface). Re-keying them changes their ids, so it is an owner call, not a silent migration.

## Intraday OBSERVATION screen — the last untested axis

Five nulls in, all at daily frequency. Frequency is the one axis that changes the physics rather than the
costume, but it must be added the way the cost arithmetic permits, not the way it is usually imagined.

**Trading faster is excluded a priori.** B2 measured the raw per-signal edge at ~0.05% against a ~0.04%
round trip at 2 bps. Crypto costs ~10 bps, so a round trip is ~0.2% — 4× the entire measured edge. Rebalancing
hourly would pay ~4.8%/day in fees. No model recovers that; screening it would be theatre.

**So "intraday" here means OBSERVE fine, TRADE slow** — and BlackSwan already supports exactly that: `fidelity_set`
sets what the model SEES (`fidelity_input`) independently of when it DECIDES (`fidelity_run`). `1h` and `1h+1d`
both resolve to `fidelity_run=1d`. Holding the decision cadence at daily across every arm makes cost a constant
and isolates the only variable worth testing: does finer observation improve the decision?

**Venue: crypto, and only crypto.** It is the sole class with intraday on disk (1m for 9 symbols, 2022→2026;
BTC from 2017) — equities/ETFs/commodities are `intervals=('1d',)` in the catalog, so equity intraday is a
mining project with hard vendor limits, deferred until this screen justifies it. Crypto is also the only place
`asset_volume_taker_base` is real, so the taker order-flow feature carries actual information here rather than
the neutral fill it becomes on gold/stocks.

### PRE-REGISTERED gate (written before any run — do not edit afterwards)

- **Corpus:** arms {`1d` (baseline observation), `1h`, `1h+1d`} × assets {BTCUSDT, ETHUSDT, SOLUSDT} × windows
  {alt-2024, alt-2025, alt-2026, alt-oos-2024} × seeds {0,1,2} = **108 cells**. `supervised-gbm`,
  `projection=standard`, `transaction_fee=0.001` (real crypto cost), `fill_mode=next_open`,
  **`fidelity_run=1d` in every arm** so cost is held constant.
- **Primary metric:** `return_vs_hold_pct` (vs buy-and-hold that asset).
- **An arm PASSES iff**, paired against the identical `1d` cell (same asset × window × seed), ALL hold:
  1. mean paired improvement in `return_vs_hold_pct` > 0; 2. it increases the count of cells clearing
  `return_vs_hold_pct > 0`; 3. condition 1 holds in **≥3 of the 4 windows**.
- **Decision:** if an arm passes, escalate (RL at that fidelity, then consider mining equity intraday). If none
  passes, finer observation does not help either, and with six nulls the honest campaign conclusion is that this
  formulation family has no exploitable edge net of cost — report that rather than trying a seventh costume.

### RESULT: NULL — neither arm passed (Aug 2026)

**81 of 108 cells completed.** The 9 `alt-2026` cells in each arm FAILED, correctly: the L4 guard refused the
window because only 6 of its 12 test months are mined (2026 is half elapsed) and running it would have silently
truncated the span. That is the guard doing its job — but the corpus should have been checked for data
completeness at pre-registration time, not discovered at run time. **Process lesson: verify the requested
windows are fully mined BEFORE writing the corpus into a gate.** The window was NOT substituted after the fact —
swapping in a replacement once results are visible is selection-on-test.

Experiments `534dd1e8e108` (`1d` baseline), `f32dfefb7e4a` (`1h`), `9cc2b2513226` (`1h+1d`), hypothesis
`3be0a28e488b`. Baseline: 12/27 cells clearing vs-hold, mean −45.46 (10 bps costs bleed everything).

| arm | mean paired Δ | cells clearing (base 12) | windows Δ>0 | gate |
| --- | --- | --- | --- | --- |
| `1h` | **+15.73** | **9** | 2/4 (2/3 runnable) | **FAIL** (conditions 2, 3) |
| `1h+1d` | **+12.71** | **3** | 2/4 (2/3 runnable) | **FAIL** (conditions 2, 3) |

**The verdict does not hinge on the missing window:** condition 2 fails decisively (both arms clear FEWER cells
than the daily baseline), and on the three runnable windows condition 3 is 2/3 — still short. Both arms lift the
MEAN while clearing fewer cells, and the lift is carried by one window (alt-oos-2024: +45.54 / +38.61) — the
identical "moves the average without broadening the win" signature the Stage-1 `regime` channel showed. Finer
observation buys variance, not selectivity.

## Risk axis + reversal — nulls SEVEN and EIGHT (Aug 2026)

Two further axes were screened under pre-registered gates and both failed; full detail in
`docs/risk-axis-plan.md` and `docs/cross-sectional-plan.md` §1b.

- **Risk axis (vol targeting).** The first axis in the campaign aimed at a *predictable* quantity: trailing→future
  correlation is **+0.426** for volatility (12/12 symbols) against **+0.015** for return. 324/324 cells, 0 failed.
  FAILS: 56/144 cells clear `sharpe_vs_hold>0`, 0 of 4 windows. **Three independent reads agree** — the
  exploratory probe's 9/12 was a HORIZON artefact (per calendar year it is 38/78 = **49%**, a coin flip), and the
  one condition that passed does not survive an honest control: vol targeting saved +3.10pp of drawdown where
  merely holding 0.835 of the market constantly would have saved **+3.85pp** (only 22% of cells beat that
  control). **Vol targeting is strictly dominated by holding less.**
- **Cross-sectional reversal.** FAILS all four conditions. Its real value was diagnostic: *both* signs win
  stk-2022 and lose every other window, so the P&L is not coming from the ranking — against a basket benchmark a
  net-flat book measures the **absence of beta**, which means §1 was never a verdict on momentum.

### What the combination axis is worth, measured (Aug 2026)

The one axis never tested is **combination** — a portfolio of weak per-market signals, whose Sharpe comes from
decorrelation rather than from any signal being good (distinct from cross-sectional, which bets assets *against*
each other). Measured over the on-disk universe: 12 symbols carry only **3.89 effective independent bets**
(mean pairwise ρ +0.189), a Sharpe multiplier of **1.97×**. The sub-structure matters — crypto's 3 symbols are
**1.16** bets (ρ +0.791, effectively one asset), US equities+SPY **1.57**, and only macro/rates/FX/gold is
genuinely diversifying at **3.53** (ρ +0.044); cross-class correlations are ~0.

**But diversification multiplies an edge, and there is none to multiply.** Per-signal expectancy net of the
round trip it pays, by rule family over 284 cells: `supervised-gbm` **−0.044%**, `momentum` **−0.043%**,
`breakout` **−0.075%**, `ma_crossover` **+0.044%**. The one positive family is **carried entirely by 5 GOLD
cells** (+0.818% expectancy, 60.3% hit); strip GOLD and it is negative, and 17/32 of its cells are positive — a
coin flip. **Acquiring more markets (B3) would multiply zero**, so breadth is a *product* answer (hold a
diversified basket) rather than a *model* answer — and no config in the corpus has beaten a basket.

### L3 PRE-REGISTERED test of the GOLD lead (written BEFORE any DSR was computed — do not edit afterwards)

The lead: GOLD is the only asset positive under two independent lenses (`ma_crossover` signal expectancy
+0.818% at 60.3% hit over 5 cells; the only positive mean ΔSharpe in the vol-target screen, +0.005). It is also
exactly what a multiple-testing artefact looks like — the best of ~24 asset × rule combinations, drawn from a
campaign that has searched far more than that. The Deflated Sharpe Ratio exists to settle precisely this.

**Corpus / trial count.** The persisted store holds **1,109** completed configs, of which **861** carry the full
moment bundle (`oos_sharpe`, `oos_ret_skew`, `oos_ret_kurt`, `oos_n_obs`) — the 248 `blackswan-run` records
predate its emission. `trial_sr_std` = the standard deviation of `oos_sharpe` over those 861.
`n_trials` = **1,109**: every completed config on disk was a trial that was looked at, whether or not it emitted
moments. Deflating against the *whole* campaign rather than only the cells that produced the candidate is the
honest reading — the campaign was one search for one edge, and reporting its best means N is everything that
was tried. This is also the register's own wording ("`n_trials` = the real config count").

**Candidate.** The **best** `oos_sharpe` among GOLD × `ma_crossover` cells. Best-of is precisely what DSR
corrects for, so the best is the right thing to submit.

**The GOLD lead is REAL iff ALL hold:**
1. **DSR ≥ 0.95** at `n_trials = 1109`.
2. **DSR ≥ 0.95** at `n_trials = 20888`, the historical search size the plan records. More trials can only
   lower DSR, so this is the stricter half and it is the one that reflects what was actually searched.
3. **`min_track_record_length ≤ oos_n_obs`** for the candidate — the second half of L3's stated acceptance,
   i.e. the track record is long enough to establish the Sharpe it claims.
4. **Consistency:** the **median** GOLD × `ma_crossover` cell's `oos_sharpe` also exceeds the deflation level
   SR*. A lead that exists only in its luckiest window is a window, not an edge — the standing bar this
   campaign has applied to every other arm.

**Decision.** Pass → GOLD is the first real lead in the campaign; escalate under Stage 3 discipline (multi-seed,
lockbox window **and** held-out asset, provenance). Fail → the campaign closes on eight nulls plus a lead that
did not survive multiple-testing correction, which is a genuine and well-supported result rather than a
repetition.

**Stated in advance: I expect this to FAIL.** Recording the expectation now so that a failure cannot later be
reframed as having been obvious, and so that a pass would be genuinely surprising evidence rather than a
result the test was shaped to produce.

### L3 RESULT: the GOLD lead is NOT ESTABLISHED (Aug 2026) — the campaign closes

Judged against the gate above, unedited. Computed **twice**, independently — `trainer/sharpe.py` (the golden
reference) and the engine's `deflatedCorpusVerdict` — because one number decided this. They agree to 6
significant figures (DSR 0.0017632 vs 0.0017631; the residual is the TS `normalPpf` rational approximation).

Corpus: 1,109 completed configs, **861** with the full moment bundle, `trial_sr_std` = **0.065857**.
Candidate: the best of **8** GOLD × `ma_crossover` cells — `oos_sharpe` **0.090491**, n_obs 625,
window stk-oos-2024 (skew −1.42, kurtosis 16.4).

| condition (pre-registered) | result | |
| --- | --- | --- |
| (1) DSR ≥ 0.95 at n_trials=1,109 | **0.0018** | FAIL |
| (2) DSR ≥ 0.95 at n_trials=20,888 | **0.0000** | FAIL |
| (3) `min_track_record_length ≤ oos_n_obs` | 384.3 vs 625 | pass |
| (4) median GOLD cell beats SR* | 0.0526 vs **0.2163** | FAIL |

**The margin is the story.** The deflation level SR* — the Sharpe a search of this size produces by luck alone
under a true edge of zero — is **0.2163**. The candidate's Sharpe is **0.0905**, *less than half* of it. The
median GOLD cell (0.0526) is a quarter of it. And minTRL measured against SR* rather than zero is **infinite**:
this configuration could never establish that it beats the multiple-testing threshold, no matter how long it
ran. GOLD was the best of ~24 asset × rule combinations drawn from a 20,888-config search, and it looks exactly
like what that search produces from noise.

**Condition 3 is vacuous and should not be read as support.** The adversarial review proved it: fed the null
winner its own proof constructs (sharpe 0.173204, skew −0.001524, kurt 2.911413, n_obs 252), the undeflated
pair emits psr 0.9968 and minTRL 92.5 ≤ 252 — **pure noise satisfies L3's second half verbatim**, because
minTRL at `sr_benchmark=0` knows nothing about multiple testing. It was pre-registered that way and so it is
reported as passing, but the gate's work is done entirely by conditions 1, 2 and 4. The register's wording
should be tightened to benchmark minTRL against SR*, which is the number that carries meaning (and which is
infinite here).

**Decision: the campaign closes.** Eight nulls, plus the single surviving lead failing multiple-testing
correction by two orders of magnitude. This is a result, not a stall: the question "is there an exploitable
directional or risk edge in this universe, net of real costs" has been answered no, and — for the first time —
answered with the correction that makes a no trustworthy.

### Remaining L3 work (the gate is now proven, but not yet honestly wired everywhere)

- **The champion path under-deflates.** `diagnosticsUtils.ts:275` passes `nTrials = setupSharpes.length` — the
  setups in the *current comparison*, not the ~1,100 persisted or ~20,888 searched. A champion would therefore
  be deflated against a handful of trials instead of the search that produced it, which is the lenient
  direction. The honest trial count must be threadable into `diagnostics.dsr`.
- **`min_track_record_length` emits `null` when not establishable.** Safe in Python (a comparison raises), and
  currently harmless because `metricOf` collapses `null` to `undefined` and nothing reads the key yet — but in
  JS `null <= 7` is **true**, so a gate written in the register's literal wording would read TRUE for every
  no-edge run. Omitting the key is safe in both languages and loses nothing (`psr` present + minTRL absent
  already encodes "measured, no edge").

### The one open lead, and the machinery that must judge it

GOLD is the only asset positive under two independent lenses (the `ma_crossover` expectancy above; and the only
positive mean ΔSharpe in the vol-target screen, +0.005). **That is also exactly what a multiple-testing artefact
looks like** — best of ~24 asset × rule combinations. **L3 is still dead code**: `deflated_sharpe_ratio`
(`trainer/sharpe.py:96`) is called from nowhere, so every result here — all eight nulls and this lead — has been
read without correction across ~20,888 configs. Wiring L3 and judging the GOLD lead through it is the highest-value
remaining work, because it is the difference between a lead and noise, and it applies retroactively to everything.

**This is the sixth null, and the pre-registered decision rule applies: report the conclusion rather than trying
a seventh costume.** Across single-asset direction, long-only breadth, long/short, features, cross-sectional and
now intraday observation, the measured finding is consistent — in this universe, at costs that are real, apparent
wins are exposure, not skill. Remaining untried axes are genuine DATA projects (equity intraday, which the
catalog declares unavailable — `intervals=('1d',)` — and needs mining under hard vendor limits; or new asset
classes), not further modelling on what is already on disk.

## Stage 3. Campaign to a champion (steady-win declaration)

Escalate survivors: multi-seed, ALL walk-forward windows, cross-asset generalization (checkpoint-replay on held-out
assets + post-cutoff), breadth scaling. **Gate with the honesty layer** (needs modeltrainer A4.3 + BlackSwan metric
emission): deflated-Sharpe verdict, multi-window AND-of-medians acceptance, beta/up-vs-down-capture (kill the
closet-long), drawdown-bound, `trades_per_day` floor → the composite **"declare champion"** verdict = the stop
condition the human approves against.

**Honest fork:** if no config clears it, the finding is "no robust single-asset directional edge in this universe"
→ pivot to cross-sectional (B1), the signal model (B2), or broaden data (rates/commodities). A null is a result.

### The fork, DECIDED on evidence already paid for (Aug 2026)

Four nulls in (Stage 0 single-asset, Stage 2 long-only breadth, long/SHORT, Stage 1 features). Rather than guess,
the 264 side-experiment cells already persisted were mined for the two lenses the summary emits — no new compute.

**B2 (position-blind signal model) — DO NOT BUILD.** The plan's own precondition is "build only if the
forward-horizon signal lens shows a generalisable per-signal edge". Measured over the 203 cells with >1% signal
coverage: mean `signal_expectancy` **+0.050%** per signal at a **52.2%** hit rate (supervised-gbm +0.069% / 52.4%,
ma_crossover +0.084% / 54.5%). A raw edge exists but it is ~0.05% against a ~0.04% round-trip cost at 2 bps — i.e.
inside the cost floor, which is the already-`proven` hypothesis `e4ed1bb153b6` ("transaction costs erase the
supervised-ML directional edge OOS"), now quantified. Not a project-worthy edge.

**B3 (broaden data) — not the bottleneck.** It adds markets to the same formulation that shows no selectivity.

**The measured root cause (this is the finding, not an opinion).** Across all 264 cells the beta lens says every
apparent win is **exposure reduction, not skill**:

| cohort | n | mean beta | up_capture | down_capture |
| --- | --- | --- | --- | --- |
| beats buy-and-hold | 53 | **0.19** | 0.20 | 0.19 |
| does not | 211 | 0.52 | 0.51 | 0.55 |

The winners win by being barely invested, and their capture is **symmetric** (0.20 up vs 0.19 down) — a genuine
directional edge would capture materially more upside than downside. Only 103/264 cells capture more up than down,
and the best asymmetry gap in the entire corpus is a trivial 0.10. There is **no directional selectivity anywhere
in this universe**; "defensive-in-bear/lose-in-bull" was beta timing all along, now measured rather than inferred.

**→ B1 (cross-sectional long/short) is the fork to take — planned in `docs/cross-sectional-plan.md`**, whose §1 is
a 1–2 day deterministic screen with its own pre-registered gate that can kill the 3–4 week build before it starts. It is the only option that changes the QUESTION from
"will this market go up?" (answered: no exploitable selectivity, 264 cells) to "will A outperform B?" — a relative
bet that is beta-neutral by construction, attacking the exact failure mode measured above. It needs no new mining
to start: `levers.asset` already carries 25 symbols (crypto + 10 stocks + GOLD/SPY/UUP/TLT/IEF/SHY). Caveat to
carry in: cross-sectional edges are ALSO cost-sensitive, so the pre-registered-gate discipline applies from day one,
and the `signal_expectancy`-vs-cost arithmetic above is the first thing to check on any candidate.

## Side-track (continuous) — improve BlackSwan + the tooling as we go

Every hole found → filed per repo-split. Seeds already identified:
- **BlackSwan (here):** GOLD/SPY/UUP in `levers.asset`; `calendar_features` + `regime` levers; fix z_score (QW1) +
  taker order-flow (QW2); `windows_supported` truncation guard; derive-and-cache 1m (today re-derives ~1.5GB/run);
  data sanitation (gap/dedup/NaN); checkpoints-on for keepers; **emit DSR + beta/up-vs-down-capture metrics** so the
  A4.3 gates have inputs.
- **modeltrainer (its plan):** A4.1 cross-project + scheduler, A4.2 self-improvement execution, A4.3 honesty
  guardrails + champion verdict; A2/A3 to raise autonomy.

## Success criteria (measurable, pre-registered)

- **Stage 0 pass:** ≥1 asset clears the pre-registered gate in ≥3 windows.
- **Champion:** composite steady-win verdict = steady on a **held-out window AND a held-out asset**, deflated-Sharpe
  above threshold, beta-gate pass, max-drawdown within envelope, `trades_per_day` floor met.
- **Every result, every stage:** the leakage suite (Correctness bar) is GREEN and the run is provenance-stamped —
  otherwise the result is discarded, not interpreted. This gate outranks every other criterion.

## Open questions / decisions (for review before executing)

- **Autonomy target now:** run the manual-approve loop (shippable today) for Stages 0–2, or build modeltrainer
  A4.1/A4.2 first so overnight campaigns self-drive? (Recommend: run Stage 0 manually now; build A4 in parallel.)
- **Data acquisition:** add rates ETFs (TLT/IEF/SHY — cheapest, highest historical trend, same yfinance path) into
  the Stage-0 universe now, or stay strictly on-disk for the first pass?
- **Stage 1 scope:** build both feature levers before RL, or run Stage 0 → RL on raw survivors first, then add
  features only if Stage 0 is thin?
- **A4 ordering:** does the modeltrainer A4 sequencing (orchestration → self-improvement → guardrails) match your
  priority, or should the honesty guardrails (A4.3) come first as the trust prerequisite?
