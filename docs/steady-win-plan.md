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
| **L8** | 🟡 **PARTIAL** | Feature-health **DONE**: `_dead_feature_flags` surfaces `dead_features:N` (constant/zero-variance obs entries — catches the known constant-0 `z_score`) on every run's `health`, +tests. **Deferred (risk-flagged):** root-fixing `z_score_1m/1y` changes the **obs shape** (breaks model/run compat); refactoring the blanket `fillna(0)` is **load-bearing** for the `[-1,1]` clamp — both need deliberate treatment, not a tail-of-turn rush. |

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
5. **Test "features > asset swap":** re-run Stage-0 baselines on survivors with each channel; keep only channels
   that widen the gate margin.

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

## Stage 3. Campaign to a champion (steady-win declaration)

Escalate survivors: multi-seed, ALL walk-forward windows, cross-asset generalization (checkpoint-replay on held-out
assets + post-cutoff), breadth scaling. **Gate with the honesty layer** (needs modeltrainer A4.3 + BlackSwan metric
emission): deflated-Sharpe verdict, multi-window AND-of-medians acceptance, beta/up-vs-down-capture (kill the
closet-long), drawdown-bound, `trades_per_day` floor → the composite **"declare champion"** verdict = the stop
condition the human approves against.

**Honest fork:** if no config clears it, the finding is "no robust single-asset directional edge in this universe"
→ pivot to cross-sectional (B1), the signal model (B2), or broaden data (rates/commodities). A null is a result.

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
