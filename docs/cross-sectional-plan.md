# B1 — Cross-sectional long/short: implementation plan

**Remaining work only.** Campaign context + the four nulls that led here: `docs/steady-win-plan.md`.
Correctness bar, leak register and the discard rule in that document GOVERN this one too.

## Why this, and why now

Four measured nulls (single-asset directional, long-only breadth, long/SHORT, features) share ONE cause, and it
is now measured rather than inferred — over the 264 persisted side-experiment cells:

| cohort | n | mean beta | up_capture | down_capture |
| --- | --- | --- | --- | --- |
| beats buy-and-hold | 53 | **0.19** | 0.20 | 0.19 |
| does not | 211 | 0.52 | 0.51 | 0.55 |

Every apparent win is **exposure reduction, not skill**: winners are barely invested and their capture is
symmetric. Only 103/264 cells capture more up than down; the best asymmetry gap in the corpus is 0.10.

So the question itself is wrong. "Will THIS market go up?" has no exploitable answer here net of cost. B1 changes
the question to **"will A outperform B?"** — a RELATIVE bet that is beta-neutral by construction, which is exactly
the failure mode above. Universe is already on disk: `levers.asset` carries 25 symbols (9 crypto + 10 US equities
+ GOLD/SPY/UUP/TLT/IEF/SHY), so **no new mining is required to start** (B3 is not a prerequisite).

## The discipline that must not slip

Two of today's decisions were made by cheap probes that killed expensive builds (B2 died to a lens over data we
already had; long/SHORT died to a 4-minute screen). **The same rule applies to B1: falsify before building.** The
3-D env is ~3–4 weeks; the screen that can kill it is ~1–2 days. Do the screen first, and never start §2 until §1
clears a gate written in advance.

---

## §1. Cheap deterministic cross-sectional screen — FALSIFY FIRST (~1–2 days)

No RL, no new env, no 3-D observation. A ranking rule over the existing daily files, executed through the trainer
CLI contract so it persists as a normal `-experiment` record and is re-runnable/inspectable like everything else.

1. **A minimal multi-asset backtest entry point** (`trainer/xsection.py` + a `trainer-xsection.json` manifest, or a
   `xsection` model type on a new `--xsection` CLI flag). Contract-conformant: reads a config JSON, writes a
   RunSummary with the SAME metric vocabulary (`total_return_pct`, `return_vs_hold_pct`, `oos_sharpe`, `beta`,
   `up_capture`/`down_capture`, `signal_*`, DSR inputs) so every existing gate, lens and scorecard applies unchanged.
   The benchmark for `return_vs_hold_pct` is the **equal-weight universe basket**, not a single asset.
2. **The rule (deliberately crude, published):** cross-sectional momentum — rank the universe by trailing
   `lookback` return, go long the top `k`, short the bottom `k`, equal-weight, rebalance every `rebalance_days`,
   charge the per-asset fee both ways. Levers: `universe`, `lookback`, `k`, `rebalance_days`, `long_only`.
3. **Correctness (cross-sectional adds NEW leak vectors — these are gating, not optional):**
   - **Survivorship.** The universe must be the symbols tradeable AT THAT BAR, never today's list. Any symbol whose
     history starts after the window opens must be excluded from ranks before it lists.
   - **Alignment.** N symbols on one clock: a misaligned join silently fabricates P&L. Unit-test against a
     3-symbol × 100-bar fixture with deliberately ragged calendars + holidays.
   - **Universe selection is not free.** Choosing the 25 symbols with hindsight is itself selection-on-test;
     pre-register the universe and never tune it on results.
   - The **future-corruption invariance test** must extend to the multi-asset provider (corrupt every symbol's bars
     after `t`, assert ranks at `t` are byte-identical).
4. **Run it through the side-experiment framework** (`runSideExperimentCampaign`) linked to a hypothesis, so the
   result joins the trail whether it passes or fails.

### §1 PRE-REGISTERED gate (write before the first run; do not edit afterwards)

- **Corpus:** `lookback` {30, 90, 252} × `k` {3, 5} × `rebalance_days` {5, 21} × 4 walk-forward windows
  {stk-2022, stk-2023, stk-2024, stk-oos-2024} = **48 gated cells** (long/short), plus a non-gated `long_only`
  reference arm on the same grid (48) = 96. Universe `macro+stocks` (16 US-session symbols), 2 bps on turnover.
  *(Amended before any gated run, criteria unchanged: seeds are NOT swept — the screen is fully deterministic,
  so three identical cells would fake replication rather than demonstrate it. A plumbing smoke-run confirmed
  determinism.)*
- **Primary metric:** `return_vs_hold_pct` against the **equal-weight universe basket**.
- **PASSES iff** all hold: (1) a majority of cells clear `return_vs_hold_pct > 0` in **≥3 of 4 windows**;
  (2) mean `oos_sharpe` > 0 across the corpus; (3) **|beta| < 0.3 AND up_capture − down_capture > 0.15** — the
  market-neutrality-plus-selectivity condition every single-asset config failed. Note dollar-neutral is NOT
  beta-neutral: an equal-weight long/short book can still carry large net beta if the two legs differ in beta,
  which is precisely what this condition tests; (4) the edge exceeds the cost it pays —
  mean `return_vs_hold_pct` > mean `realized_cost_bps`/100 (the turnover-cost arithmetic that killed B2,
  expressed in the terms this screen emits).
- **Decision:** proceed to §2 only if it passes. If it fails, B1 is a null too and the honest conclusion is that
  this universe has no exploitable structure at daily frequency — at which point the remaining levers are
  frequency (intraday) or asset class (B3), not another model.

### §1 RESULT: FAILS the gate (Aug 2026) — §2 is NOT authorised

96 cells (48 gated long/short + 48 long-only reference), 0 failures, ~8 minutes. Persisted as
`blackswan-xsection-experiment` records (`2d05b97d40cd` long/SHORT, `0b394dbb4015` long-only) under their own
recordType, linked to hypothesis `6fff5570dc63`. Universe `macro+stocks` (16 US-session symbols), 2 bps on
turnover, out-of-sample accounting only.

| condition (pre-registered) | result | |
| --- | --- | --- |
| (1) majority clearing vs-basket in ≥3 of 4 windows | **1/4 windows** | FAIL |
| (2) mean `oos_sharpe` > 0 | 0.004 | pass (but ≈ zero) |
| (3) \|beta\| < 0.3 AND up−down capture > 0.15 | \|beta\| **0.67**, gap **−0.20** | FAIL |
| (4) edge exceeds the cost it pays | **−36.66%** vs 0.83% | FAIL |

Per window (long/short): stk-2022 **11/12 clear, mean +9.91**; stk-2023 **0/12, mean −63.50**; stk-2024 4/12,
−21.26; stk-oos-2024 2/12, −71.79.

**Reading.** The same shape as every prior null, in a new costume: it wins the bear year and is destroyed
everywhere else. Two specifics worth carrying forward:
- **Dollar-neutral is not beta-neutral.** An equal-weight long/short book still carried \|beta\| 0.67 (a single
  smoke cell showed −1.05), because the two legs have different betas. The premise that cross-sectional ranking
  is "beta-neutral by construction" is FALSE as implemented — neutrality has to be engineered (beta-matched or
  vol-weighted legs), not assumed.
- **The capture gap is NEGATIVE (−0.20)** — it captures more downside than upside, the opposite of selectivity.
- The stk-2023 magnitude (−63.50%) is not a defect: 2023 was a documented momentum-crash year (the beaten-down
  names the rule shorts were exactly the ones that ripped). The internal consistency — works in the bear, breaks
  in the reversal — is itself evidence the mechanics are right; the 11 correctness properties in
  `trainer/test_xsection.py` (alignment, survivorship, causality, fee-on-turnover) are green.
- The **long-only reference** did better (28/48 clearing, mean +11.48) but at \|beta\| **1.37** — that is not an
  edge, it is the market. It re-confirms the beta finding rather than contradicting it.

**Decision: §2 is not authorised.** Five nulls now. The remaining honest levers are FREQUENCY (intraday, a data
question) or ASSET CLASS (B3), or accepting that this universe at daily frequency has no exploitable structure
net of cost. Anything further should be gated the same way: cheap screen first, criteria written in advance.

---

## §2. The multi-asset environment — only if §1 clears (~3–4 weeks) — NOT AUTHORISED (see §1 result)

Reuses BlackSwan's reward components, feature engineering, SB3 algos and walk-forward harness. Four genuinely new
pieces, each TDD:

1. **N-symbol timestamp-aligning data provider.** One clock, N symbols, explicit holiday/missing-bar policy.
   Emits a **3-D observation** (`asset × lookback × features`). The single-asset providers stay untouched.
2. **Portfolio action space.** Per-asset weight (or long/flat/short + sizing). The env must enforce gross/net
   exposure caps and charge turnover-based fees — turnover is the dominant cost term in a rebalanced book.
3. **Portfolio scorecard.** Reward = portfolio return net of fees, with a **correlation penalty as a CONSTRAINT**,
   not blended into the reward. Fitness = Sharpe/Calmar; gates = the market-neutrality condition from §1 plus a
   drawdown envelope. Reward stays a training proxy; the scorecard defines "good" (as everywhere else here).
4. **Its own manifest** (`trainer-xsection.json`) + record type, so cross-sectional runs never mix with the
   single-asset run store (same governance as side-experiments).

## §3. RL on the cross-sectional env — only after §2's deterministic rule is reproduced

Escalate exactly as the main plan's Stage 3: multi-seed, all windows, deflated-Sharpe verdict, held-out window AND
held-out symbols, beta gate, `trades_per_day` floor. A champion must beat the §1 deterministic rule, not just the
basket — otherwise the RL machinery is decoration on a rule that already worked.

## Open questions (decide before §2, not before §1)

- **Where does it live?** A new env inside BlackSwan (max reuse, one repo, risks bloating the single-asset line) vs
  a sibling repo consuming BlackSwan as a library. §1 needs neither decision — it is one CLI + a manifest.
- **Universe composition.** Mixing crypto (24/7) with equities (session-bound, holidays) on one clock is the
  hardest alignment case. Options: run separate universes per class first, or take the intersection calendar.
  Pre-register whichever is chosen.
- **Frequency.** Daily may be too slow for cross-sectional momentum net of turnover cost; intraday needs the 1m
  path and is a data question (B3), not a modelling one.
