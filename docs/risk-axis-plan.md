# Risk axis — trade the predictable quantity

**Remaining work only.** Campaign context + the six nulls that led here: `docs/steady-win-plan.md`.
The correctness bar, leak register and discard rule in that document GOVERN this one too.

## Why this, and why it is not a seventh costume

Six nulls — single-asset direction, long-only breadth, long/short, features, cross-sectional, intraday
observation — asked one question in six outfits: **which way will price go?** Measured over the 12 symbols on
disk, that question is asking about the one quantity this data does not carry:

| quantity | trailing 20d → next 20d correlation |
| --- | --- |
| **return** | **+0.015** — noise, sign-mixed across symbols |
| **volatility** | **+0.426** — positive on **12/12** symbols (+0.205 SOL … +0.565 IEF) |

Volatility clustering is the most robust stylised fact in the series, and **nothing in the corpus has ever
traded it**. Audited across all 716 persisted configs: every one of the 372 that carries the lever ran
`position_sizing="fixed"`. `vol_target`, `stop_loss`, `take_profit` and `trailing_take_profit` have **zero
distinct values** between them. The entire risk axis is unexplored — not tried and rejected, never switched on.

This is a different question, not a different costume: it forecasts **risk**, never direction, so the measured
"no directional selectivity anywhere in this universe" finding does not apply to it.

**Two things must be said in advance, because both are easy to spin after the fact.**

1. **This is not alpha.** An unlevered vol-targeted book holds average exposure below 1, so it will usually
   earn **less raw return** than buy-and-hold and will fail `return_vs_hold_pct > 0` **by construction**. That
   is expected and is written here before any run so it cannot later be reported as a surprise or as a failure.
2. **A Sharpe win is not money until it is levered.** Higher return per unit of risk converts to more money
   only via leverage, and the exploratory probe suggests leverage is asset-class dependent (it paid on
   equities, it hurt crypto). The levered grid is therefore a **non-gated reference arm**, never part of the
   verdict.

### The exploratory probe (NOT a result — it is the reason to run a gated screen)

Causal, unlevered, expanding-median target, fees charged on turnover, full-sample, no walk-forward, no
pre-registration: drawdown improved on **12/12** symbols (mean −48.2% → −40.3%), Sharpe on **9/12**
(mean +0.591 → +0.671). It is a hint about where to spend a screen, and it carries no evidentiary weight.
Nothing below was chosen after seeing it beyond the decision to run at all.

## Why a new entry point rather than the env

`_position_size()` (`src/environment/base_crypto_env.py:512`) is real and live in `TradeAllCryptoEnv`, but it
is applied **only at entry** (`src/environment/trade_all_crypto_env.py:23`) — a held position is never re-sized.
It cannot express a continuously vol-targeted book, so the lever as it stands does not test this thesis.
(The manifest's note that it is "not currently wired into the sim" is **stale** — it is wired; it is
`"active": false`, which is why no sweep ever reached it. Corrected in place.)

Rewiring the env into a per-bar exposure overlay is the expensive build. Per the governance that killed B2 with
a lens and B1 with a two-day screen: **falsify first**. `trainer/voltarget.py` is a deterministic,
contract-conformant screen with its own `blackswan-voltarget` recordType, so its cells never mix with the RL run
store, and it can kill the env work before it starts.

## §1 PRE-REGISTERED gate (written before the first run — do not edit afterwards)

**Corpus (gated):** assets {SPY, GOLD, AAPL, NVDA, TLT, IEF} × windows {stk-2022, stk-2023, stk-2024, stk-2025}
× `vol_window` {20, 60} × `rebalance_days` {1, 5, 21}, at `weight_cap = 1.0`, `transaction_fee = 0.0002`
= **144 gated cells**.

*Window completeness was verified BEFORE writing this corpus* — the process lesson from the intraday screen,
where `alt-2026` was named in a gate and then correctly refused by the L4 guard at run time. All four windows
are fully mined (12/12 months) for all six symbols. 2026 is **6–7 months mined** everywhere, so no 2026 window
appears here.

**Non-gated reference arms** (reported in full whatever their sign, never part of the verdict):
- **Leverage:** the same grid at `weight_cap = 1.5` (144 cells).
- **Crypto:** {BTCUSDT, ETHUSDT, SOLUSDT} × {alt-2024, alt-2025} × the same lever grid, `transaction_fee = 0.001`,
  `weight_cap = 1.0` (36 cells). Crypto is **excluded from the gate for lack of window coverage** — only two
  fully-mined fixed `alt-*` windows exist, which cannot support a ≥3-of-4 consistency condition. It is *not*
  excluded because the probe disfavoured it, and its numbers are reported alongside regardless of sign.

**Primary metric:** `sharpe_vs_hold` (strategy `oos_sharpe` − `hold_sharpe`, both measured through the *same*
`_oos_stats` code path so the comparison is symmetric).

**PASSES iff ALL hold:**
1. **It broadens the win, not just the mean** — a majority of gated cells (**> 72 of 144**) clear
   `sharpe_vs_hold > 0`. This is the condition that rejected the Stage-1 `regime` channel and both intraday
   arms; it is unchanged and it is the one that matters.
2. **It survives regime change** — mean `sharpe_vs_hold > 0` in **≥3 of the 4 windows**.
3. **It is genuinely steadier** — median `drawdown_vs_hold_pct > 0` (shallower than buy-and-hold) **and**
   positive in ≥3 of the 4 windows.
4. **It is not degenerate de-risking** — mean deployed exposure **≥ 0.5** across the gated cells. A book that
   mostly sits in cash has a meaningless Sharpe; this is the analogue of the `trades_per_day` floor.

All metrics are net of turnover fees by construction, so condition 1 already enforces "the edge exceeds the
cost it pays" — `realized_cost_bps` is reported for the arithmetic to be auditable.

**Decision:**
- **PASS →** authorise the env work: convert `_position_size()` from an entry-only sizing into a **per-bar
  exposure overlay**, then escalate to RL under the Stage-3 discipline (multi-seed, DSR ≥ 0.95, lockbox window
  **and** held-out asset, beta gate).
- **FAIL →** the risk axis is a null too. That is the seventh, it is the last axis available on data already on
  disk, and the honest next lever is a **data** project — screened the same way, criteria written in advance.

### §1 RESULT: FAILS the gate (Aug 2026) — §2 is NOT authorised

**324 of 324 cells completed, 0 failed** (144 gated + 144 levered reference + 36 crypto reference), ~6 min.
Experiments `b58ae4deeefe` (gated), `3a435c4bc948` (levered ref), `287af689d724` (crypto ref) under hypothesis
`4454deed52bc`, in their own `blackswan-voltarget-experiment` recordType.

| condition (pre-registered) | result | |
| --- | --- | --- |
| (1) majority of 144 cells clear `sharpe_vs_hold > 0` | **56/144** | FAIL |
| (2) mean `sharpe_vs_hold` > 0 in ≥3 of 4 windows | **0/4 windows** (mean −0.0031) | FAIL |
| (3) median `drawdown_vs_hold_pct` > 0 in ≥3 of 4 | median **+1.50**, 4/4 | pass |
| (4) mean exposure ≥ 0.5 | **0.835** | pass |

Per window (gated): stk-2022 16/36 clearing, median ΔDD +7.31; stk-2023 4/36, +0.48; stk-2024 18/36, +0.17;
stk-2025 18/36, +1.90.

**The verdict is not close, and three independent reads agree.**

1. **Sharpe is not improved.** 56/144 cells, 0 of 4 windows, mean −0.0031. Vol targeting scales risk and return
   down *together*.
2. **The exploratory probe's 9/12 was a HORIZON artefact.** Measured per calendar year — the unit the gate
   scores — the same rule improves Sharpe in **38 of 78 symbol-years (49%)**, an exact coin flip, while
   full-sample over ~8 years it improved 9/12 symbols. The improvement exists only across regimes, not inside
   any ~250-bar window. And the 12 full-sample "wins" are not 12 independent observations: the symbols share
   the same 2020 crash and 2022 bear, so that result is closer to **one** regime observation replicated across
   correlated assets than to twelve.
3. **The one condition that PASSED does not survive an honest control.** Sharpe is scale-invariant, so a
   *constant*-exposure book at the same mean exposure has exactly the hold's Sharpe and `1 − w` of its
   drawdown — beating full buy-and-hold on drawdown is free. Against that control: vol targeting saved
   **+3.10 pp** of drawdown where merely holding 0.835 of the market constantly would have saved **+3.85 pp**.
   Mean excess **−0.75 pp**; only **31/144 cells (22%)** beat the trivial control. Vol targeting is
   **strictly dominated by holding less**.

This re-confirms, on a new axis and with a purpose-built instrument, what the 264-cell beta lens already said:
in this universe reducing exposure does not buy risk-adjusted quality, it just scales everything down.
(Corpus-wide, corr(|beta|, `oos_sharpe`) = **+0.13** — *positive* — and the |beta|<0.3 cohort's mean Sharpe is
−0.03 against +0.03 for the rest.)

**Reference arms, reported whatever the sign.** Levered cap=1.5 is *worse* (52/144, mean −0.0047) — leverage
does not rescue it. Crypto is mildly positive (23/36 clearing, mean +0.0018, median ΔDD +1.81) but sits on only
two windows, which is exactly why it was excluded from the gate in advance; it is not evidence and is not
being spun as any.

**Decision: §2 is not authorised. This is the seventh null.**

## §2 Env work — only if §1 clears — NOT YET AUTHORISED

1. **Per-bar exposure overlay.** `_position_size()` re-sizes a *held* position, not only a new entry. The
   discrete action space (Hold/Buy/Sell) cannot express fractional rebalancing, so this is a genuine env
   change, not a flag: it needs an exposure-tracking path plus turnover accounting on every re-size.
2. **Its own invariance test first.** `_realized_vol()` includes the *current* step's price, which is causal
   under `fill_mode=next_open` but is same-bar information under `fill_mode=close`. Point the
   future-corruption invariance test at the sizing path before any result from it is trusted, and mutation-test
   the guard.
3. **Re-activate the manifest levers** (`position_sizing`, `vol_target`, `vol_target_min`) only once the
   semantics match the thesis — an active lever whose behaviour is entry-only would invite sweeps that silently
   test something else.
4. **Do NOT reuse §1's condition 3 on the env path without fixing the drawdown asymmetry first.** The
   adversarial review found that `drawdown_vs_hold_pct` in `trainer/summary.py` compares a **non-compounding
   fixed-stake** strategy curve against a **compounding** hold curve, so an identical adverse move reads as a
   much shallower drawdown on the strategy side — and the more a fixed-stake run profits, the shallower its
   measured drawdown becomes. Demonstrated: 21 identical +10% round trips then one −10% move, in a market doing
   exactly the same thing, emits strategy −3.33% vs hold −10.00%, crediting +6.67 of "steadiness" that is pure
   stake-reset accounting. **This does not affect §1**: `trainer/voltarget.py` builds both curves as 1-unit
   compounding series, which is why condition 3 is sound there. It affects the RL/env path, where the bias lives
   in the pre-existing strategy-side `max_drawdown_pct` that is already shipped across the run store and read by
   `combo_drawdown_penalty` — so it is a run-store-wide change, not a metric tweak, and it must be decided
   before any env result is gated on drawdown.
5. **Known asymmetry, deliberately left alone:** the round-trip fee enters the hold curve as a constant
   multiplicative factor and therefore cancels out of every per-step return, so `hold_sharpe` is the *gross*
   price-return Sharpe while the strategy's `oos_sharpe` carries per-trade fees. The bias runs **against** the
   strategy — it inflates the bar the strategy must clear — which is the safe direction for a gate, and the
   alternatives invent volatility a holder never experienced.
