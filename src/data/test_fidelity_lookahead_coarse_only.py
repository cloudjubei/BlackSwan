"""Deep-dive look-ahead REGRESSION guard for the COARSE-ONLY fidelity family.

The audited leak (RETURN_ENGINE_AUDIT.md, +765%/+835% traded_return in a -65% bear window) lived in the
one config family where an HOURLY/MINUTE step observes ONLY coarser layers: neither the run cadence
(``fidelity_run``) nor the base (``fidelity_input``) is among the observed ``layers``. For that family NONE
of the layer-loop price/plot slicing branches in ``MultiTimelineDataProvider.__init__`` fire, so — before the
fix at multitimeline_dataprovider.py lines 131-141 — ``self.prices`` / ``self.raw_df_for_plotting`` stayed the
FULL base array while the observation ``get_values(step)`` is anchored at ``start_index + step*divider_run``.
``get_price(step)`` then read raw row ``step`` (~``start_index`` bars in the PAST relative to what the model
observed), i.e. the model saw the FUTURE price of the bar it traded — a massive look-ahead.

The FIX strides BOTH the priced series AND the plot frame to the decision cadence, and the reward/signal
series are derived from the strided plot frame AFTER it. This file pins the whole family:

  * ``1h`` step over ``1d`` (the two AUDITED runs 1h/1d supervised-logreg + supervised-gbm both resolve here)
  * ``1h`` step over ``1d+1w``  (needs a 12-month window — the 1w warmup is ~224 days)
  * ``1m`` step over ``1d``, ``1h``, ``1h+1d``  (data-gated on the 1m source)

For every case it asserts, with many parametrized steps:
  I3  len(prices) == len(raw_price[start_index::divider_run]) and STRICTLY less than the full base length.
  I5  get_price(step) is the close of raw row start_index+step*divider_run AND get_timestamp(step) reports
      that same decision bar — the bar the observation ends on.
  I4  raw_df_for_plotting and all three reward/signal series (signals_buy_sell / signals_buy_profitable /
      signals_buy_drawdown) are on the decision cadence, matching len(prices).
  + byte-identity of the O(1) fast path (get_values) vs the per-step compute path (_compute_values) across a
    step sample AND the out-of-range edge step, boundary steps (0, 1, last) finite with no IndexError, and a
    get_timesteps sanity floor.

RED on the buggy tree: with the fix block removed, len(prices) jumps to the full base length, so I3 fails
(len mismatch, NOT strictly less than full base), I4 fails (signals derive from the un-strided plot frame),
and I5 fails (get_price(0) == raw_price[0] != raw_price[start_index]).

A POSITIVE control (1h over 1h+1d — base observed, NOT the affected family) confirms the very same
assertions PASS on a correct sibling whose cadence is set by a DIFFERENT branch, proving the invariants are
real alignment checks and not tautologies.

Needs the real BTCUSDT klines under binance/; skipped wholesale when absent. 1m cases additionally gate on
_HAVE_1M so a checkout without minute data stays green.
"""

import numpy as np
import pandas as pd
import pytest

import trainer.config_builder as cb
import trainer.walk_forward as wf
from src.data.data_factory import create_provider
from src.data.test_daily_fidelity_lookahead_e2e import _HAVE_1M, _HAVE_DATA, _build

pytestmark = pytest.mark.skipif(
    not _HAVE_DATA, reason="needs binance/ BTCUSDT 1h+1d klines for the coarse-only look-ahead guard"
)


def _months(*months):
    """A resolve_walk_forward_window stand-in yielding the given 2020 train months (test month is unused —
    only the train provider is stepped). 1w-containing layers need a 12-month train span because their 1w
    warmup (multiplier 168 * lookback 32 = ~224 days over a 1h base) empties the weekly substream at the
    3-month window the shared _tiny_pairs helper uses, silently dropping the affected 1h@1d+1w combo."""

    def _pairs(cfg=None):
        return [(2020, m) for m in months], [(2020, 1)], {"walk_forward_window": "tiny"}

    return _pairs


def _build_window(timeframe, fidelity_set, months, monkeypatch):
    pairs = _months(*months)
    monkeypatch.setattr(wf, "resolve_walk_forward_window", pairs)
    monkeypatch.setattr(cb, "resolve_walk_forward_window", pairs)
    dc = cb.build_data_config({"timeframe": timeframe, "fidelity_set": fidelity_set})
    return create_provider(
        dc,
        dc.train_data_paths,
        dc.fidelity_input,
        dc.fidelity_run,
        list(dc.layers),
        dc.buyreward_maxwait,
        dc.buyreward_percent,
    )


def _assert_cadence_and_alignment(p, label, base_observed):
    """The shared invariant battery (I3 + I4 + I5 + byte-identity + boundaries + timesteps floor) for a
    resolved-from-fidelity MULTI provider. ``base_observed`` records the structural distinction the fix keys
    on: the affected family has NEITHER fidelity_input NOR fidelity_run in layers; the positive control has
    the base observed. divider_run is 1 for every case here (finer step, coarser-only layers)."""
    layers = list(p.layers)
    input_or_run_observed = (p.fidelity_input in layers) or (p.fidelity_run in layers)
    assert input_or_run_observed is base_observed, (
        f"{label}: expected base_observed={base_observed} for layers={layers} "
        f"(fidelity_input={p.fidelity_input}, fidelity_run={p.fidelity_run})"
    )

    raw_price = np.asarray(p.raw_df["price"].to_numpy(), dtype=float)
    raw_ts = pd.to_datetime(p.raw_df["timestamp"])
    si, dr = p.get_start_index(), p.divider_run
    n = p.get_timesteps()

    # get_timesteps sanity — a handful of steps is not a meaningful guard.
    assert n > 5, f"{label}: too few steps ({n}) to be a meaningful guard"

    # I3 — prices are strided to the decision cadence, NOT left at the full base length.
    strided_len = len(raw_price[si::dr])
    assert len(p.prices) == strided_len, (
        f"{label}: I3 — len(prices) {len(p.prices)} != strided {strided_len} "
        f"(full base {len(raw_price)}); un-strided prices desync from the observation → look-ahead."
    )
    assert len(p.prices) < len(raw_price), (
        f"{label}: I3 — len(prices) {len(p.prices)} is the FULL base length {len(raw_price)}; the decision "
        f"cadence stride never happened (this is exactly the pre-fix coarse-only leak)."
    )

    # I4 — plot frame AND all reward/signal series ride the same decision cadence as the priced series.
    assert len(p.raw_df_for_plotting) == len(p.prices), (
        f"{label}: I4 — raw_df_for_plotting {len(p.raw_df_for_plotting)} != prices {len(p.prices)}"
    )
    for name in ("signals_buy_sell", "signals_buy_profitable", "signals_buy_drawdown"):
        series = getattr(p, name)
        assert len(series) == len(p.prices), (
            f"{label}: I4 — {name} length {len(series)} != prices {len(p.prices)}; reward/plot cadence "
            f"diverged from price cadence (signals derive from the plot frame AFTER the stride)."
        )

    # I5 — every traded price is the close of raw row si+step*dr, and get_timestamp reports THAT bar.
    for step in range(0, n, max(1, n // 200)):
        anchor = si + step * dr
        assert p.get_price(step) == raw_price[anchor], (
            f"{label}: I5 — step {step} get_price {p.get_price(step)} != decision-bar close "
            f"{raw_price[anchor]} at raw row {anchor}; trade fills {anchor - step} bars in the past → "
            f"the model observes the future (look-ahead)."
        )
        assert p.get_timestamp(step) == raw_ts.iloc[anchor], (
            f"{label}: I5 — step {step} get_timestamp {p.get_timestamp(step)} != decision-bar open "
            f"{raw_ts.iloc[anchor]} at raw row {anchor}; timestamp desynced from the priced bar."
        )

    # Byte-identity: the O(1) fast path must equal the per-step compute path across a step sample + the
    # out-of-range edge step the env reads at `done` (must fall back to _compute_values without raising).
    for step in list(range(0, n, max(1, n // 120))) + [n]:
        fast = np.asarray(p.get_values(step))
        slow = np.asarray(p._compute_values(step))
        assert np.array_equal(fast, slow), f"{label}: step {step} fast get_values != slow _compute_values"

    # Boundaries: step 0 (front-pad / top<0 zero-grid), step 1, and the last in-range step — finite, no
    # IndexError, and get_price stays inside the strided series.
    for step in (0, 1, n - 1):
        v = np.asarray(p.get_values(step))
        assert np.all(np.isfinite(v)), f"{label}: step {step} non-finite observation"
        assert np.isfinite(p.get_price(step)), f"{label}: step {step} non-finite price"


# (timeframe, fidelity_set, months, needs_1m) — the AFFECTED coarse-only family. 1h@1d is the audited config
# (both flagged runs resolve here); 1h@1d+1w needs the 12-month window; the 1m cases are minute-base.
_AFFECTED = [
    ("1h", "1d", (1, 2, 3), False),
    ("1h", "1d+1w", tuple(range(1, 13)), False),
    ("1m", "1d", (1, 2, 3), True),
    ("1m", "1h", (1, 2, 3), True),
    ("1m", "1h+1d", (1, 2, 3), True),
]


@pytest.mark.parametrize("timeframe,fidelity_set,months,needs_1m", _AFFECTED)
def test_coarse_only_family_strides_price_and_aligns_observation(
    timeframe, fidelity_set, months, needs_1m, monkeypatch
):
    """AFFECTED family — a finer step observing ONLY coarser layers. Pre-fix, self.prices stayed the full
    base array (I3 fails), the reward/signal series rode that un-strided frame (I4 fails), and get_price(0)
    returned raw_price[0] instead of the decision-bar close raw_price[start_index] (I5 fails). This is the
    exact look-ahead the audited 1h@1d runs suffered."""
    if needs_1m and not _HAVE_1M:
        pytest.skip("needs binance/ BTCUSDT 1m klines for the minute-base coarse-only guard")
    p = _build_window(timeframe, fidelity_set, months, monkeypatch)
    _assert_cadence_and_alignment(p, f"{timeframe}@{fidelity_set}", base_observed=False)


def test_positive_control_base_observed_is_not_the_affected_family(monkeypatch):
    """POSITIVE control — 1h@1h+1d observes its 1h base, so a DIFFERENT branch (not the coarse-only fix)
    sets the decision cadence. The identical I3/I4/I5 battery PASSES here, proving the assertions are real
    alignment checks that hold for a correct sibling and are not tautologically green. Its layer membership
    (base observed) is the structural opposite of every affected case above."""
    p = _build("1h", "1h+1d", monkeypatch)
    _assert_cadence_and_alignment(p, "1h@1h+1d", base_observed=True)
