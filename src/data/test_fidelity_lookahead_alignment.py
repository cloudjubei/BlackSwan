"""Universal price/observation ALIGNMENT + look-ahead guard across EVERY serveable provider build.

This is the centrepiece regression guard for the fidelity look-ahead family. It parametrises over the
whole (timeframe x fidelity_set) matrix the launch form can serve (3 timeframes x 9 fidelity-set ids = 27
combos), builds each provider through the REAL pipeline (config_builder.build_data_config -> real resampling
-> create_provider), and pins that the price the model TRADES is the close of the same decision bar its
observation is anchored on. No config family may silently trade a price misaligned from what it observes.

The invariants (resolved-from-fidelity MULTI provider), for a dense sample of steps that always includes
step 0 and the last step:

  I1 (price/observation alignment): get_price(step) == raw_df['price'].iloc[start_index + step*divider_run].
     The traded price is the close of the warmup-anchored decision bar the observation ends on.
  I3 (price cadence): len(prices) == len(raw_price[start_index::divider_run]) — the price series is strided
     to the decision cadence, NOT left at the full base length.
  I5 (timestamp alignment): get_timestamp(step) reports that same decision bar (raw row si + step*dr).
  I2 (no look-ahead): no observed bar in ANY layer closes AFTER the decision bar's close timestamp.

WHY THIS IS RED IF THE FIX IS REMOVED: for the family where an hourly/minute step observes ONLY layers
strictly coarser than the base (fidelity_run not in layers AND fidelity_input not in layers — e.g. 1h@1d,
1m@1h, 1h@1d+1w), NONE of the layer-loop slicing branches fire, so without the fix block in
MultiTimelineDataProvider.__init__ self.prices stays the FULL base array. Then len(prices) jumps to the raw
base length (I3 fails) and get_price(step) returns raw base row `step` while the observation is anchored at
si + step*divider_run (I1 fails: get_price(0) == raw_price[0], not raw_price[si] ~ si bars of FUTURE price).

For SingleDataProvider combos (fidelity_set == timeframe in {1h,1d,1m}: 1h@1h, 1d@1d/1d@auto, 1m@1m) the
alignment math differs (get_price(step) == prices[step + start_index], no divider_run / raw_df), so those
combos assert their own alignment: the traded price is the close at the decision row and get_values' window
ENDS on that same row.

MIDDLE-LAYER FIX (now folded into I2): a combo whose observed stack contains a MIDDLE layer (an observed
layer that is neither the base nor the coarsest observed layer) previously leaked FUTURE bars into the
observation, because _compute_values' `index = (offset - mapping)/multiplier` omitted the per-layer warmup
trim the coarser layer forces on the middle layer's resample frame. That trim is now subtracted via
MultiTimelineDataProvider._layer_trim(), so middle layers observe the most-recent CLOSED bar and I2 asserts
on them. The ONLY combos still excluded from I2 are sub-step observed layers (finer than the run cadence):
they resample to zero substreams (multiplier_run==0) and get_values raises — a process_fidelity limitation,
not an index bug. I1/I3/I5 still guard every combo.

Data-gated: 1m-base combos skip without the BTCUSDT 1m source; 1w combos widen to a 12-month window (the
weekly warmup is multiplier(1w)*lookback ~ 224 days); the two 1m+1w combos (~527k base rows, ~10080 weekly
substreams) are opt-in behind BS_SLOW_TESTS=1 so the default suite stays fast and memory-safe.
"""

import os

import numpy as np
import pandas as pd
import pytest

import trainer.config_builder as cb
import trainer.walk_forward as wf
from trainer.fidelity import resolve_fidelity
from src.data.data_factory import create_provider
from src.data.multitimeline_dataprovider import _period_ratio
from src.data.single_dataprovider import SingleDataProvider
from src.data.test_daily_fidelity_lookahead_e2e import (
    _HAVE_1M,
    _HAVE_DATA,
    _observed_close,
    _ts_col,
)

pytestmark = pytest.mark.skipif(
    not _HAVE_DATA, reason="needs binance/ BTCUSDT 1h+1d klines for the real-pipeline alignment guard"
)

_RUN_SLOW = os.environ.get("BS_SLOW_TESTS", "") == "1"

# Every fidelity_set id the launch form serves (auto + the eight explicit layer stacks) x every timeframe.
_FIDELITY_SETS = ["auto", "1m", "1m+1h", "1m+1h+1d", "1d", "1h", "1h+1d", "1h+1d+1w", "1d+1w"]
_TIMEFRAMES = ["1h", "1d", "1m"]


def _spec(timeframe, fidelity_set):
    _id, spec = resolve_fidelity({"timeframe": timeframe, "fidelity_set": fidelity_set})
    return spec


def _is_single(spec):
    # create_provider picks SingleDataProvider iff a lone layer that IS the loaded base is stepped at the
    # base cadence (no resampling, no day-by-day stepping).
    layers = spec["layers"]
    return (
        len(layers) == 1
        and layers[0] == spec["fidelity_input"]
        and spec["fidelity_run"] == spec["fidelity_input"]
    )


def _needs_1m(spec):
    return spec["fidelity_input"] == "1m"


def _has_1w(spec):
    return "1w" in spec["layers"]


def _months(spec):
    # 3 months exercises daily/lookback warmup cheaply; a 1w layer needs ~224 days (multiplier(1w)*lookback)
    # of warmup before the first decision, so widen to a full year.
    return list(range(1, 13)) if _has_1w(spec) else list(range(1, 4))


def _is_slow(spec):
    # The two 1m-base + 1w combos load ~527k base rows and resample ~10080 weekly substreams — minutes to
    # build. Opt-in only; every other combo (including the fast 1h/1d-base 1w combos) always runs.
    return _needs_1m(spec) and _has_1w(spec)


def _observation_is_lookahead_safe(spec):
    """The MULTI observation (get_values) is look-ahead-safe iff no observed layer is a SUB-STEP layer
    (finer than the run cadence): such a layer resamples to zero substreams (multiplier_run==0) and
    get_values raises — its fix needs process_fidelity, not the index arithmetic. Every other observed layer
    — base (read from the raw frame at the exact decision offset), MIDDLE, or coarsest — is look-ahead-safe
    now that _layer_trim() subtracts each resampled layer's per-layer warmup trim from the row index
    (previously a middle layer read a FUTURE row). The base layer is skipped (raw-frame read, always safe)."""
    layers = spec["layers"]
    base = spec["fidelity_input"]
    divider_run = _period_ratio(base, spec["fidelity_run"])
    for layer in layers:
        if layer == base:
            continue
        if _period_ratio(base, layer) < divider_run:
            return False  # sub-step layer -> multiplier_run==0, get_values raises (needs process_fidelity)
    return True


# One classification per combo, resolved through the real fidelity logic so the catalog can never drift
# from what the provider actually builds.
_COMBOS = {}
for _tf in _TIMEFRAMES:
    for _fs in _FIDELITY_SETS:
        _s = _spec(_tf, _fs)
        _COMBOS[(_tf, _fs)] = {
            "spec": _s,
            "single": _is_single(_s),
            "needs_1m": _needs_1m(_s),
            "months": _months(_s),
            "slow": _is_slow(_s),
            "i2_safe": (not _is_single(_s)) and _observation_is_lookahead_safe(_s),
        }

_MULTI = [c for c, info in _COMBOS.items() if not info["single"]]
_SINGLE = [c for c, info in _COMBOS.items() if info["single"]]
_ID = lambda c: f"{c[0]}@{c[1]}"


def _pairs(months):
    train = [(2020, m) for m in months]
    test = [(2020, 12)]
    return lambda cfg=None: (train, test, {"walk_forward_window": "tiny"})


def _build(timeframe, fidelity_set, months, monkeypatch):
    window = _pairs(months)
    monkeypatch.setattr(wf, "resolve_walk_forward_window", window)
    monkeypatch.setattr(cb, "resolve_walk_forward_window", window)
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


def _gate(info):
    if info["needs_1m"] and not _HAVE_1M:
        pytest.skip("needs binance/ BTCUSDT 1m source for the minute-base build")
    if info["slow"] and not _RUN_SLOW:
        pytest.skip("slow 1m+1w combo (~527k rows); set BS_SLOW_TESTS=1 to run")


def _dense(n, k=200):
    """A bounded, deterministic sample of steps that ALWAYS includes step 0 and the last step (the two
    off-by-one boundaries: the lookback front-pad and the end-of-data clamp)."""
    step = max(1, n // k)
    return sorted(set(list(range(0, n, step)) + [0, n - 1]))


@pytest.mark.parametrize("timeframe,fidelity_set", _MULTI, ids=[_ID(c) for c in _MULTI])
def test_multi_provider_trades_the_decision_bar_it_observes(timeframe, fidelity_set, monkeypatch):
    """MULTI provider: the traded price, price cadence and reported timestamp all anchor on the decision
    bar si + step*divider_run (I1/I3/I5); and where the observation is currently look-ahead-safe, no
    observed bar closes past that decision bar (I2). RED without the fix block for the coarse-only family
    (1h@1d, 1m@1h, 1h@1d+1w, ...): self.prices stays the full base array, so len(prices) no longer matches
    the strided cadence (I3) and get_price(0) returns raw_price[0] instead of raw_price[start_index] (I1)."""
    info = _COMBOS[(timeframe, fidelity_set)]
    _gate(info)
    p = _build(timeframe, fidelity_set, info["months"], monkeypatch)
    assert not isinstance(p, SingleDataProvider), f"{_ID((timeframe, fidelity_set))} unexpectedly SINGLE"

    raw = p.raw_df
    raw_ts = _ts_col(raw)
    raw_price = np.asarray(raw["price"].to_numpy(), dtype=float)
    raw_open = pd.to_datetime(raw["timestamp"])
    si, dr = p.get_start_index(), p.divider_run
    n = p.get_timesteps()
    assert n > 3, f"{_ID((timeframe, fidelity_set))}: too few steps ({n}) to be a meaningful guard"

    # I3: prices are strided to the decision cadence, not left at the full base length.
    assert len(p.prices) == len(raw_price[si::dr]), (
        f"{_ID((timeframe, fidelity_set))}: price cadence desync — len(prices)={len(p.prices)} vs strided "
        f"{len(raw_price[si::dr])} (full base {len(raw_price)}); look-ahead risk (I3)."
    )

    steps = _dense(n)
    for step in steps:
        anchor = si + step * dr
        # I1: the traded price IS the close of the decision bar the observation ends on.
        assert p.get_price(step) == raw_price[anchor], (
            f"{_ID((timeframe, fidelity_set))} step {step}: get_price {p.get_price(step)} != decision-bar "
            f"close {raw_price[anchor]} at raw row {anchor} (si {si} + step*{dr}) — price/observation "
            f"desync, the trade fills {anchor - step} bars away from what it observes (I1)."
        )
        # I5: the reported timestamp IS that same decision bar.
        assert pd.Timestamp(p.get_timestamp(step)) == raw_open.iloc[anchor], (
            f"{_ID((timeframe, fidelity_set))} step {step}: get_timestamp {p.get_timestamp(step)} != "
            f"decision-bar open {raw_open.iloc[anchor]} at raw row {anchor} (I5)."
        )

    if not info["i2_safe"]:
        return
    # I2: no observed bar in any layer closes after the decision bar's close (reuse the e2e oracle that
    # re-derives get_values' own mapping/top arithmetic).
    checks = 0
    for step in steps:
        offset = si + step * dr
        decision = pd.to_datetime(raw[raw_ts].iloc[offset])
        for i, layer in enumerate(p.layers):
            observed = _observed_close(p, raw, raw_ts, offset, i, layer)
            if observed is None:
                continue
            checks += 1
            assert observed <= decision, (
                f"LOOK-AHEAD LEAK {_ID((timeframe, fidelity_set))} step {step} layer '{layer}': observed "
                f"close {observed} > decision close {decision} (margin {observed - decision}) (I2)."
            )
    assert checks > 0, f"{_ID((timeframe, fidelity_set))}: no layer/step was actually checked for look-ahead"


# NOTE: the former test_middle_layer_observation_lookahead_is_flagged (which PINNED the middle-layer future-row
# leak as a KNOWN defect) has been removed: _layer_trim() fixes the leak, so the clean-middle combos are now
# folded into _observation_is_lookahead_safe above and asserted by I2 in the parametrized guard.


@pytest.mark.parametrize("timeframe,fidelity_set", _SINGLE, ids=[_ID(c) for c in _SINGLE])
def test_single_provider_trades_the_decision_bar_it_observes(timeframe, fidelity_set, monkeypatch):
    """SINGLE provider (fidelity_set == timeframe in {1h,1d,1m}): get_price(step) == prices[step +
    start_index] must be the close at the decision row, and get_values' window must END on that same row —
    so the price traded is exactly the close of the bar the observation ends on. prices and the feature
    frame share one row index (asserted by the length match), so a common offset means they reference the
    same bar; a get_price / get_values offset regression breaks one side and fails here."""
    info = _COMBOS[(timeframe, fidelity_set)]
    _gate(info)
    p = _build(timeframe, fidelity_set, info["months"], monkeypatch)
    assert isinstance(p, SingleDataProvider), f"{_ID((timeframe, fidelity_set))} unexpectedly MULTI"

    si = p.get_start_index()
    n = p.get_timesteps()
    lookback = p.config.lookback_window_size
    prices = np.asarray(p.prices, dtype=float)
    assert n > 3, f"{_ID((timeframe, fidelity_set))}: too few steps ({n})"
    # prices and the feature frame are the SAME result_df rows -> equal length -> a shared row index.
    assert len(prices) == p.df.shape[0], (
        f"{_ID((timeframe, fidelity_set))}: prices ({len(prices)}) and feature frame ({p.df.shape[0]}) "
        f"differ in length — they no longer share a row index, so a common offset is not a common bar."
    )

    for step in _dense(n):
        offset = step + si
        assert p.get_price(step) == prices[offset], (
            f"{_ID((timeframe, fidelity_set))} step {step}: get_price {p.get_price(step)} != "
            f"prices[{offset}] {prices[offset]} — price offset regression."
        )
        v = np.asarray(p.get_values(step))
        last_bar = v if lookback <= 1 else v[-1]
        assert np.array_equal(last_bar, p.df.loc[offset].values), (
            f"{_ID((timeframe, fidelity_set))} step {step}: get_values window does not end on the decision "
            f"row {offset} that get_price trades — observation/price desync."
        )
