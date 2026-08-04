"""End-to-end look-ahead REGRESSION guard for the daily-step multi-fidelity path.

The fast unit guards in test_multitimeline_dataprovider.py (test_resolved_*_never_observes_future,
test_resolved_daily_step_observes_end_of_day_never_future) build the provider via the `_fid` helper,
which hand-wires fidelity_dfs and BYPASSES process_fidelity. The +261% look-ahead leak in
RETURN_ENGINE_AUDIT.md lived in the get_values offset/index/top arithmetic AND the process_fidelity
resampling layout — so the only test that can catch a recurrence is one that runs the WHOLE real pipeline
(build_data_config -> real ensure_derived resampling -> create_provider) and asserts, for every step and
every observed layer, that the newest observed bar's CLOSE never reaches past the decision moment.

The invariant (mirrors the unit guards, on real timestamps): at a daily step the decision bar is
`raw_df row step*divider_run + get_start_index()` (the warmup-anchored bar get_price(step) is also sliced
from; start_index is an end-of-period row, so this is the day's CLOSE). No observed bar in ANY layer may
close after that timestamp. (It MAY close before — a coarser still-forming layer observes the previous
closed bar via the top=index-1 guard; that is conservative-safe, not a leak.) A second test pins the
ALIGNMENT: the observation must end on exactly that bar, not behind it (the omits-start_index desync).

Needs the real BTCUSDT 1h+1d klines under binance/; skipped when absent so a data-less checkout stays green.
"""

import os

import numpy as np
import pandas as pd
import pytest

import trainer.config_builder as cb
import trainer.walk_forward as wf
from src.data.data_factory import create_provider

_HAVE_DATA = os.path.exists("binance/BTCUSDT-1h-2020-1.json") and os.path.exists(
    "binance/BTCUSDT-1d-2020-1.json"
)
_HAVE_1M = os.path.exists("binance/BTCUSDT-1m-2020-1.json")
pytestmark = pytest.mark.skipif(
    not _HAVE_DATA, reason="needs binance/ BTCUSDT 1h+1d klines for the real-resampling e2e guard"
)


def _one_month_pairs(cfg=None):
    # A single month of 1m base (~44k bars) is enough to exercise the resample + lookback + look-ahead
    # arithmetic without loading multi-year minute data.
    return [(2020, 1)], [(2020, 2)], {"walk_forward_window": "tiny"}


@pytest.mark.skipif(not _HAVE_1M, reason="needs binance/ BTCUSDT 1m klines for the minute-base e2e guard")
def test_minute_base_hourly_decision_decouples_cadence_byte_identical(monkeypatch):
    # B0 + B2 end-to-end on the REAL 1m source: observing 1m micro-structure while DECIDING hourly loads
    # the 1m base and advances 60 bars per decision (divider_run=60), so the decision count stays ~hourly
    # (NOT 60x) — the minute-data unlock. The O(1) fast path must remain byte-identical to the per-step
    # compute path (no look-ahead/observation drift), and the standard look-ahead invariant must hold.
    monkeypatch.setattr(wf, "resolve_walk_forward_window", _one_month_pairs)
    monkeypatch.setattr(cb, "resolve_walk_forward_window", _one_month_pairs)
    dc = cb.build_data_config({"timeframe": "1h", "fidelity_set": "1m+1h", "use_indicators": True})
    assert dc.fidelity_input == "1m" and dc.fidelity_run == "1h"
    p = create_provider(
        dc, dc.train_data_paths, dc.fidelity_input, dc.fidelity_run, list(dc.layers),
        dc.buyreward_maxwait, dc.buyreward_percent,
    )
    assert p.divider_run == 60  # 60 one-minute base bars advance per hourly decision
    n = p.get_timesteps()
    assert 600 < n < 800, f"a month of hourly decisions over a 1m base should be ~720 steps, got {n}"
    # byte-identity fast (precomputed) vs slow (per-step) across a sample of steps + the edge step
    for step in list(range(0, n, max(1, n // 120))) + [n]:
        fast = np.asarray(p.get_values(step))
        slow = np.asarray(p._compute_values(step))
        assert np.array_equal(fast, slow), f"1m base step {step}: fast != slow"
    # look-ahead: no observed bar closes after the decision bar (re-derives get_values' own arithmetic)
    raw = p.raw_df
    raw_ts = _ts_col(raw)
    checks = 0
    for step in range(0, n, max(1, n // 200)):
        offset = step * p.divider_run + p.get_start_index()
        decision = pd.to_datetime(raw[raw_ts].iloc[offset])
        for i, layer in enumerate(p.layers):
            observed = _observed_close(p, raw, raw_ts, offset, i, layer)
            if observed is None:
                continue
            checks += 1
            assert observed <= decision, (
                f"LOOK-AHEAD LEAK [1m base] step {step} layer '{layer}': {observed} > {decision}"
            )
    assert checks > 0


def _tiny_pairs(cfg=None):
    # Three months is enough for daily/weekly resampling + the lookback warmup, and keeps the derived
    # cache build fast. Test window is irrelevant here (we only step the train provider).
    return [(2020, 1), (2020, 2), (2020, 3)], [(2020, 4)], {"walk_forward_window": "tiny"}


def _ts_col(df):
    for c in ("timestamp_close", "timestamp"):
        if c in df.columns:
            return c
    raise AssertionError(f"no timestamp column in {list(df.columns)}")


def _build(timeframe, fidelity_set, monkeypatch):
    monkeypatch.setattr(wf, "resolve_walk_forward_window", _tiny_pairs)
    monkeypatch.setattr(cb, "resolve_walk_forward_window", _tiny_pairs)
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


def _observed_close(p, raw, raw_ts, offset, i, layer):
    """The CLOSE timestamp of the newest bar layer `i` observes at base offset `offset`, re-deriving
    get_values' own offset/mapping/index/top arithmetic. Returns None when no bar has closed yet
    (top < 0 -> the zero-pad sentinel, nothing observed)."""
    mapping = p.get_current_mapping(raw, offset, layer, p.fidelity_run)
    index = int((offset - mapping - p._layer_trim(i)) / p.multipliers[i])
    top = index if p.multipliers[i] <= p.divider_run else index - 1
    if top < 0:
        return None
    if layer == p.fidelity_input:
        src = raw  # the base layer reads the strided raw frame, not fidelity_raw_dfs
    else:
        # fidelity_raw_dfs holds ONLY the resampled (non-base) layers, in layer order — map i past the base.
        j = sum(1 for prev in p.layers[:i] if prev != p.fidelity_input)
        src = p.fidelity_raw_dfs[j][mapping]
    src_ts = _ts_col(src)
    return pd.to_datetime(src[src_ts].iloc[min(top, len(src) - 1)])


@pytest.mark.parametrize("fidelity_set", ["1h+1d", "1h", "1d+1w"])
def test_daily_step_no_observed_bar_closes_after_decision_real_provider(fidelity_set, monkeypatch):
    p = _build("1d", fidelity_set, monkeypatch)
    raw = p.raw_df
    raw_ts = _ts_col(raw)
    steps = p.get_timesteps()
    assert steps > 5, f"{fidelity_set}: too few steps ({steps}) to be a meaningful guard"

    checks = 0
    for step in range(steps):
        offset = step * p.divider_run + p.get_start_index()
        decision = pd.to_datetime(raw[raw_ts].iloc[offset])
        for i, layer in enumerate(p.layers):
            observed = _observed_close(p, raw, raw_ts, offset, i, layer)
            if observed is None:
                continue
            checks += 1
            assert observed <= decision, (
                f"LOOK-AHEAD LEAK [{fidelity_set}] step {step} layer '{layer}': observed close {observed} "
                f"> decision close {decision} (margin {observed - decision})"
            )
    assert checks > 0, f"{fidelity_set}: no layer/step was actually checked"


@pytest.mark.parametrize("fidelity_set", ["1h+1d", "1h"])
def test_daily_step_observation_offset_matches_price_anchor_real_provider(fidelity_set, monkeypatch):
    """The observation must end on the SAME base bar get_price(step) is sliced from
    (start_index + step*divider_run). As-of-encode the BASE layer (each row -> its own base-row index) so
    its newest observed value reveals the offset get_values actually used; assert that == the price anchor
    for every step. RED before the get_start_index() offset fix (the observation was anchored at
    divider_run-1, a full warmup ~31 days BEHIND the priced/rewarded bar — the desync)."""
    p = _build("1d", fidelity_set, monkeypatch)
    base_idx = list(p.layers).index(p.fidelity_input)
    # Single-column as-of encoding so the per-layer grids stack cleanly: base row r -> r (reveals the
    # offset); other layers -> 0 (ignored by the anchor assertion, and trivially <= offset so this also
    # re-confirms nothing is observed past the decision bar).
    for i in range(len(p.fidelity_dfs)):
        for m in range(len(p.fidelity_dfs[i])):
            ln = len(p.fidelity_dfs[i][m])
            vals = [float(r) for r in range(ln)] if i == base_idx else [0.0] * ln
            p.fidelity_dfs[i][m] = pd.DataFrame({"asof": vals})
    # Rebuild the precompute caches from the as-of-encoded frames so the O(1) get_values path reflects
    # this white-box override (it materialised numpy matrices from the ORIGINAL fidelity_dfs at build).
    p._precompute_values()
    si = p.get_start_index()
    for step in range(p.get_timesteps()):
        anchor = si + step * p.divider_run
        result = np.asarray(p.get_values(step))
        assert result[base_idx].max() == anchor, (
            f"DESYNC [{fidelity_set}] step {step}: observation ends at base row {result[base_idx].max()} "
            f"but get_price(step) is anchored at {anchor} (start_index {si} + step*{p.divider_run})"
        )
        assert result.max() <= anchor


@pytest.mark.parametrize("timeframe,fidelity_set", [("1h", "1h+1d"), ("1d", "1h+1d"), ("1d", "1d+1w")])
def test_precompute_fast_path_is_byte_identical_to_per_step_compute(timeframe, fidelity_set, monkeypatch):
    # B1: the precomputed O(1)-pandas fast path (get_values) must return EXACTLY what the original per-step
    # compute path (_compute_values) returns, for EVERY step and the out-of-range edge step the env reads
    # at `done`. This is the byte-identity gate that makes the speedup safe (no observation/look-ahead drift).
    p = _build(timeframe, fidelity_set, monkeypatch)
    assert p._step_mapping is not None, "precompute did not run"
    n = p.get_timesteps()
    assert n > 5
    for step in range(n):
        fast = np.asarray(p.get_values(step))
        slow = np.asarray(p._compute_values(step))
        assert np.array_equal(fast, slow), f"[{timeframe}/{fidelity_set}] step {step}: fast != slow"
    np.asarray(p.get_values(n))  # edge step falls back to _compute_values without raising


def test_precompute_fast_path_does_no_per_step_mapping(monkeypatch):
    # SPEEDUP PROOF (B1): the fast path must NOT call get_current_mapping (which runs pd.to_datetime) per
    # step — eliminating that per-step pandas IS the speedup. The slow path calls it once per layer per
    # step; the fast path calls it ZERO times (the mapping/top indices were precomputed once).
    p = _build("1h", "1h+1d", monkeypatch)
    sample = min(40, p.get_timesteps())
    calls = {"n": 0}
    real = p.get_current_mapping

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(p, "get_current_mapping", counting)
    for step in range(sample):
        p.get_values(step)
    fast_calls = calls["n"]
    calls["n"] = 0
    for step in range(sample):
        p._compute_values(step)
    slow_calls = calls["n"]
    assert fast_calls == 0, f"fast path made {fast_calls} per-step mapping calls (should be 0)"
    assert slow_calls >= sample, "slow path should call get_current_mapping per layer per step"


@pytest.mark.parametrize("fidelity_set", ["1h+1d", "1h", "1d+1w"])
def test_daily_step_boundary_get_values_finite_and_no_indexerror(fidelity_set, monkeypatch):
    # The classic off-by-one boundaries: step 0 (lookback front-pad / top<0 zero-grid) and the last step
    # (offset near len(raw_df), top=min(top,len-1) clamp). Both must produce finite, correctly shaped
    # observations with no IndexError.
    p = _build("1d", fidelity_set, monkeypatch)
    steps = p.get_timesteps()
    for step in (0, 1, steps - 1):
        v = np.asarray(p.get_values(step))
        assert np.all(np.isfinite(v)), f"{fidelity_set} step {step}: non-finite observation"


@pytest.mark.parametrize("fidelity_set", ["1d"])
def test_hourly_step_coarse_only_price_anchors_on_decision_bar(fidelity_set, monkeypatch):
    """An HOURLY step observing ONLY coarser layers (fidelity_set=1d) — the run-cadence layer is
    NOT among the observed layers, and neither is the base (fidelity_input=1h). None of the price/plot
    slicing branches in the provider fire for this family, so self.prices stayed the FULL base array and
    get_price(step) read raw row `step` while the observation is anchored at start_index + step*divider_run
    (~767 bars ahead). That desync leaks ~32 days of FUTURE price into every decision (the crazy-return
    look-ahead in the runs audit). Invariant: get_price(step) must be the close of the SAME decision bar
    the observation ends on, and prices must be on the decision cadence — not the full base length."""
    p = _build("1h", fidelity_set, monkeypatch)
    assert p.divider_run == 1 and p.fidelity_input == "1h" and p.fidelity_run == "1h"
    assert "1h" not in list(p.layers), f"{fidelity_set}: expected the base layer to be UNobserved"

    raw_price = np.asarray(p.raw_df["price"].to_numpy(), dtype=float)
    si, dr = p.get_start_index(), p.divider_run
    n = p.get_timesteps()
    assert n > 5, f"{fidelity_set}: too few steps ({n}) to be a meaningful guard"

    # prices must be strided to the decision cadence, not left at the full base length.
    assert len(p.prices) == len(raw_price[si::dr]), (
        f"{fidelity_set}: prices not sliced to decision cadence — len {len(p.prices)} vs "
        f"{len(raw_price[si::dr])} (full base {len(raw_price)}) — price/observation desync (look-ahead)."
    )
    # every traded price must be the close of the decision bar the observation ends on.
    for step in range(0, n, max(1, n // 200)):
        anchor = si + step * dr
        assert p.get_price(step) == raw_price[anchor], (
            f"[{fidelity_set}] step {step}: get_price {p.get_price(step)} != decision-bar close "
            f"{raw_price[anchor]} at raw row {anchor} — the observation is anchored here but the trade "
            f"fills {anchor - step} bars in the past → the model observes the future (look-ahead)."
        )


def _pairs_for(train_months):
    def _f(cfg=None):
        return [(2020, m) for m in train_months], [(2020, 12)], {"walk_forward_window": "tiny"}
    return _f


def _build_months(timeframe, fidelity_set, train_months, monkeypatch):
    fn = _pairs_for(train_months)
    monkeypatch.setattr(wf, "resolve_walk_forward_window", fn)
    monkeypatch.setattr(cb, "resolve_walk_forward_window", fn)
    dc = cb.build_data_config({"timeframe": timeframe, "fidelity_set": fidelity_set})
    return create_provider(
        dc, dc.train_data_paths, dc.fidelity_input, dc.fidelity_run, list(dc.layers),
        dc.buyreward_maxwait, dc.buyreward_percent,
    )


@pytest.mark.parametrize(
    "timeframe,fidelity_set", [("1h", "1d+1w"), ("1h", "1h+1d+1w"), ("1d", "1h+1d+1w")]
)
def test_observation_is_time_prefix_causal(timeframe, fidelity_set, monkeypatch):
    """GOLD-STANDARD, implementation-agnostic look-ahead detector (never re-derives get_values' arithmetic):
    an observation at decision step t must be a pure function of data <= its decision bar. So a provider
    built on a strict TIME-PREFIX of the data (fewer trailing months) must yield BYTE-IDENTICAL
    get_values(step) for every step whose decision bar lies inside the prefix — adding future months cannot
    change a causal observation. A MIDDLE-layer future-row leak makes the full-data provider observe bars the
    prefix lacks (it clamps to a different row), so its early-step observations differ -> RED. This is the
    guard the per-arithmetic oracle can't be (the oracle shares the bug); it stays green only if no future
    bar can enter any layer's observation. Middle-layer combos: '1d' is a middle layer over a 1h base."""
    long_p = _build_months(timeframe, fidelity_set, list(range(1, 13)), monkeypatch)
    short_p = _build_months(timeframe, fidelity_set, list(range(1, 9)), monkeypatch)
    n = short_p.get_timesteps()
    assert n > 8, f"{timeframe}@{fidelity_set}: too few steps ({n}) in the prefix provider"
    compared = 0
    for step in range(n - 3):  # leave a margin: the prefix's last bars sit at its data edge (clamped)
        a = np.asarray(long_p.get_values(step))
        b = np.asarray(short_p.get_values(step))
        assert np.array_equal(a, b), (
            f"[{timeframe}@{fidelity_set}] step {step}: the observation CHANGED when future months were "
            f"added — a future bar leaked into the observation (look-ahead). shapes {a.shape}/{b.shape}"
        )
        compared += 1
    assert compared > 5, f"{timeframe}@{fidelity_set}: only {compared} steps compared"
