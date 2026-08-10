"""Direct tests for the shared past-only extreme-signal core.

Both the funding probe and the order-flow probe turn a per-bar scalar (funding rate; taker imbalance) into an
entry side by asking "is this value extreme versus its OWN PAST?". That single rule carries one silent
lookahead — judging a value against the WHOLE-sample quantile instead of the past-only one — so it lives in
one place, guarded here once, and both probes delegate to it. The guard is the same shape as the intraday
regime gate: a value that is extreme versus the full sample but ordinary versus its own past must be left flat.
"""

import numpy as np

from trainer.signal_extremes import past_only_extreme_sides


def test_high_and_low_extremes_take_raw_sides_and_invert_flips():
    vals = [0.0] * 20 + [5.0]  # a lone high extreme after a calm run
    hi = past_only_extreme_sides(vals, invert=False, pct=0.90, min_history=5)
    lo = past_only_extreme_sides(vals, invert=True, pct=0.90, min_history=5)
    assert hi[-1] == 1.0 and lo[-1] == -1.0  # invert (contrarian) is exactly the negation
    vals2 = [0.0] * 20 + [-5.0]
    assert past_only_extreme_sides(vals2, invert=False, pct=0.90, min_history=5)[-1] == -1.0
    assert past_only_extreme_sides(vals2, invert=True, pct=0.90, min_history=5)[-1] == 1.0


def test_mid_value_is_flat():
    # A symmetric past then a probe sitting dead centre: neither tail, so no side.
    vals = [(0.02 if i % 2 else -0.02) for i in range(30)] + [0.0]
    assert past_only_extreme_sides(vals, invert=False, pct=0.90, min_history=10)[-1] is None


def test_warmup_requires_min_history():
    vals = [0.0] * 4 + [9.0]  # only 4 observations precede the extreme
    assert all(s is None for s in past_only_extreme_sides(vals, invert=False, pct=0.90, min_history=10))


def test_none_values_are_skipped_and_do_not_enter_history():
    # None gaps neither fire nor pollute the distribution; the extreme after them still judges on real past.
    vals = [0.0] * 12 + [None, None] + [5.0]
    sides = past_only_extreme_sides(vals, invert=False, pct=0.90, min_history=10)
    assert sides[12] is None and sides[13] is None
    assert sides[-1] == 1.0


def _past_only_fixture():
    # Extreme-vs-sample but ordinary-vs-own-past: 40 calm, a 5-long burst, then a held level above calm but
    # far below the burst. A held-level bar just after the burst is top-decile vs the WHOLE sample yet
    # unremarkable vs its own (burst-dominated) past.
    return [0.0001] * 40 + [0.02] * 5 + [0.002] * 45


def test_quantile_is_past_only_not_whole_sample():
    vals = _past_only_fixture()
    t = 47
    assert vals[t] >= np.quantile(vals, 0.90)       # extreme vs the FULL sample
    assert vals[t] < np.quantile(vals[:t], 0.90)    # ordinary vs its OWN past
    sides = past_only_extreme_sides(vals, invert=False, pct=0.90, min_history=10)
    assert sides[t] is None  # past-only leaves it flat; whole-sample would fire here


def test_quantile_still_fires_on_a_genuine_onset():
    vals = _past_only_fixture()
    sides = past_only_extreme_sides(vals, invert=False, pct=0.90, min_history=10)
    assert sides[42] == 1.0  # a burst bar towers over everything before it -> fires (non-vacuous)
