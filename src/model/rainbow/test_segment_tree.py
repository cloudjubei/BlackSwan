"""Direct unit tests for the prioritized-replay segment trees.

We assert the segment-tree invariants (sum / min over a range, prefix-sum
``retrieve`` index lookup) against a brute-force python reference over the
underlying leaf array. These are pure data-structure tests: no torch, no env,
no file IO. A repo-root sys.path insert mirrors ``test_base_crypto_env.py`` so
the absolute ``src.`` imports resolve regardless of the invocation cwd.
"""

import operator
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.model.rainbow.segment_tree import (
    MinSegmentTree,
    SegmentTree,
    SumSegmentTree,
)


# ---------------------------------------------------------------------------
# Construction / capacity assertions
# ---------------------------------------------------------------------------

def test_init_allocates_double_capacity_and_init_value():
    t = SegmentTree(4, operator.add, 0.0)
    assert t.capacity == 4
    # internal storage is 2*capacity, all seeded with init_value.
    assert len(t.tree) == 8
    assert all(v == 0.0 for v in t.tree)


@pytest.mark.parametrize("bad", [0, 3, 5, 6, 7, 12])
def test_init_rejects_non_power_of_two_capacity(bad):
    with pytest.raises(AssertionError):
        SegmentTree(bad, operator.add, 0.0)


@pytest.mark.parametrize("good", [1, 2, 4, 8, 16, 1024])
def test_init_accepts_power_of_two_capacity(good):
    t = SegmentTree(good, operator.add, 0.0)
    assert t.capacity == good


# ---------------------------------------------------------------------------
# __setitem__ / __getitem__ round-trip
# ---------------------------------------------------------------------------

def test_setitem_getitem_roundtrip():
    t = SumSegmentTree(8)
    for i in range(8):
        t[i] = float(i + 1)
    for i in range(8):
        assert t[i] == float(i + 1)


def test_getitem_out_of_range_raises():
    t = SumSegmentTree(4)
    with pytest.raises(AssertionError):
        _ = t[4]
    with pytest.raises(AssertionError):
        _ = t[-1]


# ---------------------------------------------------------------------------
# SumSegmentTree.sum vs brute force
# ---------------------------------------------------------------------------

def _fill_sum_tree(values):
    t = SumSegmentTree(len(values))
    for i, v in enumerate(values):
        t[i] = v
    return t


def test_sum_full_range_default_args():
    vals = [1.0, 2.0, 3.0, 4.0]
    t = _fill_sum_tree(vals)
    # sum() with default args covers the whole array.
    assert t.sum() == pytest.approx(sum(vals))


def test_sum_partial_ranges_match_brute_force():
    vals = [5.0, 1.0, 7.0, 3.0, 9.0, 2.0, 4.0, 6.0]
    t = _fill_sum_tree(vals)
    # exhaustively check every contiguous [start, end] window. ``end`` in this
    # API is treated as inclusive-then-decremented, so sum(s, e) == vals[s:e].
    for start in range(len(vals)):
        for end in range(start + 1, len(vals) + 1):
            assert t.sum(start, end) == pytest.approx(sum(vals[start:end])), (start, end)


def test_sum_single_element_window():
    vals = [10.0, 20.0, 30.0, 40.0]
    t = _fill_sum_tree(vals)
    for i in range(4):
        assert t.sum(i, i + 1) == pytest.approx(vals[i])


def test_sum_reflects_updates():
    t = _fill_sum_tree([1.0, 1.0, 1.0, 1.0])
    assert t.sum() == pytest.approx(4.0)
    t[2] = 10.0
    assert t.sum() == pytest.approx(13.0)
    assert t.sum(2, 3) == pytest.approx(10.0)


def test_sum_randomized_against_brute_force():
    rng = random.Random(1234)
    cap = 16
    vals = [rng.uniform(0, 100) for _ in range(cap)]
    t = _fill_sum_tree(vals)
    for _ in range(200):
        start = rng.randrange(cap)
        end = rng.randrange(start + 1, cap + 1)
        assert t.sum(start, end) == pytest.approx(sum(vals[start:end])), (start, end)


# ---------------------------------------------------------------------------
# MinSegmentTree.min vs brute force
# ---------------------------------------------------------------------------

def _fill_min_tree(values):
    t = MinSegmentTree(len(values))
    for i, v in enumerate(values):
        t[i] = v
    return t


def test_min_full_range_default_args():
    vals = [4.0, 2.0, 7.0, 1.0]
    t = _fill_min_tree(vals)
    assert t.min() == pytest.approx(1.0)


def test_min_partial_ranges_match_brute_force():
    vals = [5.0, 1.0, 7.0, 3.0, 9.0, 2.0, 4.0, 6.0]
    t = _fill_min_tree(vals)
    for start in range(len(vals)):
        for end in range(start + 1, len(vals) + 1):
            assert t.min(start, end) == pytest.approx(min(vals[start:end])), (start, end)


def test_min_reflects_updates():
    t = _fill_min_tree([5.0, 5.0, 5.0, 5.0])
    assert t.min() == pytest.approx(5.0)
    t[1] = 0.5
    assert t.min() == pytest.approx(0.5)
    # the window not containing idx 1 is unaffected.
    assert t.min(2, 4) == pytest.approx(5.0)


def test_min_randomized_against_brute_force():
    rng = random.Random(99)
    cap = 16
    vals = [rng.uniform(-50, 50) for _ in range(cap)]
    t = _fill_min_tree(vals)
    for _ in range(200):
        start = rng.randrange(cap)
        end = rng.randrange(start + 1, cap + 1)
        assert t.min(start, end) == pytest.approx(min(vals[start:end])), (start, end)


def test_min_unwritten_leaves_are_infinity():
    # min seeds leaves with +inf; an all-default tree returns inf.
    t = MinSegmentTree(4)
    assert t.min() == float("inf")


# ---------------------------------------------------------------------------
# SumSegmentTree.retrieve (prefix-sum index lookup)
# ---------------------------------------------------------------------------

def _brute_retrieve(values, upperbound):
    """Reference: the smallest index i where prefix_sum(0..i) > upperbound."""
    prefix = 0.0
    for i, v in enumerate(values):
        prefix += v
        if prefix > upperbound:
            return i
    return len(values) - 1


def test_retrieve_uniform_weights():
    # all-equal weights => each unit interval maps to one index.
    t = _fill_sum_tree([1.0, 1.0, 1.0, 1.0])
    # upperbound in [0,1) -> idx 0, [1,2) -> idx1, etc.
    assert t.retrieve(0.0) == 0
    assert t.retrieve(0.5) == 0
    assert t.retrieve(1.0) == 1
    assert t.retrieve(1.5) == 1
    assert t.retrieve(2.0) == 2
    assert t.retrieve(3.0) == 3


def test_retrieve_skips_zero_weight_leaves():
    # zero-probability leaves must never be selected.
    vals = [0.0, 5.0, 0.0, 3.0]
    t = _fill_sum_tree(vals)
    for ub in [0.0, 1.0, 4.99]:
        assert t.retrieve(ub) == 1
    for ub in [5.0, 6.0, 7.99]:
        assert t.retrieve(ub) == 3


def test_retrieve_matches_brute_force_randomized():
    rng = random.Random(2024)
    cap = 16
    vals = [rng.uniform(0.1, 10.0) for _ in range(cap)]
    t = _fill_sum_tree(vals)
    total = sum(vals)
    for _ in range(500):
        ub = rng.uniform(0, total - 1e-6)
        assert t.retrieve(ub) == _brute_retrieve(vals, ub), ub


def test_retrieve_rejects_upperbound_above_total():
    t = _fill_sum_tree([1.0, 1.0, 1.0, 1.0])
    # total is 4; anything above 4 + 1e-5 must assert.
    with pytest.raises(AssertionError):
        t.retrieve(5.0)


def test_retrieve_rejects_negative_upperbound():
    t = _fill_sum_tree([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(AssertionError):
        t.retrieve(-0.1)


def test_retrieve_at_total_returns_last_index():
    # upperbound == total (within tolerance) walks all the way right.
    t = _fill_sum_tree([2.0, 2.0, 2.0, 2.0])
    assert t.retrieve(8.0) == 3
