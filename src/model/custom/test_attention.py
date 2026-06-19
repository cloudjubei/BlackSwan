"""Forward-pass shape correctness for the custom attention blocks used in CustomQNetwork MLPs.

Tiny CPU tensors only. These blocks sit *inside* a flat MLP, so the realistic input is a 2-D
(batch, features) tensor. We assert each block's forward runs and preserves the feature width
where the contract implies it should. GlobalContextAttention now handles any 3-D
(batch, seq, hidden) input; the remaining 2-D-MLP-input case is a separate integration
redesign and stays xfail.
"""

import pytest
import torch as th

from src.model.custom.attention import (
    AdditiveAttention,
    GlobalContextAttention,
    MultiHeadAttention,
    ScaledDotProductAttention,
    SelfAttention,
)


# --- SelfAttention ----------------------------------------------------------
# Hardcodes `batch_size, seq_len = x.size()`, so it only accepts 2-D (batch, in_dim) and
# treats each scalar feature as a sequence position of dim 1.


@pytest.mark.parametrize("batch,in_dim", [(1, 4), (4, 8), (3, 5)])
def test_self_attention_2d_preserves_shape(batch, in_dim):
    m = SelfAttention(in_dim)
    out = m(th.randn(batch, in_dim))
    assert tuple(out.shape) == (batch, in_dim)
    assert th.isfinite(out).all()


def test_self_attention_rejects_3d_input():
    # 3-D input breaks the 2-tuple unpack of x.size() -> ValueError (documents the 2-D-only contract).
    m = SelfAttention(8)
    with pytest.raises(ValueError):
        m(th.randn(4, 3, 8))


# --- ScaledDotProductAttention ----------------------------------------------
# Uses transpose(-2, -1), so it is shape-agnostic and preserves the input shape.


@pytest.mark.parametrize("shape", [(4, 8), (4, 3, 8), (1, 6)])
def test_scaled_dot_product_preserves_shape(shape):
    m = ScaledDotProductAttention(shape[-1])
    out = m(th.randn(*shape))
    assert tuple(out.shape) == shape
    assert th.isfinite(out).all()


def test_scaled_dot_product_rows_are_convex_combinations():
    # output = softmax(QK^T)V : every output row is a convex combo of value rows, so it lies within
    # the value's per-feature [min, max] envelope. Use identity-ish maps via large/zero weights is
    # hard; instead assert the softmax-weighted mean stays finite and within value bounds per feature.
    th.manual_seed(0)
    m = ScaledDotProductAttention(4)
    x = th.randn(1, 5, 4)
    out = m(x)
    v = m.value(x)
    assert (out <= v.max(dim=1, keepdim=True).values + 1e-4).all()
    assert (out >= v.min(dim=1, keepdim=True).values - 1e-4).all()


# --- MultiHeadAttention -----------------------------------------------------
# Wraps th.nn.MultiheadAttention(batch_first=False); preserves the input shape.


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_multihead_preserves_shape(num_heads):
    m = MultiHeadAttention(8, num_heads=num_heads)
    out = m(th.randn(4, 8))
    assert tuple(out.shape) == (4, 8)
    assert th.isfinite(out).all()


def test_multihead_seq_first_3d_preserves_shape():
    # batch_first=False -> (seq, batch, dim) in/out.
    m = MultiHeadAttention(8, num_heads=2)
    out = m(th.randn(3, 4, 8))
    assert tuple(out.shape) == (3, 4, 8)


def test_multihead_rejects_indivisible_head_count():
    with pytest.raises(AssertionError):
        MultiHeadAttention(8, num_heads=3)


# --- AdditiveAttention ------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 8)])
def test_additive_attention_preserves_shape(shape):
    m = AdditiveAttention(shape[-1])
    out = m(th.randn(*shape))
    assert tuple(out.shape) == shape
    assert th.isfinite(out).all()


# --- GlobalContextAttention -------------------------------------------------
# scores = bmm(query, key.transpose(1,2)) -> (batch, seq, seq); works for any 3-D
# (batch, seq, hidden) input regardless of whether seq_len == hidden_dim.


def test_global_context_square_3d_preserves_shape():
    # Square seq_len == hidden_dim case.
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 8, 8))
    assert tuple(out.shape) == (4, 8, 8)
    assert th.isfinite(out).all()


def test_global_context_nonsquare_3d_should_work():
    # A general (batch, seq, hidden) input with seq != hidden is now handled correctly.
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 3, 8))
    assert tuple(out.shape) == (4, 3, 8)
    assert th.isfinite(out).all()


def test_global_context_tall_nonsquare_3d_should_work():
    # seq_len > hidden_dim also works (regression guard for the dropped redundant transpose).
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 16, 8))
    assert tuple(out.shape) == (4, 16, 8)
    assert th.isfinite(out).all()


@pytest.mark.xfail(
    reason="SEPARATE WORK: GlobalContextAttention's 3-D bmm path is now correct, but it still "
    "requires a 3-D input. Running it inside CustomQNetwork's flat 2-D (batch, features) MLP "
    "needs a separate 2-D integration redesign; a bare (batch, features) tensor still raises "
    "IndexError on transpose(1, 2)",
    strict=True,
)
def test_global_context_2d_mlp_input_should_work():
    # As wired into CustomQNetwork's flat MLP the block receives (batch, features) and should run.
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 8))
    assert tuple(out.shape)[0] == 4
