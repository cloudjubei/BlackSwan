"""Forward-pass shape correctness for the custom attention blocks used in CustomQNetwork MLPs.

Tiny CPU tensors only. These blocks sit *inside* a flat MLP, so the realistic input is a 2-D
(batch, features) tensor. We assert each block's forward runs and preserves the feature width
where the contract implies it should, and document the GlobalContextAttention shape restriction
(only square 3-D inputs work) as an xfail bug.
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
# Comment claims (batch, seq, hidden), but bmm(query.transpose(1,2), key) silently transposes the
# query back, so the contraction only aligns when seq_len == hidden_dim. Anything else raises.


def test_global_context_square_3d_preserves_shape():
    # seq_len == hidden_dim is the only shape that currently works.
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 8, 8))
    assert tuple(out.shape) == (4, 8, 8)
    assert th.isfinite(out).all()


@pytest.mark.xfail(
    reason="BUG: GlobalContextAttention transposes query and then transposes it back before bmm, "
    "so scores only align when seq_len == hidden_dim; a generic (batch, seq, hidden) input raises",
    strict=False,
)
def test_global_context_nonsquare_3d_should_work():
    # The documented contract is a general (batch, seq, hidden) input; seq != hidden should be fine.
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 3, 8))
    assert tuple(out.shape) == (4, 3, 8)


@pytest.mark.xfail(
    reason="BUG: GlobalContextAttention requires 3-D input (uses transpose(1,2)), but in a "
    "CustomQNetwork MLP it is fed a 2-D (batch, features) tensor and crashes with IndexError",
    strict=False,
)
def test_global_context_2d_mlp_input_should_work():
    # As wired into CustomQNetwork's flat MLP the block receives (batch, features) and should run.
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 8))
    assert tuple(out.shape)[0] == 4
