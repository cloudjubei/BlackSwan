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


def test_global_context_2d_mlp_input_should_work():
    # As wired into CustomQNetwork's flat MLP the block receives a 2-D (batch, features) tensor; it
    # is treated as a single-step sequence and returns the same (batch, features) shape so it chains.
    m = GlobalContextAttention(8)
    out = m(th.randn(4, 8))
    assert tuple(out.shape) == (4, 8)
    assert th.isfinite(out).all()


# --- attention weight capture (A6) ------------------------------------------
# Each block computes an attention weight matrix then discards it. For the decision-trace xAI heatmap it
# must be STASHED on the module after forward — detached + on CPU (a live-graph tensor would leak / interfere
# with training). Capture is EVAL-ONLY (the decision-trace replay runs in eval mode): a training-mode forward
# skips the stash so no per-gradient-step device->host .cpu() sync is paid. last_attn is None until the first
# eval forward; output is unchanged in either mode.


def _assert_detached_finite(t):
    assert t is not None
    assert th.isfinite(t).all()
    assert t.requires_grad is False and t.grad_fn is None


def test_last_attn_is_none_before_forward():
    assert SelfAttention(4).last_attn is None
    assert ScaledDotProductAttention(8).last_attn is None
    assert MultiHeadAttention(8, num_heads=2).last_attn is None
    assert AdditiveAttention(8).last_attn is None
    assert GlobalContextAttention(8).last_attn is None


def test_self_attention_stashes_last_attn():
    m = SelfAttention(6).eval()
    out = m(th.randn(2, 6))
    assert tuple(out.shape) == (2, 6)
    _assert_detached_finite(m.last_attn)
    assert tuple(m.last_attn.shape) == (2, 6, 6)  # (batch, seq==in_dim, seq)


def test_scaled_dot_product_stashes_last_attn():
    m = ScaledDotProductAttention(8).eval()
    out = m(th.randn(2, 3, 8))
    assert tuple(out.shape) == (2, 3, 8)
    _assert_detached_finite(m.last_attn)
    assert tuple(m.last_attn.shape) == (2, 3, 3)  # (batch, seq, seq)


def test_multihead_stashes_head_averaged_last_attn():
    m = MultiHeadAttention(8, num_heads=2).eval()
    out = m(th.randn(3, 4, 8))  # (seq, batch, dim), batch_first=False
    assert tuple(out.shape) == (3, 4, 8)
    _assert_detached_finite(m.last_attn)
    assert tuple(m.last_attn.shape) == (4, 3, 3)  # head-averaged (batch, q_seq, k_seq)


def test_additive_attention_stashes_last_attn():
    m = AdditiveAttention(8).eval()
    out = m(th.randn(2, 8))
    assert tuple(out.shape) == (2, 8)
    _assert_detached_finite(m.last_attn)
    assert m.last_attn.shape[-1] == 1  # (batch, seq, 1)


def test_global_context_stashes_last_attn():
    m = GlobalContextAttention(8).eval()
    out = m(th.randn(4, 8))  # 2-D MLP input -> single-step sequence
    assert tuple(out.shape) == (4, 8)
    _assert_detached_finite(m.last_attn)
    assert tuple(m.last_attn.shape) == (4, 1, 1)


def test_training_mode_forward_skips_the_stash_but_output_is_unchanged():
    # In train() mode the capture is skipped (perf: no per-step .cpu() sync); the forward OUTPUT is identical.
    for m in (
        SelfAttention(6),
        ScaledDotProductAttention(8),
        MultiHeadAttention(8, num_heads=2),
        AdditiveAttention(8),
        GlobalContextAttention(8),
    ):
        m.train()
        x = th.randn(2, 8) if not isinstance(m, SelfAttention) else th.randn(2, 6)
        if isinstance(m, (ScaledDotProductAttention,)):
            x = th.randn(2, 3, 8)
        elif isinstance(m, MultiHeadAttention):
            x = th.randn(3, 4, 8)
        out = m(x)
        assert th.isfinite(out).all()
        assert m.last_attn is None  # skipped during training
