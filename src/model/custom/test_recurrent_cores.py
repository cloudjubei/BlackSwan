"""Contract + numerics tests for the non-LSTM recurrent cores.

Both cores present the EXACT nn.LSTM interface sb3-contrib's RecurrentActorCriticPolicy
threads: ``forward(input, (h, c)) -> (output, (h, c))`` with ``.input_size`` / ``.hidden_size``
/ ``.num_layers`` attributes. The integration tests drive each core through the unmodified
``RecurrentActorCriticPolicy._process_sequence`` to prove the drop-in actually holds.
"""

import numpy as np
import pytest
import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from src.model.custom.recurrent_cores import GRURecurrentCore, S4DRecurrentCore


# ---------------------------------------------------------------------------------------
# GRURecurrentCore — nn.LSTM-compatible contract
# ---------------------------------------------------------------------------------------

def _zero_state(num_layers, batch, hidden):
    z = th.zeros(num_layers, batch, hidden)
    return (z, z.clone())


def test_gru_core_exposes_lstm_like_attributes():
    core = GRURecurrentCore(input_size=8, hidden_size=16, num_layers=1)
    assert core.input_size == 8
    assert core.hidden_size == 16
    assert core.num_layers == 1


def test_gru_core_forward_returns_output_and_two_tuple_state():
    core = GRURecurrentCore(input_size=8, hidden_size=16)
    x = th.randn(3, 4, 8)  # (seq_len, batch, input_size)
    out, (h, c) = core(x, _zero_state(1, 4, 16))
    assert out.shape == (3, 4, 16)  # output width == hidden_size
    assert h.shape == (1, 4, 16)
    assert c.shape == (1, 4, 16)


def test_gru_core_cell_slot_is_inert_zeros():
    # A GRU has no cell state; the second state element exists only to satisfy the
    # (hidden, cell) buffer contract and must never carry information.
    core = GRURecurrentCore(input_size=5, hidden_size=7)
    x = th.randn(2, 3, 5)
    nonzero_cell = th.ones(1, 3, 7)
    _, (_, c) = core(x, (th.zeros(1, 3, 7), nonzero_cell))
    assert th.all(c == 0.0)


def test_gru_core_threads_memory_across_calls():
    core = GRURecurrentCore(input_size=5, hidden_size=7)
    x = th.randn(4, 2, 5)
    _, (h1, _) = core(x, _zero_state(1, 2, 7))
    # A non-trivial input must move the hidden state off zero.
    assert not th.allclose(h1, th.zeros_like(h1))


def test_gru_core_is_deterministic():
    core = GRURecurrentCore(input_size=5, hidden_size=7).eval()
    x = th.randn(4, 2, 5)
    s = _zero_state(1, 2, 7)
    out_a, (ha, _) = core(x, s)
    out_b, (hb, _) = core(x, s)
    assert th.allclose(out_a, out_b)
    assert th.allclose(ha, hb)


def test_gru_core_runs_through_process_sequence():
    # The real integration proof: the unmodified sb3-contrib sequence processor must drive
    # the core (it reads .input_size and the (h, c) tuple) without any awareness it is a GRU.
    core = GRURecurrentCore(input_size=6, hidden_size=9)
    n_seq, length = 2, 3
    features = th.randn(n_seq * length, 6)
    states = _zero_state(1, n_seq, 9)
    episode_starts = th.zeros(n_seq * length)
    out, (h, c) = RecurrentActorCriticPolicy._process_sequence(features, states, episode_starts, core)
    assert out.shape == (n_seq * length, 9)
    assert h.shape == (1, n_seq, 9)


def test_gru_core_resets_state_on_episode_start_via_process_sequence():
    # episode_starts == 1 must zero the incoming state for that step (the slow path in
    # _process_sequence multiplies the state by (1 - episode_start)).
    core = GRURecurrentCore(input_size=4, hidden_size=5)
    n_seq, length = 1, 4
    features = th.randn(length, 4)
    warm_h = th.randn(1, n_seq, 5)
    states = (warm_h, th.zeros(1, n_seq, 5))
    # First step starts a new episode -> the warm hidden state must be discarded.
    episode_starts = th.tensor([1.0, 0.0, 0.0, 0.0])
    out_reset, _ = RecurrentActorCriticPolicy._process_sequence(features, states, episode_starts, core)
    out_fresh, _ = RecurrentActorCriticPolicy._process_sequence(
        features, _zero_state(1, n_seq, 5), episode_starts, core
    )
    assert th.allclose(out_reset, out_fresh, atol=1e-6)


# ---------------------------------------------------------------------------------------
# S4DRecurrentCore — diagonal state-space model, complex state packed into (Re, Im)
# ---------------------------------------------------------------------------------------

def _s4d_zero_state(core, batch):
    z = th.zeros(core.num_layers, batch, core.hidden_size)
    return (z, z.clone())


def test_s4d_core_exposes_lstm_like_attributes_with_packed_state_width():
    core = S4DRecurrentCore(input_size=8, hidden_size=6, state_dim=4)
    assert core.input_size == 8
    assert core.num_layers == 1
    # The threaded state is the complex SSM state (channels x state_dim), split into Re/Im.
    # hidden_size is the PACKED width RecurrentPPO must allocate per (hidden, cell) tensor.
    assert core.hidden_size == 6 * 4
    # The per-step OUTPUT handed to the MLP head is one value per channel.
    assert core.out_features == 6


def test_s4d_core_forward_shapes():
    core = S4DRecurrentCore(input_size=8, hidden_size=6, state_dim=4)
    x = th.randn(3, 5, 8)  # (seq_len, batch, input_size)
    out, (h, c) = core(x, _s4d_zero_state(core, 5))
    assert out.shape == (3, 5, 6)  # (seq_len, batch, out_features)
    assert h.shape == (1, 5, 24)
    assert c.shape == (1, 5, 24)
    assert th.isfinite(out).all()


def test_s4d_core_is_finite_and_stable_over_long_sequence():
    # A diagonal SSM with Re(A) < 0 is contractive; a long constant drive must not blow up.
    core = S4DRecurrentCore(input_size=4, hidden_size=8, state_dim=16)
    x = th.ones(200, 2, 4)
    out, (h, c) = core(x, _s4d_zero_state(core, 2))
    assert th.isfinite(out).all()
    assert th.isfinite(h).all() and th.isfinite(c).all()
    assert h.abs().max() < 1e4


def test_s4d_core_zero_drive_keeps_zero_state_zero():
    # With no drive (u == 0) the recurrence is x_k = dA * x_{k-1}; from a zero state it must stay
    # exactly zero. The input projection's bias is zeroed so that a zero observation yields u == 0.
    core = S4DRecurrentCore(input_size=4, hidden_size=3, state_dim=5)
    with th.no_grad():
        core.in_proj.bias.zero_()
    x = th.zeros(6, 2, 4)
    _, (h, c) = core(x, _s4d_zero_state(core, 2))
    assert th.allclose(h, th.zeros_like(h), atol=1e-6)
    assert th.allclose(c, th.zeros_like(c), atol=1e-6)


def test_s4d_core_packed_state_resumes_recurrence_across_a_split():
    # Threading the returned (Re, Im) state must reproduce the unsplit run exactly — the proof
    # that packing the complex state into (hidden, cell) loses no information.
    core = S4DRecurrentCore(input_size=5, hidden_size=4, state_dim=8).eval()
    x = th.randn(10, 3, 5)
    out_whole, _ = core(x, _s4d_zero_state(core, 3))
    out_a, state_a = core(x[:4], _s4d_zero_state(core, 3))
    out_b, _ = core(x[4:], state_a)
    assert th.allclose(th.cat([out_a, out_b], dim=0), out_whole, atol=1e-5)


def test_s4d_core_matches_complex_reference_recurrence():
    # Independently validate the real-arithmetic SSM against a torch-complex implementation
    # driven by the SAME learned parameters. Guards every Re/Im bookkeeping step.
    th.manual_seed(0)
    core = S4DRecurrentCore(input_size=6, hidden_size=5, state_dim=7).eval()
    L, B = 8, 2
    x = th.randn(L, B, 6)
    out, _ = core(x, _s4d_zero_state(core, B))

    with th.no_grad():
        u = core.in_proj(x).to(th.cdouble)  # (L, B, H)
        A = (-th.exp(core.A_log_real) + 1j * core.A_imag).to(th.cdouble)  # (H, N)
        dt = th.exp(core.log_dt).to(th.cdouble)  # (H,)
        dA = th.exp(dt[:, None] * A)  # (H, N)
        dB = (dA - 1.0) / A  # B == 1
        C = (core.C_real + 1j * core.C_imag).to(th.cdouble)  # (H, N)
        D = core.D.to(th.cdouble)  # (H,)
        state = th.zeros(B, core.out_features, core.state_dim, dtype=th.cdouble)
        ys = []
        for k in range(L):
            state = dA * state + dB * u[k].unsqueeze(-1)
            y = 2.0 * (C * state).real.sum(-1) + (D * u[k]).real
            ys.append(y)
        ref = th.stack(ys).to(th.float32)
        ref = th.nn.functional.gelu(ref)
    assert th.allclose(out, ref, atol=1e-4)


def test_s4d_core_runs_through_process_sequence_with_reset():
    core = S4DRecurrentCore(input_size=4, hidden_size=3, state_dim=5)
    n_seq, length = 1, 4
    features = th.randn(length, 4)
    warm = th.randn(1, n_seq, core.hidden_size)
    states = (warm, th.randn(1, n_seq, core.hidden_size))
    episode_starts = th.tensor([1.0, 0.0, 0.0, 0.0])
    out_reset, _ = RecurrentActorCriticPolicy._process_sequence(features, states, episode_starts, core)
    out_fresh, _ = RecurrentActorCriticPolicy._process_sequence(
        features, _s4d_zero_state(core, n_seq), episode_starts, core
    )
    assert out_reset.shape == (length, 3)
    assert th.allclose(out_reset, out_fresh, atol=1e-5)


def test_s4d_core_is_trainable_gradients_reach_all_parameters():
    # Canonical S4D LEARNS A (both A_log_real and A_imag) from the S4D-Lin init — it is not frozen.
    # So A_imag is intentionally an nn.Parameter and must receive gradients; do NOT convert it to a
    # buffer. Stability comes from Re(A) = -exp(A_log_real) < 0, independent of whether A_imag moves.
    core = S4DRecurrentCore(input_size=4, hidden_size=3, state_dim=5)
    assert "A_imag" in dict(core.named_parameters())
    x = th.randn(5, 2, 4)
    out, _ = core(x, _s4d_zero_state(core, 2))
    out.sum().backward()
    for name, p in core.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert th.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_s4d_core_rejects_multilayer():
    # The (Re, Im) packing holds one complex state vector; stacking layers is not supported and must
    # fail loud rather than silently mis-reshape the threaded state.
    with pytest.raises(ValueError):
        S4DRecurrentCore(input_size=4, hidden_size=3, state_dim=5, num_layers=2)
