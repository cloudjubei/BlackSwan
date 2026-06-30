"""Non-LSTM recurrent cores that drop into sb3-contrib's RecurrentActorCriticPolicy.

sb3-contrib threads recurrent memory as a ``(hidden, cell)`` tuple and drives the core through
``RecurrentActorCriticPolicy._process_sequence``, which only ever:
  * reads ``core.input_size`` to reshape the flat batch into a sequence,
  * calls ``core(input, (hidden, cell))`` expecting ``(output, (hidden, cell))`` back, and
  * masks BOTH state tensors by ``(1 - episode_start)`` to reset memory at episode starts.
RecurrentPPO additionally reads ``core.num_layers`` / ``core.hidden_size`` to size its rollout
buffers. Any module honouring that surface is a drop-in replacement for ``nn.LSTM`` — these cores
exploit exactly that seam, so the policy/buffer machinery stays untouched.
"""

from typing import Tuple

import torch as th
from torch import nn

# S4D diagonal state-space defaults (Gu, Goel & Re 2022, "On the Parameterization and
# Initialization of Diagonal State Space Models"). dt range and the S4D-Lin A-initialisation.
S4D_DEFAULT_STATE_DIM = 64
S4D_DT_MIN = 1e-3
S4D_DT_MAX = 1e-1
S4D_A_REAL_INIT = 0.5  # |Re(A)|; A_real is pinned negative as -S4D_A_REAL_INIT


class GRURecurrentCore(nn.Module):
    """A GRU presented through the ``nn.LSTM`` ``(hidden, cell)`` interface.

    A GRU keeps a single hidden state, so the ``cell`` slot is inert: it is ignored on input and
    returned as zeros. The reset mask the policy applies to it is therefore a no-op, and the
    rollout buffer simply stores zeros for the unused half — no correctness impact.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)

    def forward(
        self, input: th.Tensor, states: Tuple[th.Tensor, th.Tensor]
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        output, hidden = self.gru(input, states[0].contiguous())
        return output, (hidden, th.zeros_like(hidden))


class S4DRecurrentCore(nn.Module):
    """A diagonal state-space (S4D) layer presented through the ``nn.LSTM`` interface.

    Each of the ``hidden_size`` channels owns an independent diagonal SSM with ``state_dim``
    complex modes: ``x_k = exp(dt*A) x_{k-1} + dB u_k`` (ZOH discretisation, ``B = 1``), read out
    as ``y_k = 2 Re(sum_N C x_k) + D u_k``. ``A`` is a learnable complex diagonal, INITIALISED
    S4D-Lin (``Re(A) = -0.5`` via a log-parameter that keeps it negative — hence contractive —
    under training; ``Im(A) = pi*n``) and then LEARNED along with ``C`` / ``D`` / ``dt``. This
    follows canonical S4D, where ``A`` is trained from the S4D-Lin init rather than frozen; the
    log-parameterisation of ``Re(A)`` (not ``A_imag``) is what guarantees stability throughout.

    The complex state is packed into the policy's two real state tensors — ``hidden = Re(x)`` and
    ``cell = Im(x)`` — so the SSM recurs over env time exactly like the LSTM arm, threaded by
    RecurrentPPO. The recurrence is done in pure real arithmetic (no ``torch.complex``) so it runs
    identically on MPS, CPU and CUDA. ``hidden_size`` here is the channel count (the per-step output
    width); the threaded-state width reported to the buffer is ``hidden_size * state_dim``. The
    packing is single-layer by construction, so ``num_layers`` must be 1.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_dim: int = S4D_DEFAULT_STATE_DIM,
        num_layers: int = 1,
        dt_min: float = S4D_DT_MIN,
        dt_max: float = S4D_DT_MAX,
    ):
        super().__init__()
        if num_layers != 1:
            raise ValueError(
                f"S4DRecurrentCore packs a single complex state into (Re, Im); num_layers must be 1, got {num_layers}."
            )
        channels = hidden_size
        self.input_size = input_size
        self.out_features = channels
        self.state_dim = state_dim
        self.num_layers = num_layers
        # Width of EACH packed (Re / Im) state tensor RecurrentPPO must allocate per env.
        self.hidden_size = channels * state_dim

        self.in_proj = nn.Linear(input_size, channels)

        log_dt = th.rand(channels) * (th.log(th.tensor(dt_max)) - th.log(th.tensor(dt_min))) + th.log(
            th.tensor(dt_min)
        )
        self.log_dt = nn.Parameter(log_dt)
        # A = -exp(A_log_real) + i*A_imag keeps Re(A) strictly negative (contractive) under training.
        self.A_log_real = nn.Parameter(th.full((channels, state_dim), float(th.log(th.tensor(S4D_A_REAL_INIT)))))
        self.A_imag = nn.Parameter(th.pi * th.arange(state_dim, dtype=th.float32).repeat(channels, 1))
        self.C_real = nn.Parameter(th.randn(channels, state_dim))
        self.C_imag = nn.Parameter(th.randn(channels, state_dim))
        self.D = nn.Parameter(th.ones(channels))
        self.act = nn.GELU()

    def _discretize(self):
        """ZOH discretisation of the diagonal SSM, in real (re, im) component pairs."""
        a_real = -th.exp(self.A_log_real)
        a_imag = self.A_imag
        dt = th.exp(self.log_dt).unsqueeze(-1)  # (channels, 1)
        dta_real, dta_imag = dt * a_real, dt * a_imag
        mag = th.exp(dta_real)
        da_real = mag * th.cos(dta_imag)
        da_imag = mag * th.sin(dta_imag)
        # dB = (dA - 1) / A  (B == 1), complex divide by A = a_real + i*a_imag.
        num_real, num_imag = da_real - 1.0, da_imag
        denom = a_real * a_real + a_imag * a_imag
        db_real = (num_real * a_real + num_imag * a_imag) / denom
        db_imag = (num_imag * a_real - num_real * a_imag) / denom
        return da_real, da_imag, db_real, db_imag

    def forward(
        self, input: th.Tensor, states: Tuple[th.Tensor, th.Tensor]
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        length, batch, _ = input.shape
        u = self.in_proj(input)  # (length, batch, channels)
        x_real = states[0].reshape(batch, self.out_features, self.state_dim)
        x_imag = states[1].reshape(batch, self.out_features, self.state_dim)

        da_real, da_imag, db_real, db_imag = self._discretize()
        outputs = []
        for k in range(length):
            u_k = u[k]  # (batch, channels)
            u_e = u_k.unsqueeze(-1)  # (batch, channels, 1) -> broadcast over state_dim
            new_real = da_real * x_real - da_imag * x_imag + db_real * u_e
            new_imag = da_real * x_imag + da_imag * x_real + db_imag * u_e
            x_real, x_imag = new_real, new_imag
            y_k = 2.0 * (self.C_real * x_real - self.C_imag * x_imag).sum(-1) + self.D * u_k
            outputs.append(y_k)

        output = self.act(th.stack(outputs, dim=0))  # (length, batch, channels)
        h_new = x_real.reshape(1, batch, self.hidden_size)
        c_new = x_imag.reshape(1, batch, self.hidden_size)
        return output, (h_new, c_new)
