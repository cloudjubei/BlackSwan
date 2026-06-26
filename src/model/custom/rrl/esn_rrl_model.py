"""ESN-RRL (Borrageiro 2022, "The Recurrent Reinforcement Learning Crypto Agent").

RRL whose feature map φ is a fixed ECHO STATE NETWORK reservoir: a random recurrent network, scaled to a
sub-unit spectral radius (the echo-state property), is run over the inputs WITHOUT training; its
high-dimensional leaky-integrator state is the feature the (trained) RRL policy reads. Only the reservoir
READOUT — i.e. the RRL position layer — is learned; the reservoir itself is frozen. Inherits the whole
RRL policy / Sharpe-ascent training / discretised-action eval from RRLModel; only φ changes.
"""

import numpy as np
import torch

from src.environment.abstract_env import AbstractEnv
from src.model.custom.rrl.rrl_model import RRLModel

_RESERVOIR = 128  # reservoir units
_LEAK = 0.3  # leaky-integrator rate α
_SPECTRAL_RADIUS = 0.9  # < 1 for the echo-state property
_INPUT_SCALE = 0.5


class ESNRRLModel(RRLModel):
    def _setup_encoder(self, obs_dim: int, env: AbstractEnv) -> int:
        n = _RESERVOIR
        gen = torch.Generator().manual_seed(int(self.rl_config.seed) if self.rl_config.seed is not None else 0)
        self._w_in = ((torch.rand(n, obs_dim, generator=gen) * 2 - 1) * _INPUT_SCALE).to(self.device)
        w = torch.rand(n, n, generator=gen) * 2 - 1
        spectral = torch.linalg.eigvals(w).abs().max().real
        self._w_res = (w * (_SPECTRAL_RADIUS / (spectral + 1e-8))).to(self.device)  # frozen reservoir
        self._alpha = _LEAK
        self._state = None
        return n

    def _reservoir_step(self, x: torch.Tensor) -> torch.Tensor:
        prev = self._state if self._state is not None else torch.zeros(_RESERVOIR, device=self.device)
        pre = self._w_in @ x + self._w_res @ prev
        self._state = (1 - self._alpha) * prev + self._alpha * torch.tanh(pre)
        return self._state

    def _encode_sequence(self, feats: np.ndarray) -> torch.Tensor:
        X = torch.tensor(feats, dtype=torch.float32, device=self.device)
        self._state = None
        with torch.no_grad():
            states = [self._reservoir_step(X[t]) for t in range(X.shape[0])]
        return torch.stack(states)  # (T, reservoir) — fixed features, no grad

    def _reset_encoder(self) -> None:
        self._state = None

    def _encode_step(self, x: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self._reservoir_step(torch.tensor(x, dtype=torch.float32, device=self.device))
