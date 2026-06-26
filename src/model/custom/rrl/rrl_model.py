"""Recurrent Reinforcement Learning (Moody & Saffell 2001, "Learning to Trade via Direct Reinforcement").

A DIRECT-policy agent — no Q-values, no SB3. A recurrent linear policy outputs a CONTINUOUS target
position F_t = tanh(w·φ(x_t) + u·F_{t-1} + b) in [-1, 1] (the previous position feeds back, so it learns to
hold through noise), trained by gradient ASCENT on the Sharpe ratio of the realised strategy return
    R_t = F_{t-1}·r_t − δ·|F_t − F_{t-1}|        (position return minus transaction cost on rebalancing)
which is exactly the objective the online *differential* Sharpe ratio estimates. Trained with truncated
BPTT over the return series; at eval the continuous position is discretised (long / flat / short) and the
env's discrete action that moves toward it is emitted.

φ is a feature ENCODER hook — identity here, a fixed Echo State Network reservoir in ESN-RRL (the
subclass), so both share the whole policy/training/discretisation machinery.
"""

import numpy as np
import torch
import torch.nn as nn

from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv
from src.model.abstract_model import BaseRLModel

_DEADBAND = 0.05  # |F| below this is treated as flat — avoids churning on near-zero positions
_TBPTT = 256  # truncated-BPTT window: the projection is vectorised per chunk, only the recurrence loops


class RRLModel(BaseRLModel):
    def __init__(self, config: ModelConfig, env: AbstractEnv, device: str):
        super(RRLModel, self).__init__(config)
        self.device = device
        self.obs_dim = int(np.asarray(env.data_provider.get_values(0)).reshape(-1).shape[0])
        self.feat_dim = self._setup_encoder(self.obs_dim, env)  # subclass may build φ + change the dim
        self.proj = nn.Linear(self.feat_dim, 1).to(device)  # w, plus its bias
        self.u = nn.Parameter(torch.zeros((), device=device))  # recurrent weight on F_{t-1}
        self.allow_short = bool(env.env_config.allow_shorting)
        self.delta = float(env.env_config.transaction_fee)

    # --- feature encoder φ (identity here; ESN-RRL overrides) -------------------------------------
    def _setup_encoder(self, obs_dim: int, env: AbstractEnv) -> int:
        return obs_dim

    def _encode_sequence(self, feats: np.ndarray) -> torch.Tensor:
        """(T, obs_dim) numpy -> (T, feat_dim) tensor, for training."""
        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    def _reset_encoder(self) -> None:
        pass

    def _encode_step(self, x: np.ndarray) -> torch.Tensor:
        """One step's (obs_dim,) numpy -> (feat_dim,) tensor, for eval (may carry encoder state)."""
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    # --- lifecycle --------------------------------------------------------------------------------
    def is_pretrained(self) -> bool:  # RRL keeps its own weights; always (re)train, never resume a zip
        return False

    def produces_checkpoint(self) -> bool:
        return False

    def train(self, env: AbstractEnv) -> None:
        dp = env.data_provider
        steps = dp.get_timesteps()
        feats = np.stack([np.asarray(dp.get_values(t), dtype=np.float32).reshape(-1) for t in range(steps)])
        prices = np.array([float(dp.get_price(t)) for t in range(steps)], dtype=np.float64)
        returns = np.zeros(steps, dtype=np.float32)
        returns[1:] = prices[1:] / np.where(prices[:-1] == 0, np.nan, prices[:-1]) - 1.0
        r = torch.tensor(np.nan_to_num(returns), dtype=torch.float32, device=self.device)

        feat_seq = self._encode_sequence(feats)  # (T, feat_dim) — φ applied once (fixed for ESN)
        opt = torch.optim.Adam(
            [*self.proj.parameters(), self.u], lr=float(self.rl_config.learning_rate or 1e-3)
        )
        for _ in range(max(1, int(self.rl_config.episodes))):
            f_prev = torch.zeros((), device=self.device)
            for start in range(1, steps, _TBPTT):
                end = min(start + _TBPTT, steps)
                proj = self.proj(feat_seq[start:end]).squeeze(-1)  # w·φ(x)+b, vectorised, fresh weights
                positions = []
                fp = f_prev
                for k in range(end - start):
                    fp = torch.tanh(proj[k] + self.u * fp)
                    positions.append(fp)
                if not positions:
                    continue
                F = torch.stack(positions)  # positions at t = start..end-1
                F_lag = torch.cat([f_prev.detach().reshape(1), F[:-1]])  # F_{t-1}
                strat = F_lag * r[start:end] - self.delta * torch.abs(F - F_lag)
                sharpe = strat.mean() / (strat.std() + 1e-8)
                opt.zero_grad()
                (-sharpe).backward()
                opt.step()
                f_prev = F[-1].detach()

    def _action_for(self, env: AbstractEnv, target: int) -> int:
        """The discrete action that moves the env from its current position toward `target` (1/0/-1)."""
        pos = env.positions[-1]
        current = 1 if pos > 0 else (-1 if pos < 0 else 0)
        if current == target:
            return 0  # hold
        if current == 0:
            return 1 if target == 1 else 3  # open long / open short
        return 2 if current == 1 else 4  # close long / cover short (re-open next step if flipping)

    def test(self, env: AbstractEnv, deterministic: bool = True) -> None:
        dp = env.data_provider
        env.reset()
        self.proj.eval()
        self._reset_encoder()
        f_prev = torch.zeros((), device=self.device)
        with torch.no_grad():
            while True:
                feat = self._encode_step(np.asarray(dp.get_values(env.current_step), dtype=np.float32).reshape(-1))
                f_prev = torch.tanh(self.proj(feat).squeeze(-1) + self.u * f_prev)
                f = float(f_prev.item())
                target = 1 if f > _DEADBAND else (-1 if (f < -_DEADBAND and self.allow_short) else 0)
                _, _, done, _, _ = env.step(self._action_for(env, target))
                if done:
                    break
