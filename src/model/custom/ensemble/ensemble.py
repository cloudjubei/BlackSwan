from typing import Optional, Tuple

import numpy as np
import torch as th


class EnsembleModel:
    """A bagged ensemble of N independently-trained DQN-family members.

    Predicts with the standard discrete-action ensemble rule — AVERAGE the members' Q-values, then argmax —
    and trains each member on the same env. It duck-types the subset of the SB3 ``BaseAlgorithm`` surface
    that ``RLModel`` uses (``learn`` / ``predict`` / ``save`` / ``policy`` / ``device``); it is deliberately
    NOT an ``OffPolicyAlgorithm`` subclass — inheriting that class's ``learn`` (without a real
    ``super().__init__``) was the original bug (``AttributeError: ... has no attribute 'device'``)."""

    def __init__(self, ensemble):
        if not ensemble:
            raise ValueError("EnsembleModel requires at least one member model")
        self.ensemble = ensemble
        # Expose the first member's policy/device so RLModel's eval-fold + any device reads resolve; all
        # members share the same env + device by construction.
        self.policy = ensemble[0].policy
        self.device = ensemble[0].device

    def set_logger(self, logger) -> None:
        self._logger = logger
        for model in self.ensemble:
            if hasattr(model, "set_logger"):
                model.set_logger(logger)

    def learn(self, total_timesteps, **kwargs):
        for i, model in enumerate(self.ensemble):
            print(f"ENSEMBLE member {i + 1}/{len(self.ensemble)}")
            model.learn(total_timesteps=total_timesteps, **kwargs)
        return self

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Average the members' Q-values and take the argmax (the discrete-action ensemble vote)."""
        q_sum = None
        for model in self.ensemble:
            model.policy.set_training_mode(False)
            obs_tensor, _ = model.policy.obs_to_tensor(observation)
            with th.no_grad():
                q = model.q_net(obs_tensor)
            q_sum = q if q_sum is None else q_sum + q
        actions = th.argmax(q_sum, dim=1).reshape(-1).cpu().numpy()
        return actions, state

    def save(self, path: str) -> None:
        for i, model in enumerate(self.ensemble):
            model.save(f"{path}_member{i}")

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        for model in self.ensemble:
            if hasattr(model, "set_random_seed"):
                model.set_random_seed(seed)
