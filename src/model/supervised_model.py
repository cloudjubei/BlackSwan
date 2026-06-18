import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.conf.model_config import ModelConfig
from src.environment.abstract_env import AbstractEnv
from src.model.abstract_model import AbstractModel


class SupervisedModel(AbstractModel):
    """A non-RL baseline: a supervised classifier predicts next-bar direction from the same market
    features the RL agent observes, and a fixed rules layer turns each prediction into a trade
    (go long on a predicted up-move, close — or short, when enabled — on a predicted down-move).
    Exits still flow through the env's TP/SL. It runs the unchanged env so its RunSummary is directly
    comparable to the RL runs."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.supervised_config = config.model_supervised
        self._clf = None
        self._constant = None

    def get_id(self, config: ModelConfig):
        c = config.model_supervised
        return f'{config.model_type}_{c.model_name}_h{c.forward_horizon}_p{c.prob_threshold}_{time.time()}'.replace('.', '~')

    def is_pretrained(self):
        return False

    def produces_checkpoint(self) -> bool:
        return False

    @staticmethod
    def _features(data_provider, step) -> np.ndarray:
        """Flatten the provider's market values at ``step`` into a 1-D feature vector. Identical layout
        is used at fit and at predict time, so the exact ordering is immaterial to the classifier."""
        values = data_provider.get_values(step)
        if isinstance(values, (list, tuple)):
            arrays = [np.asarray(v, dtype=np.float32).reshape(-1) for v in values]
            return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
        return np.asarray(values, dtype=np.float32).reshape(-1)

    def _make_classifier(self):
        seed = int(self.supervised_config.seed or 0)
        if 'gbm' in self.supervised_config.model_name:
            return HistGradientBoostingClassifier(random_state=seed)
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))

    def _fit(self, features: np.ndarray, labels: np.ndarray):
        classes = set(np.asarray(labels).tolist())
        if features.shape[0] == 0 or len(classes) < 2:
            self._constant = int(classes.pop()) if classes else 1
            self._clf = None
            return
        self._constant = None
        self._clf = self._make_classifier()
        self._clf.fit(np.nan_to_num(features), np.asarray(labels))

    def _predict_up(self, feature: np.ndarray) -> bool:
        if self._clf is None:
            return bool(self._constant)
        probabilities = self._clf.predict_proba(np.nan_to_num(feature).reshape(1, -1))[0]
        classes = list(self._clf.classes_)
        up = probabilities[classes.index(1)] if 1 in classes else 0.0
        return bool(up > self.supervised_config.prob_threshold)

    def _action_for(self, env: AbstractEnv, up: bool) -> int:
        position = env.positions[-1] if env.positions else 0
        allow_shorting = bool(getattr(env.env_config, "allow_shorting", False))
        if up:
            if position < 0:
                return 4  # cover a short
            if position == 0:
                return 1  # open a long
            return 0  # already long — hold
        if position > 0:
            return 2  # close the long
        if position == 0 and allow_shorting:
            return 3  # open a short
        return 0

    def train(self, env: AbstractEnv):
        provider = env.data_provider
        horizon = max(1, int(self.supervised_config.forward_horizon))
        timesteps = provider.get_timesteps()
        features, labels = [], []
        for step in range(0, timesteps - horizon + 1):
            current = provider.get_price(step)
            future = provider.get_price(step + horizon)
            if not current > 0:
                continue
            features.append(self._features(provider, step))
            labels.append(1 if (future / current - 1.0) > 0 else 0)
        self._fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=int))

    def test(self, env: AbstractEnv, deterministic: bool = True):
        env.reset()
        provider = env.data_provider
        while True:
            feature = self._features(provider, env.current_step)
            action = self._action_for(env, self._predict_up(feature))
            _, _, done, _, _ = env.step(action)
            if done:
                break
