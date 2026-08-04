import numpy as np
import stable_baselines3
from src.model.abstract_model import BaseRLModel
from src.conf.model_config import ModelConfig, ModelRLConfig
from src.environment.abstract_env import AbstractEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import ProgressBarCallback
import os

# based on https://stable-baselines3.readthedocs.io/en/master/modules/base.html

# RecurrentPPO-backed models (see model_factory): their recurrent state MUST be threaded across
# predict() calls at eval. Miss one and SB3 re-zeros the memory every step, so a model trained WITH
# memory is evaluated WITHOUT it — a silent train/eval mismatch. "reppo-custom" was previously omitted.
# Covers the LSTM (reppo*), GRU (gru*) and diagonal-SSM (s4d*) cores — eval threads opaque state, so
# the same branch serves all three.
RECURRENT_MODEL_NAMES = ("reppo", "reppo-custom", "gru", "gru-custom", "s4d", "s4d-custom")


def is_recurrent_model_name(model_name) -> bool:
    return model_name in RECURRENT_MODEL_NAMES


def fold_eval_parametrizations(policy) -> int:
    """Bake every weight reparametrization (weight_norm / spectral_norm) into a plain weight for EVAL.

    weight_norm recomputes the effective weight from its (g, v) factors on EVERY forward; at eval those
    factors are frozen, so removing the parametrization while KEEPING the current weight
    (``leave_parametrized=True``) is numerically identical and drops the per-forward recompute. The
    default reppo-custom net carries three weight_norm layers, so this trims each test/replay forward.
    Idempotent; returns the number of modules folded."""
    from torch.nn.utils import parametrize

    folded = 0
    for module in policy.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            folded += 1
    return folded


class RLModel(BaseRLModel):
    def __init__(self, config: ModelConfig, rl_model: BaseAlgorithm):
        super(RLModel, self).__init__(config)

        self.rl_model = rl_model

    def train(self, env: AbstractEnv):
        timesteps = env.get_timesteps()

        # A continue_from run extra-trains a LOADED parent checkpoint on a NEW dataset: bind the new env to
        # the loaded model and NEVER reset the step counter (keep the parent's timesteps + anneal schedules).
        continuing = self.rl_config.continue_from is not None
        if continuing:
            self.rl_model.set_env(env)

        # A6: optionally snapshot a RETAINED checkpoint every N timesteps (for mid-training decision traces)
        # — only when a snapshot_interval is set AND checkpoints are kept (a snapshot must persist). The SAME
        # callback rides every episode so its thresholds accumulate across the whole run. None otherwise, so
        # the default learn() call (callback=None) is byte-identical.
        snapshot_interval = getattr(self.rl_config, "snapshot_interval", None)
        callback = None
        if snapshot_interval and self.config.save_checkpoint:
            from src.model.custom.snapshot_callback import SnapshotCheckpointCallback
            callback = SnapshotCheckpointCallback(
                interval=snapshot_interval,
                save_fn=self.rl_model.save,
                base_id=self.id,
                folder=self.rl_config.checkpoints_folder,
                max_snapshots=getattr(self.rl_config, "snapshot_cap", None),
            )

        for i in range(0, self.rl_config.episodes):
            print(f"TRAINING EPISODE {i+1}/{self.rl_config.episodes}")
            # Only the first episode resets the step counter; later episodes continue it so
            # learning_starts is paid once and the exploration schedule anneals across all episodes. A
            # continued run never resets — it keeps climbing from the loaded checkpoint's progress.
            self.rl_model.learn(total_timesteps=timesteps, progress_bar=self.rl_config.progress_bar, log_interval=1000, reset_num_timesteps=(i == 0 and not continuing), callback=callback)

        # Retained mid-training snapshots (empty unless the callback ran) — run.py traces each afterwards.
        self.snapshots = callback.snapshots if callback else []

        if self.config.save_checkpoint:
            path = os.path.join(self.rl_config.checkpoints_folder, self.id)
            self.rl_model.save(path)
            print(f"Saved RL model to {path}")
        else:
            print("Checkpoint saving disabled (config.save_checkpoint=False) — skipping RL model save")

    def test(self, env: AbstractEnv, deterministic: bool = True, progress_bar: bool = True):
        obs, _ = env.reset()

        # Fold weight_norm/spectral_norm into plain weights for the eval passes (this test + the
        # decision-trace replay that reuses this policy): numerically identical, skips the per-forward
        # weight recompute. Best-effort — JAX/sbx policies aren't torch modules, so guard and move on.
        try:
            fold_eval_parametrizations(self.rl_model.policy)
        except Exception:
            pass

        if progress_bar:
            fake_model = stable_baselines3.dqn.DQN(env=env, policy='MlpPolicy')
            progress = ProgressBarCallback()
            progress.init_callback(fake_model)
            progress.num_timesteps = env.get_timesteps()
            progress.on_training_start({"total_timesteps": progress.num_timesteps}, {})
            self.progress = progress

        if is_recurrent_model_name(self.rl_config.model_name):
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,), dtype=bool)
            while True:
                action, lstm_states = self.rl_model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
                obs, rewards, done, finished_early, info = env.step(action)
                episode_starts = done

                if done:
                    break

                if progress_bar:
                    self.progress.on_step()
        else:
            while True:
                (action, extra_info) = self.rl_model.predict(obs, deterministic=deterministic)
                obs, reward, done, finished_early, info = env.step(action)

                if done:
                    break
                
                if progress_bar:
                    self.progress.on_step()

        if progress_bar:
            self.progress.on_training_end()

    def predict(self, env: AbstractEnv, deterministic: bool = True):
        (action, extra_info) = self.rl_model.predict(env.last_obs, deterministic=deterministic)
        obs, reward, done, finished_early, info = env.step(action)

        if isinstance(action, np.ndarray):
            action = action.item()
        return -1 if action == 2 else action
    
    def predictOnline(self, observation: np.ndarray, deterministic: bool = True):
        (action, extra_info) = self.rl_model.predict(observation, deterministic=deterministic)
        
        if isinstance(action, np.ndarray):
            action = action.item()
        return -1 if action == 2 else action