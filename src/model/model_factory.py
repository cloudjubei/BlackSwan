from omegaconf import ListConfig
import ray.rllib.algorithms.ppo
import ray.rllib.models
import ray.rllib.models.preprocessors
import ray.rllib.utils.spaces.space_utils
import sbx.common
import sbx.common.type_aliases
import sbx.core
import sbx.dqn
import sbx.ppo
from src.conf.model_config import ModelConfigSearch, ModelConfig, ModelTechnicalConfig, ModelTimeConfig, ModelMomentumConfig, ModelDayConfig, ModelRLConfig, ModelRegressionConfig, ModelSupervisedConfig
from src.model.custom.agent57.agent57 import Agent57
from src.model.custom.customqnetwork import CustomQNetwork
from src.model.custom.dgwo import DGWO
from src.model.custom.ensemble.ensemble import EnsembleModel
from src.model.custom.policies import CustomActorCriticPolicy, CustomDQNPolicy, CustomDuelingDQNPolicy, CustomQRDQNPolicy, CustomRainbowPolicy, CustomRecurrentActorCriticPolicy
from src.model.custom.sequence_extractor import SequenceFeaturesExtractor
from src.model.custom.policy_iqn import CustomIQNPolicy
from src.model.dqn_lstm_policy import LSTMFCE
from src.model.dueling_dqn.dueling_dqn import DuelingDQN
from src.model.dueling_dqn.policies import DuelingDQNPolicy
from src.model.hodl_model import HodlModel
from src.model.supervised_model import SupervisedModel
from src.model.iqn.iqn import IQN
from src.model.munchhausen_dqn.munchhausen_dqn import MunchausenDQN
from src.model.rainbow_dqn.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.model.rainbow_dqn.rainbow_dqn import RainbowDQN
from src.model.regression_model import RegressionModel
from src.model.rl_model import RLModel
from src.model.time_strategy_model import TimeStrategyModel
from src.model.technical_strategy_model import TechnicalStrategyModel
from src.model.momentum_strategy_model import MomentumStrategyModel
from src.model.day_of_week_strategy_model import DayOfWeekStrategyModel
from src.environment.abstract_env import AbstractEnv

from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from typing import Any, Dict, List, Tuple, Union, get_origin
from stable_baselines3 import DQN, PPO, A2C, HerReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib import RecurrentPPO, ARS, QRDQN, TRPO, TQC
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
import sbx
import optax
import torch

import sys
import itertools
import os

optimizer_classes = {
    "Adam" : torch.optim.Adam,
    "AdamW" : torch.optim.AdamW,
    "Adadelta" : torch.optim.Adadelta,
    "Adagrad" : torch.optim.Adagrad,
    "Adamax" : torch.optim.Adamax,
    "ASGD" : torch.optim.ASGD,
    "SparseAdam" : torch.optim.SparseAdam,
    "LBFGS" : torch.optim.LBFGS,
    "NAdam" : torch.optim.NAdam,
    "RAdam" : torch.optim.RAdam,
    "RMSprop" : torch.optim.RMSprop,
    "Rprop" : torch.optim.Rprop,
    "SGD" : torch.optim.SGD,
    "optax.adam" : optax.adam,
    "optax.adamw" : optax.adamw,
    "optax.adadelta" : optax.adadelta,
    "optax.adagrad" : optax.adagrad,
    "optax.adamax" : optax.adamax,
    "optax.amsgrad" : optax.amsgrad,
    "optax.nadam" : optax.nadam,
    "optax.nadamw" : optax.nadamw,
    "optax.radam" : optax.radam,
    "optax.rmsprop" : optax.rmsprop,
    "optax.rprop" : optax.rprop,
    "optax.sgd" : optax.sgd,
    "DGWO": DGWO
}
activation_fns = {
    "ReLU" : torch.nn.ReLU,
    "LeakyReLU" : torch.nn.LeakyReLU,
    "ReLU6" : torch.nn.ReLU6,
    "RReLU" : torch.nn.RReLU,
    "PReLU" : torch.nn.PReLU,
    "Sigmoid" : torch.nn.Sigmoid,
    "LogSigmoid" : torch.nn.LogSigmoid,
    "Hardsigmoid" : torch.nn.Hardsigmoid,
    "Tanh" : torch.nn.Tanh,
    "Hardtanh" : torch.nn.Hardtanh,
    "SiLU" : torch.nn.SiLU,
    "ELU" : torch.nn.ELU,
    "CELU" : torch.nn.CELU,
    "SELU" : torch.nn.SELU,
    "GLU" : torch.nn.GLU,
    "GELU" : torch.nn.GELU,
    "Mish" : torch.nn.Mish,
    "Hardswish" : torch.nn.Hardswish,
    "Tanhshrink" : torch.nn.Tanhshrink,
    "Hardshrink" : torch.nn.Hardshrink,
    "Softshrink" : torch.nn.Softshrink,
    "Softplus" : torch.nn.Softplus,
    "Softsign" : torch.nn.Softsign
}
loss_fns = {
    "mse" : torch.nn.MSELoss,
    "l1" : torch.nn.L1Loss,
    "huber" : torch.nn.HuberLoss,
    "kldiv" : torch.nn.KLDivLoss,
    "smoothl1" : torch.nn.SmoothL1Loss,
    "bce" : torch.nn.BCELoss,
    "bcelogits" : torch.nn.BCEWithLogitsLoss,
    "nll" : torch.nn.NLLLoss,
    "poissonnll" : torch.nn.PoissonNLLLoss,
    "softmargin" : torch.nn.SoftMarginLoss,
    "crossentropy" : torch.nn.CrossEntropyLoss,
    "cosine" : torch.nn.CosineEmbeddingLoss,
    "ctc" : torch.nn.CTCLoss
}

def get_combo(combo, keys, non_list_values):
    result = {key: value for key, value in zip(keys, combo)}
    result.update(non_list_values)
    return result

def get_model_combinations(config: ModelConfigSearch) -> List[ModelConfig]:
    if config.model_type == "hodl":
        return [ModelConfig(model_type="hodl")]
    if config.model_type == "rl":
        data = config.model_rl
        list_keys = [key for key, value in data.items() if type(value) == ListConfig]
        list_values = [value for value in data.values() if type(value) == ListConfig]
        non_lists = {key: value for key, value in data.items() if type(value) != ListConfig }
        combinations = itertools.product(*list_values)
        return list(map(lambda c: ModelConfig(model_type="rl", model_rl=ModelRLConfig(**get_combo(c, list_keys, non_lists))), combinations))
    if config.model_type == "regression":
        data = config.model_regression
        list_keys = [key for key, value in data.items() if type(value) == ListConfig]
        list_values = [value for value in data.values() if type(value) == ListConfig]
        non_lists = {key: value for key, value in data.items() if type(value) != ListConfig }
        combinations = itertools.product(*list_values)
        return list(map(lambda c: ModelConfig(model_type="regression", model_regression=ModelRegressionConfig(**get_combo(c, list_keys, non_lists))), combinations))
    if config.model_type == "supervised":
        data = config.model_supervised
        list_keys = [key for key, value in data.items() if type(value) == ListConfig]
        list_values = [value for value in data.values() if type(value) == ListConfig]
        non_lists = {key: value for key, value in data.items() if type(value) != ListConfig }
        combinations = itertools.product(*list_values)
        return list(map(lambda c: ModelConfig(model_type="supervised", model_supervised=ModelSupervisedConfig(**get_combo(c, list_keys, non_lists))), combinations))
    elif config.model_type == "technical":
        data = config.model_technical
        list_keys = [key for key, value in data.items() if type(value) == ListConfig]
        list_values = [value for value in data.values() if type(value) == ListConfig]
        non_lists = {key: value for key, value in data.items() if type(value) != ListConfig }
        combinations = itertools.product(*list_values)
        return list(map(lambda c: ModelConfig(model_type="technical", model_technical=ModelTechnicalConfig(**get_combo(c, list_keys, non_lists))), combinations))
    elif config.model_type == "time":
        data = config.model_time
        list_keys = [key for key, value in data.items() if type(value) == ListConfig]
        list_values = [value for value in data.values() if type(value) == ListConfig]
        non_lists = {key: value for key, value in data.items() if type(value) != ListConfig }
        combinations = itertools.product(*list_values)
        return list(map(lambda c: ModelConfig(model_type="time", model_time=ModelTimeConfig(**get_combo(c, list_keys, non_lists))), combinations))
    elif config.model_type == "momentum":
        data = config.model_momentum
        list_keys = [key for key, value in data.items() if type(value) == ListConfig]
        list_values = [value for value in data.values() if type(value) == ListConfig]
        non_lists = {key: value for key, value in data.items() if type(value) != ListConfig }
        combinations = itertools.product(*list_values)
        return list(map(lambda c: ModelConfig(model_type="momentum", model_momentum=ModelMomentumConfig(**get_combo(c, list_keys, non_lists))), combinations))
    elif config.model_type == "weekday":
        data = config.model_day
        list_keys = [key for key, value in data.items() if type(value) == ListConfig]
        list_values = [value for value in data.values() if type(value) == ListConfig]
        non_lists = {key: value for key, value in data.items() if type(value) != ListConfig }
        combinations = itertools.product(*list_values)
        return list(map(lambda c: ModelConfig(model_type="weekday", model_day=ModelDayConfig(**get_combo(c, list_keys, non_lists))), combinations))

    raise ValueError(f'{config.model_type} - model not supported')

def create_model(config: ModelConfig, env: AbstractEnv, device: str):
    if config.model_type == "hodl":
        return HodlModel(config)
    if config.model_type == "rl":
        return create_rl_model(config, env, device)
    if config.model_type == "regression":
        return create_regression_model(config, env, device)
    if config.model_type == "supervised":
        return SupervisedModel(config)
    elif config.model_type == "time":
        return TimeStrategyModel(config)
    elif config.model_type == "technical":
        return TechnicalStrategyModel(config)
    elif config.model_type == "momentum":
        return MomentumStrategyModel(config)
    elif config.model_type == "weekday":
        return DayOfWeekStrategyModel(config)

    raise ValueError(f'{config.model_type} - model not supported')

def create_rl_model(config: ModelConfig, env: AbstractEnv, device: str):
    rl_model = None

    # Direct-RL agents (Moody-Saffell RRL family) aren't SB3 BaseAlgorithms — they train their own way and
    # ARE the AbstractModel, so they're returned directly (not wrapped in RLModel).
    if config.model_rl.model_name == "rrl":
        from src.model.custom.rrl.rrl_model import RRLModel
        return RRLModel(config, env, device)
    if config.model_rl.model_name == "esn-rrl":
        from src.model.custom.rrl.esn_rrl_model import ESNRRLModel
        return ESNRRLModel(config, env, device)

    if config.model_rl.model_name == "ppo":
        print(f"Loading PPO - Proximal Policy Optimization model")
        
        rl_model = PPO(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size, 
                       n_epochs=1,
                       gamma= config.model_rl.gamma, 

                        # gae_lambda: float = 0.95,
                        # clip_range: Union[float, Schedule] = 0.2,
                        # clip_range_vf: Union[None, float, Schedule] = None,
                        # normalize_advantage: bool = True,
                        # ent_coef: float = 0.0,
                        # vf_coef: float = 0.5,
                        # max_grad_norm: float = 0.5,
                        # use_sde: bool = False,
                        # sde_sample_freq: int = -1,
                        # target_kl: Optional[float] = None,

                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                        #    "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                        #    },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch
        # ortho_init: bool = True,
        # use_sde: bool = False,
        # log_std_init: float = 0.0,
        # full_std: bool = True,
        # use_expln: bool = False,
        # squash_output: bool = False,
                       })
    elif config.model_rl.model_name == "ppo-custom":
        print(f"Loading PPO Custom - Proximal Policy Optimization model")
        rl_model = PPO(env=env, policy= CustomActorCriticPolicy, device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size, 
                       n_epochs=1,
                       gamma= config.model_rl.gamma, 
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
    elif config.model_rl.model_name == "reppo":
        print(f"Loading RecurrentPPO - LSTM based PPO model")

        rl_model = RecurrentPPO(env=env, policy= 'MlpLstmPolicy', device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size,
                       n_epochs=1,
                       gamma= config.model_rl.gamma,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "lstm_hidden_size": config.model_rl.lstm_hidden_size,
                           "shared_lstm": config.model_rl.shared_lstm,
                           "enable_critic_lstm": config.model_rl.enable_critic_lstm
                       })
    elif config.model_rl.model_name == "reppo-custom":
        print(f"Loading RecurrentPPO Custom- LSTM based PPO model")

        rl_model = RecurrentPPO(env=env, policy= CustomRecurrentActorCriticPolicy, device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size,
                       n_epochs=1,
                       gamma= config.model_rl.gamma,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch,
                           "lstm_hidden_size": config.model_rl.lstm_hidden_size,
                           "shared_lstm": config.model_rl.shared_lstm,
                           "enable_critic_lstm": config.model_rl.enable_critic_lstm
                       })
    
    elif config.model_rl.model_name in ("attn-ppo", "tcn-ppo", "itransformer-ppo"):
        # PPO over a true SEQUENCE features-extractor (self-attention over bars / TCN / iTransformer
        # inverted-attention over variates) that reshapes the flat obs back to its [lookback, per_bar] bar
        # grid and encodes it — the principled way to add modern sequence architectures (the
        # custom_net_arch attention tokens run on the flat vector, not a real sequence). The extractor
        # takes lookback from the env's data provider.
        encoder = {"attn-ppo": "attn", "tcn-ppo": "tcn", "itransformer-ppo": "itransformer"}[
            config.model_rl.model_name
        ]
        print(f"Loading {config.model_rl.model_name} - PPO with a {encoder} sequence features-extractor")
        rl_model = PPO(env=env, policy='MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size,
                       n_epochs=1,
                       gamma= config.model_rl.gamma,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "features_extractor_class": SequenceFeaturesExtractor,
                           "features_extractor_kwargs": {
                               "lookback": env.data_provider.get_lookback_window(),
                               "encoder": encoder,
                               "features_dim": 64,
                           },
                       })
    elif config.model_rl.model_name == "trpo":
        print(f"Loading TRPO - Trust Region Policy Optimization model")
        rl_model = TRPO(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size, 
                       gamma= config.model_rl.gamma, 
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch
                       })
    
    elif config.model_rl.model_name == "trpo-custom":
        print(f"Loading TRPO Custom- Trust Region Policy Optimization model")

        rl_model = TRPO(env=env, policy= CustomActorCriticPolicy, device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size, 
                       gamma= config.model_rl.gamma, 
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
        # n_steps: int = 2048,
        # batch_size: int = 128,
        # gamma: float = 0.99,
        # cg_max_steps: int = 15,
        # cg_damping: float = 0.1,
        # line_search_shrinking_factor: float = 0.8,
        # line_search_max_iter: int = 10,
        # n_critic_updates: int = 10,
        # gae_lambda: float = 0.95,
        # use_sde: bool = False,
        # sde_sample_freq: int = -1,
        # normalize_advantage: bool = True,
        # target_kl: float = 0.01,
        # sub_sampling_factor: int = 1,
 
    elif config.model_rl.model_name == "ppo-sbx":
        rl_model = sbx.ppo.PPO(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, n_steps=config.model_rl.target_update_interval, batch_size= config.model_rl.batch_size, 
                       gamma= config.model_rl.gamma)
    elif config.model_rl.model_name == "dqn-sbx":
        rl_model = sbx.dqn.DQN(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval,
                       policy_kwargs= {
                           "normalize_images": False,
                        #    "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                        #    "optimizer_kwargs": {
                        #     "eps": config.model_rl.optimizer_eps,
                        #     # "weight_decay": config.model_rl.optimizer_weight_decay,
                        #     #    "alpha": config.model_rl.optimizer_alpha,
                        #     #    "momentum": config.model_rl.optimizer_momentum,
                        #     #    "centered": config.model_rl.optimizer_centered,
                        #    },
                        # #    "activation_fn": activation_fns[config.model_rl.activation_fn], # <-- doesn't yet work properly
                        # #    "net_arch": config.model_rl.net_arch, # <-- doesn't yet work properly
                       })
        
    elif config.model_rl.model_name == "ensemble":
        ensemble = []
        for _ in range(4):
            ensemble.append(DuelingDQN(policy=CustomDuelingDQNPolicy, env=env, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       }))
        rl_model = EnsembleModel(ensemble)

    elif config.model_rl.model_name == "agent57":
        rl_model = Agent57(env=env, policy=CustomDQNPolicy, learning_rate= config.model_rl.learning_rate, buffer_size= config.model_rl.buffer_size, learning_starts= config.model_rl.learning_starts, target_update_interval= config.model_rl.target_update_interval,
            policy_kwargs= {
                "normalize_images": False,
                "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                "optimizer_kwargs": {
                #    "eps": config.model_rl.optimizer_eps,
                #    "weight_decay": config.model_rl.optimizer_weight_decay,
                #    "alpha": config.model_rl.optimizer_alpha,
                #    "momentum": config.model_rl.optimizer_momentum,
                #    "centered": config.model_rl.optimizer_centered,
                },
                "activation_fn": activation_fns[config.model_rl.activation_fn],
                "net_arch": config.model_rl.net_arch,
                "custom_net_arch": config.model_rl.custom_net_arch
        })
    elif config.model_rl.model_name == "iqn":
        rl_model = IQN(env=env, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                       })
    elif config.model_rl.model_name == "iqn-custom":
        rl_model = IQN(env=env, policy= CustomIQNPolicy, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                        buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                        tau= config.model_rl.tau, 
                        exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                        learning_starts=config.model_rl.learning_starts,
                        train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                        target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                        policy_kwargs= {
                            "normalize_images": False,
                            "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                            "optimizer_kwargs": {
                            },
                            "activation_fn": activation_fns[config.model_rl.activation_fn],
                            "net_arch": config.model_rl.net_arch,
                            "custom_net_arch": config.model_rl.custom_net_arch
                        })
    elif config.model_rl.model_name == "duel-dqn":
        rl_model = DuelingDQN(env=env, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                        buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                        tau= config.model_rl.tau, 
                        exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                        learning_starts=config.model_rl.learning_starts,
                        train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                        target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                        policy_kwargs= {
                            "normalize_images": False,
                            "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                            "optimizer_kwargs": {
                            },
                            "activation_fn": activation_fns[config.model_rl.activation_fn],
                            "net_arch": config.model_rl.net_arch
                        })
    elif config.model_rl.model_name == "duel-dqn-custom":
        rl_model = DuelingDQN(policy=CustomDuelingDQNPolicy, env=env, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
    elif config.model_rl.model_name == "duel-dqn-custom-lstm":
        rl_model = DuelingDQN(policy=CustomDuelingDQNPolicy, env=env, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch,
                           "features_extractor_class": LSTMFCE,
                           "features_extractor_kwargs": {
                               "lstm_hidden_size": 2
                           }
                       })
    elif config.model_rl.model_name == "duel-dqn-custom-lstm3":
        rl_model = DuelingDQN(policy=CustomDuelingDQNPolicy, env=env, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch,
                           "features_extractor_class": LSTMFCE,
                           "features_extractor_kwargs": {
                               "lstm_hidden_size": 3
                           }
                       })
    elif config.model_rl.model_name == "duel-dqn-lstm":
        rl_model = DuelingDQN(env=env, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "features_extractor_class": LSTMFCE,
                           "features_extractor_kwargs": {
                               "lstm_hidden_size": 4
                           }
                       }
                       )
    elif config.model_rl.model_name == "rainbow-dqn":
        print(f"Loading Rainbow DQN - Rainbow Deep Q Network model")

        rl_model = RainbowDQN(env=env, device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                       })
    elif config.model_rl.model_name == "rainbow-dqn-custom":
        print(f"Loading Rainbow DQN - Rainbow Deep Q Network model")

        rl_model = RainbowDQN(env=env, policy=CustomRainbowPolicy, device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
    elif config.model_rl.model_name == "munchausen-dqn":
        print(f"Loading MunchausenDQN DQN - Munchhausen Deep Q Network model")

        rl_model = MunchausenDQN(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                       }
                    )
    elif config.model_rl.model_name == "munchausen-dqn-custom":
        print(f"Loading MunchausenDQN Custom - Munchhausen Deep Q Network model")
        rl_model = MunchausenDQN(env=env, policy=CustomDQNPolicy, device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
    elif config.model_rl.model_name == "munchausen-duel-dqn-custom":
        print(f"Loading MunchausenDQN Custom - Munchhausen Deep Q Network model")
        rl_model = MunchausenDQN(env=env, policy=CustomDuelingDQNPolicy, device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
    elif config.model_rl.model_name == "munchausen-duel-dqn-custom-lstm":
        print(f"Loading MunchausenDQN Custom - Munchhausen Deep Q Network model")
        rl_model = MunchausenDQN(env=env, policy=CustomDuelingDQNPolicy, device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch,
                           "features_extractor_class": LSTMFCE,
                           "features_extractor_kwargs": {
                               "lstm_hidden_size": 2
                           }
                       })
    elif config.model_rl.model_name == "dqn":
        print(f"Loading DQN - Deep Q Network model")

        rl_model = DQN(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                       }
                       )
        
        # optimizer_weight_decay=0,
        # alpha=0.75,

        # lambd=1e-4,
        # t0=1e6,
        # foreach: Optional[bool] = None,
        # maximize: bool = False,
        # differentiable: bool = False,

    elif config.model_rl.model_name in ("dqn-custom", "tdqn"):
        # 'tdqn' = the Théate & Ernst (2021) Trading-DQN: mechanically the custom-head double-DQN; its
        # faithful identity (a deep net, low gamma, long/short) is carried by the launch config / the
        # paper's replicateConfig, per the "TDQN as a faithful dqn-custom config" decision.
        rl_model = DQN(env=env, policy= CustomDQNPolicy, device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size,
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
    elif config.model_rl.model_name == "dqn-lstm":
        print(f"Loading DQN - Deep Q Network model")
        rl_model = DQN(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "features_extractor_class": LSTMFCE,
                           "features_extractor_kwargs": {
                               "lstm_hidden_size": 4
                           }
                       })
    elif config.model_rl.model_name == "qrdqn":
        print(f"Loading QRDQN - Quantile Regression DQN model")

        rl_model = QRDQN(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                       }
                    )
    elif config.model_rl.model_name == "qrdqn-custom":
        print(f"Loading QRDQN Custom - Quantile Regression DQN model")

        rl_model = QRDQN(env=env, policy= CustomQRDQNPolicy, device= device, learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                       buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                       tau= config.model_rl.tau, 
                       exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                       learning_starts=config.model_rl.learning_starts,
                       train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                       target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       }
                    )
    elif config.model_rl.model_name == "a2c":
        print(f"Loading A2C - Asynchronous Advantage Actor-Critic Algorithm model")
        # rl_model = A2C(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, n_steps=1) #, n_steps= is like a batch size
    
        rl_model = A2C(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, n_steps= config.model_rl.batch_size, 
                       gamma= config.model_rl.gamma, 
                       max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                       }
                    )
    elif config.model_rl.model_name == "a2c-custom":
        print(f"Loading A2C Custom - Asynchronous Advantage Actor-Critic Algorithm model")
        rl_model = A2C(env=env, policy= CustomActorCriticPolicy, device= device, learning_rate= config.model_rl.learning_rate, n_steps= config.model_rl.batch_size, 
                       gamma= config.model_rl.gamma, 
                       max_grad_norm=config.model_rl.max_grad_norm,
                       policy_kwargs= {
                           "normalize_images": False,
                           "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                           "optimizer_kwargs": {
                            #    "eps": config.model_rl.optimizer_eps,
                            #    "weight_decay": config.model_rl.optimizer_weight_decay,
                            #    "alpha": config.model_rl.optimizer_alpha,
                            #    "momentum": config.model_rl.optimizer_momentum,
                            #    "centered": config.model_rl.optimizer_centered,
                           },
                           "activation_fn": activation_fns[config.model_rl.activation_fn],
                           "net_arch": config.model_rl.net_arch,
                           "custom_net_arch": config.model_rl.custom_net_arch
                       })
    elif config.model_rl.model_name == "ars":
        print(f"Loading ARS - Augmented Random Search model")
        # rl_model = ARS(env=env, policy= 'LinearPolicy', device= device, learning_rate= config.model_rl.learning_rate)
        rl_model = ARS(env=env, policy= 'LinearPolicy', device= device, learning_rate= config.model_rl.learning_rate, 
                       policy_kwargs= {
                        #    "normalize_images": False,
                        #    "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                        #    "activation_fn": activation_fns[config.model_rl.activation_fn],
                        #    "net_arch": config.model_rl.net_arch,
                       }
                    )
    elif config.model_rl.model_name == "ars-mlp":
        print(f"Loading ARS - Augmented Random Search model with MLP Policy")
        # rl_model = ARS(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate)
        rl_model = ARS(env=env, policy= 'MlpPolicy', device= device, learning_rate= config.model_rl.learning_rate, 
                       policy_kwargs= {
                        #    "normalize_images": False,
                        #    "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                        #    "activation_fn": activation_fns[config.model_rl.activation_fn],
                        #    "net_arch": config.model_rl.net_arch,
                       }
                    )
        
    if rl_model is not None:
        if config.model_rl.checkpoint_to_load is not None:
            path = os.path.join(config.model_rl.checkpoints_folder, config.model_rl.checkpoint_to_load)
            rl_model = rl_model.load(path)
            
        rl_model.set_logger(Logger(
            folder=None,
            output_formats=[HumanOutputFormat(sys.stdout)],
        ))
        if config.model_rl.seed is not None and hasattr(rl_model, "set_random_seed"):
            rl_model.set_random_seed(config.model_rl.seed)
        return RLModel(config, rl_model)
    
    raise ValueError(f'{config.model_rl.model_name} - rl model not supported')

def _resolve_pos_weight(reg, env, device):
    """The BCE positive-class weight: an explicit ``pos_weight`` (>0) wins, else the
    training class ratio n_neg/n_pos (auto-balance when 0/unset). None → unweighted."""
    configured = getattr(reg, "pos_weight", 0.0) or 0.0
    if configured > 0:
        return torch.tensor([float(configured)], device=device)
    n_pos = getattr(env, "n_positive", 0)
    n_neg = getattr(env, "n_negative", 0)
    if n_pos > 0 and n_neg > 0:
        return torch.tensor([n_neg / n_pos], device=device)
    return None


def create_regression_model(config: ModelConfig, env: AbstractEnv, device: str):
    regression_model = None

    if config.model_regression.model_name == "mlp":
        print(f"Loading MLP model")

        if config.model_regression.seed is not None:
            torch.manual_seed(config.model_regression.seed)

        features_dim = len(env.last_obs)
        action_dim = 1
        # action_dim = int(env.action_space.n)
        mlp = CustomQNetwork.create_mlp_custom(features_dim, action_dim, config.model_regression.net_arch, activation_fns[config.model_regression.activation_fn], config.model_regression.custom_net_arch)
        # mlp.append(torch.nn.Sigmoid())
        regression_model = torch.nn.Sequential(*mlp).to(device)

        # regression_model

                    #    buffer_size= config.model_rl.buffer_size, 
                    # 
                    #   gamma= config.model_rl.gamma, 
                    #    tau= config.model_rl.tau, 
                    #    exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                    #    learning_starts=config.model_rl.learning_starts,
                    #    train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                    #    target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,

    if regression_model is not None:
        if config.model_regression.checkpoint_to_load is not None:
            path = os.path.join(config.model_regression.checkpoints_folder, config.model_regression.checkpoint_to_load)
            regression_model.load_state_dict(torch.load(path))
            
        optimizer = optimizer_classes[config.model_regression.optimizer_class](regression_model.parameters(), lr=config.model_regression.learning_rate)
        loss_kwargs = {"reduction": config.model_regression.loss_fn_reduction}
        if config.model_regression.loss_fn == "bcelogits":
            pos_weight = _resolve_pos_weight(config.model_regression, env, device)
            if pos_weight is not None:
                loss_kwargs["pos_weight"] = pos_weight
        loss_fn = loss_fns[config.model_regression.loss_fn](**loss_kwargs)
        return RegressionModel(config, regression_model, optimizer, loss_fn)

    
    raise ValueError(f'{config.model_regression.model_name} - regression model not supported')