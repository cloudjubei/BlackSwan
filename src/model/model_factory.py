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
from src.conf.model_config import ModelConfigSearch, ModelConfig, ModelTechnicalConfig, ModelTimeConfig, ModelRLConfig, ModelLSTMConfig, ModelMLPConfig
from src.model.custom.agent57.agent57 import Agent57
from src.model.custom.customqnetwork import CustomQNetwork
from src.model.custom.dgwo import DGWO
from src.model.custom.ensemble.ensemble import EnsembleModel
from src.model.custom.policies import CustomActorCriticPolicy, CustomDQNPolicy, CustomDuelingDQNPolicy, CustomQRDQNPolicy, CustomRainbowPolicy, CustomRecurrentActorCriticPolicy
from src.model.custom.policy_iqn import CustomIQNPolicy
from src.model.dqn_lstm_policy import LSTMFCE
from src.model.dueling_dqn.dueling_dqn import DuelingDQN
from src.model.dueling_dqn.policies import DuelingDQNPolicy
from src.model.hodl_model import HodlModel
from src.model.iqn.iqn import IQN
from src.model.munchhausen_dqn.munchhausen_dqn import MunchausenDQN
from src.model.rainbow.rainbow_dqn_agent import RainbowDQNAgent
from src.model.rainbow_dqn.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.model.rainbow_dqn.rainbow_dqn import RainbowDQN
from src.model.regression_model import RegressionModel
from src.model.rl_model import RLModel
from src.model.time_strategy_model import TimeStrategyModel
from src.model.technical_strategy_model import TechnicalStrategyModel
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
    
    raise ValueError(f'{config.model_type} - model not supported')

def create_model(config: ModelConfig, env: AbstractEnv, device: str):
    if config.model_type == "hodl":
        return HodlModel(config)
    if config.model_type == "rl":
        return create_rl_model(config, env, device)
    if config.model_type == "regression":
        return create_regression_model(config, env, device)
    elif config.model_type == "time":
        return TimeStrategyModel(config)
    elif config.model_type == "technical":
        return TechnicalStrategyModel(config)
    
    raise ValueError(f'{config.model_type} - model not supported')

def create_rl_model(config: ModelConfig, env: AbstractEnv, device: str):
    rl_model = None

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
                           "net_arch": config.model_rl.net_arch
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
                           "custom_net_arch": config.model_rl.custom_net_arch
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
    elif config.model_rl.model_name == "rainbow-dqn-old":
        seed = 777
        rl_model = RainbowDQNAgent(env, memory_size= config.model_rl.buffer_size, batch_size= config.model_rl.batch_size, target_update=config.model_rl.target_update_interval, seed= seed)
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
    elif config.model_rl.model_name == "duel-dqn-custom-lstm8":
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
                           "custom_net_arch": config.model_rl.custom_net_arch,
                           "features_extractor_class": LSTMFCE,
                           "features_extractor_kwargs": {
                               "lstm_hidden_size": 8
                           }
                       })
    elif config.model_rl.model_name == "duel-dqn-custom-lstm8":
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
                           "custom_net_arch": config.model_rl.custom_net_arch,
                           "features_extractor_class": LSTMFCE,
                           "features_extractor_kwargs": {
                               "lstm_hidden_size": 8
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

    elif config.model_rl.model_name == "dqn-custom":
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
        return RLModel(config, rl_model)
    
    raise ValueError(f'{config.model_rl.model_name} - rl model not supported')

def create_regression_model(config: ModelConfig, env: AbstractEnv, device: str):
    regression_model = None

    if config.model_regression.model_name == "mlp":
        print(f"Loading MLP model")
        print(f'env last obs.shape: {env.last_obs.shape[0]} env.action_space.shape: {env.action_space.shape[0]}')

        # features_dim = len(env.last_obs)
        features_dim = env.last_obs.shape[0]
        # features_dim = spaces.utils.flatdim(observation_space)
        action_dim = env.action_space.shape[0]
        mlp = CustomQNetwork.create_mlp_custom(features_dim, action_dim, config.model_rl.net_arch, activation_fns[config.model_rl.activation_fn], config.model_rl.custom_net_arch)
        regression_model = torch.nn.Sequential(*mlp)

                    #    learning_rate= config.model_rl.learning_rate, batch_size= config.model_rl.batch_size, 
                    #    buffer_size= config.model_rl.buffer_size, gamma= config.model_rl.gamma, 
                    #    tau= config.model_rl.tau, 
                    #    exploration_final_eps=config.model_rl.exploration_final_eps, exploration_fraction=config.model_rl.exploration_fraction,
                    #    learning_starts=config.model_rl.learning_starts,
                    #    train_freq=config.model_rl.train_freq, gradient_steps=config.model_rl.gradient_steps,
                    #    target_update_interval=config.model_rl.target_update_interval, max_grad_norm=config.model_rl.max_grad_norm,
                    #    policy_kwargs= {
                    #        "optimizer_class": optimizer_classes[config.model_rl.optimizer_class],
                    #    })

    if regression_model is not None:
    #     if config.model_regression.checkpoint_to_load is not None:
    #         path = os.path.join(config.model_regression.checkpoints_folder, config.model_regression.checkpoint_to_load)
    #         regression_model = regression_model.load(path)
            
    #     regression_model.set_logger(Logger(
    #         folder=None,
    #         output_formats=[HumanOutputFormat(sys.stdout)],
    #     ))
        return RegressionModel(config, regression_model)
    
    raise ValueError(f'{config.model_regression.model_name} - regression model not supported')