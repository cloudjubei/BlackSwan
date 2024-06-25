from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from typing import List
from src.conf.data_config import DataConfig, get_datas_simple, get_datas_long, get_datas_vlong, get_datas_downtrend, get_datas_min, get_datas_short_min, get_datas_long_min, get_datas_vlong_min, get_datas_min_hour
from src.conf.model_config import ModelConfigSearch, get_models_all, get_models_simple, get_models_rl_h, get_models_rl_min
from src.conf.env_config import EnvConfig, get_envs_all, get_envs_simple, get_envs_swaps, get_envs_simple2

@dataclass
class Repo:
    repo_owner: str = "hamsterkmak"
    repo_name: str = "black-swan-experiments"

@dataclass
class Config:
    user: str = "cloudjubei"
    experiment_name: str = "hiperparams"
    run_name: str = "test"
    device: str = "cpu" #"mps"

    data_configs: List[DataConfig] = field(default_factory=List)
    env_configs: List[EnvConfig] = field(default_factory=List)
    model_configs: List[ModelConfigSearch] = field(default_factory=List)

    dagshub_repo: Repo = field(default_factory=Repo)
    local_only: bool = True
    show_render: bool = True

# # can be created from json
# json_config_simple = {
#     "run_name": "test-simple",
#     "data_configs": get_datas_simple(),
#     "model_configs": get_models_simple(),
#     "env_configs": get_envs_simple(),
# }
# oconfig_simple = OmegaConf.create(json_config_simple)
# config_simple = Config(**oconfig_simple)

# or directly via class
    
cs = ConfigStore.instance()
cs.store(name="config_simple", node=Config(run_name= "test-simple", data_configs= get_datas_simple(), model_configs=get_models_simple(), env_configs=get_envs_simple()))
cs.store(name="config_simple_all", node=Config(run_name= "test-simple_all", data_configs= get_datas_simple(), model_configs=get_models_all(), env_configs=get_envs_simple()))
cs.store(name="config_simple_all_long", node=Config(run_name= "test-simple_all_long", data_configs= get_datas_long(), model_configs=get_models_all(), env_configs=get_envs_simple()))
cs.store(name="config_simple_all_vlong", node=Config(run_name= "test-simple_all_vlong", data_configs= get_datas_vlong(), model_configs=get_models_all(), env_configs=get_envs_simple()))
cs.store(name="config_simple_rl", node=Config(run_name= "test-simple_rl", data_configs= get_datas_simple(), model_configs=get_models_rl_h(), env_configs=get_envs_simple()))
cs.store(name="config_simple_rl_long", node=Config(run_name= "test-simple_rl_long", data_configs= get_datas_long(), model_configs=get_models_rl_h(), env_configs=get_envs_simple()))
cs.store(name="config_simple_rl_vlong", node=Config(run_name= "test-simple_rl_vlong", data_configs= get_datas_vlong(), model_configs=get_models_rl_h(), env_configs=get_envs_simple()))

# cs.store(name="config_simple_rl_downtrend", node=Config(run_name= "test-simple_rl_downtrend", data_configs= get_datas_downtrend(), model_configs=get_models_rl(), env_configs=get_envs_simple()))
# cs.store(name="config_simple_rl_swaps", node=Config(run_name= "test-simple_rl_swaps", data_configs= get_datas_simple(), model_configs=get_models_rl(), env_configs=get_envs_swaps()))
# cs.store(name="config_simple_rl_swaps_long", node=Config(run_name= "test-simple_rl_swaps_long", data_configs= get_datas_long(), model_configs=get_models_rl(), env_configs=get_envs_swaps()))

cs.store(name="config_simple_min", node=Config(run_name= "test-simple_min", data_configs= get_datas_min(), model_configs=get_models_simple(), env_configs=get_envs_simple()))
cs.store(name="config_simple_long_min", node=Config(run_name= "test-simple_long_min", data_configs= get_datas_long_min(), model_configs=get_models_simple(), env_configs=get_envs_simple()))
cs.store(name="config_simple_vlong_min", node=Config(run_name= "test-simple_vlong_min", data_configs= get_datas_vlong_min(), model_configs=get_models_simple(), env_configs=get_envs_simple()))
cs.store(name="config_simple_short_rl_min", node=Config(run_name= "test-simple_rl_min", data_configs= get_datas_short_min(), model_configs=get_models_rl_min(), env_configs=get_envs_simple()))
cs.store(name="config_simple_rl_min", node=Config(run_name= "test-simple_rl_min", data_configs= get_datas_min(), model_configs=get_models_rl_min(), env_configs=get_envs_simple()))
cs.store(name="config_simple_long_rl_min", node=Config(run_name= "test-simple_long_rl_min", data_configs= get_datas_long_min(), model_configs=get_models_rl_min(), env_configs=get_envs_simple()))
cs.store(name="config_simple_vlong_rl_min", node=Config(run_name= "test-simple_vlong_rl_min", data_configs= get_datas_vlong_min(), model_configs=get_models_rl_min(), env_configs=get_envs_simple()))

# cs.store(name="config_simple_envs2", node=Config(run_name= "test-simple", data_configs= get_datas_simple(), model_configs=get_models_simple(), env_configs=get_envs_simple2()))
# cs.store(name="config_simple_long", node=Config(run_name= "test-simple-long", data_configs= get_datas_long(), model_configs=get_models_simple(), env_configs=get_envs_simple()))
# cs.store(name="config_simple_long_envs2", node=Config(run_name= "test-simple-long", data_configs= get_datas_long(), model_configs=get_models_simple(), env_configs=get_envs_simple2()))
# cs.store(name="config_simple_vlong", node=Config(run_name= "test-simple-vlong", data_configs= get_datas_vlong(), model_configs=get_models_simple(), env_configs=get_envs_simple()))
# cs.store(name="config_simple_downtrend", node=Config(run_name= "test-simple-downtrend", data_configs= get_datas_downtrend(), model_configs=get_models_simple(), env_configs=get_envs_simple()))
# cs.store(name="config_swaps", node=Config(run_name= "test-swaps", data_configs= get_datas_simple(), model_configs=get_models_all(), env_configs=get_envs_swaps()))

cs.store(name="config_simple_min_hour", node=Config(run_name= "test-simple_min_hour", data_configs= get_datas_min_hour(), model_configs=get_models_rl_min(), env_configs=get_envs_simple()))
