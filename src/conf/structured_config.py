from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from typing import List
from src.conf.data_config import DataConfig, get_datas_1h_1d, get_datas_1m_1h_1d
from src.conf.model_config import ModelConfigSearch, get_models_all, get_models_simple, get_models_rl
from src.conf.env_config import EnvConfig, get_envs_all, get_envs_simple, get_envs_swaps

@dataclass
class Repo:
    repo_owner: str = "cloudjubei"
    repo_name: str = "black-swan"

@dataclass
class Config:
    user: str = "cloudjubei"
    experiment_name: str = "hiperparams"
    run_name: str = "test"
    device: str = "auto" #"mps"

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
cs.store(name="config_simple", node=Config(run_name= "test-simple_rl_vlong", data_configs= get_datas_1h_1d(), model_configs=get_models_rl(), env_configs=get_envs_simple()))
