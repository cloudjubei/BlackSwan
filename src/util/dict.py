from omegaconf import DictConfig
from flatten_dict import flatten


def dict_config_to_dict(config: DictConfig) -> dict:
    return {str(key): value for key, value in zip(config.keys(), config.values())}


def dict_config_to_params(config: DictConfig, prefix: str):
    params = flatten(dict_config_to_dict(config), reducer="dot")
    with_prefix = {prefix + str(key): val for key, val in params.items()}
    return with_prefix
