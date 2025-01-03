from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider
from src.environment.dip_predict_env import DipPredictEnv
from src.environment.trend_predict_env import TrendPredictEnv
from .swap_crypto_env import SwapCryptoEnv
from .trade_all_crypto_env import TradeAllCryptoEnv
from src.environment.abstract_env import AbstractEnv
from src.conf.env_config import EnvConfig
import pandas as pd

def create_environment(config: EnvConfig, data_provider: AbstractDataProvider, device: str) -> AbstractEnv:
    if config.type == "trade_all":
        return TradeAllCryptoEnv(config, data_provider, device)
    elif config.type == "swap":
        return SwapCryptoEnv(config, data_provider, device)
    elif config.type == "trend_predict":
        return TrendPredictEnv(config, data_provider, device)
    elif config.type == "dip_predict":
        return DipPredictEnv(config, data_provider, device)
    raise ValueError("{config.type} - env not supported")
