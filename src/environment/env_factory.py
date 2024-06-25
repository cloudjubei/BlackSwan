from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider
from .swap_crypto_env import SwapCryptoEnv
from .trade_all_crypto_env import TradeAllCryptoEnv
from .trade_percent_crypto_env import TradePercentCryptoEnv
from .trade_position_crypto_env import TradePositionCryptoEnv
from .trade_amount_crypto_env import TradeAmountCryptoEnv
from src.environment.abstract_env import AbstractEnv
from src.conf.env_config import EnvConfig
import pandas as pd

def create_environment(config: EnvConfig, data_provider: AbstractDataProvider, device: str) -> AbstractEnv:
    # if config.type == "swap":
    #     return SwapCryptoEnv(config, data_provider, device)
    # elif config.type == "trade_all":
    if config.type == "trade_all":
        return TradeAllCryptoEnv(config, data_provider, device)
    # elif config.type == "trade_percent":
    #     return TradePercentCryptoEnv(config, data_provider, device)
    # elif config.type == "trade_position":
    #     return TradePositionCryptoEnv(config, data_provider, device)
    # elif config.type == "trade_amount":
    #     return TradeAmountCryptoEnv(config, data_provider, device)
    raise ValueError("{config.type} - env not supported")
