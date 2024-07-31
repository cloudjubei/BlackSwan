from src.environment.base_crypto_env import BaseCryptoEnv
from src.conf.env_config import EnvConfig
from gymnasium import spaces
import pandas as pd
import numpy as np

class ServerCryptoEnv(BaseCryptoEnv):
    
    def create_observation(self, signals_data) -> np.ndarray:

        values = self.data_provider.prepare_signal(signals_data)
        if values:
            return self.get_observation_buy(values)

    def get_observation_buy(self, values: np.ndarray) -> np.ndarray:
        values = self.data_provider.get_values(self.current_step)
        return self.get_observation(values, np.zeros_like(values), np.zeros_like(values), np.zeros_like(values), np.zeros_like(values))
    
    def get_observation(self, values: np.ndarray, percent_profits: np.ndarray, stoploss_closeness: np.ndarray, drawdowns: np.ndarray, positions: np.ndarray) -> np.ndarray:
        
        out = []
        for i in range(len(values)):
            out = out + values[i] + [percent_profits[i], stoploss_closeness[i], drawdowns[i], positions[i]]
        return out
        