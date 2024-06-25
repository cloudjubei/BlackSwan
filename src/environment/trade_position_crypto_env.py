from src.environment.base_crypto_env import BaseCryptoEnv
from src.conf.env_config import EnvConfig
from gymnasium import spaces
import pandas as pd
import numpy as np

class TradePositionCryptoEnv(BaseCryptoEnv):
    
    def __init__(self, env_config: EnvConfig, df: pd.DataFrame, data_has_no_price: bool):
        super(TradePositionCryptoEnv, self).__init__(env_config, df, data_has_no_price)
        
        self.position_to_trade = max(0.0, self.env_config.amount)

    def take_action(self, action) -> bool:
        if action == 1: # Buy
            if self.balance > 0:
                #TODO: incldue fee
                balance_to_trade = min(self.balance, self.position_to_trade * self.current_price)
                self.position += balance_to_trade / self.current_price
                self.balance = max(0, self.balance - balance_to_trade)
                return True
        elif action == 2: # Sell
            if self.position > 0:
                #TODO: incldue fee
                position_to_trade = min(self.position, self.position_to_trade)
                self.balance += position_to_trade * self.current_price
                self.position = max(0, self.position - position_to_trade)
                return True
        return False