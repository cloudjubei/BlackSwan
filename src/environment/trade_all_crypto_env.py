from src.environment.base_crypto_env import BaseCryptoEnv
from src.conf.env_config import EnvConfig
from gymnasium import spaces
import pandas as pd
import numpy as np

class TradeAllCryptoEnv(BaseCryptoEnv):
    
    # Action space: 0 = Hold, 1 = Buy, 2 = Sell (BEWARE: by default the RL models return actions as [0..<n])
    def create_action_space(self) -> spaces.Discrete:
        if self.env_config.no_sell_action:
            return spaces.Discrete(2)
        return spaces.Discrete(3)
    
    def take_action(self, action) -> bool:
        if action == 1: # Buy
            balance = self.balances[-1]
            if balance > 0:
                amount = balance / self.current_price
                fee = amount * self.transaction_fee_multiplier
                self.positions[-1] = amount - fee
                self.balances[-1] = 0
                self.fees.append(fee * self.current_price)
                return True
        elif action == 2: # Sell
            position = self.positions[-1]
            if position > 0:
                amount = position * self.current_price
                fee = amount * self.transaction_fee_multiplier
                self.positions[-1] = 0
                self.balances[-1] = amount - fee
                self.fees.append(fee)
                return True
            
        return False