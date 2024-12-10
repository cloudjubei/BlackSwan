from src.environment.base_crypto_env import BaseCryptoEnv
from gymnasium import spaces
import pandas as pd
import numpy as np

class SwapCryptoEnv(BaseCryptoEnv):
    """
    Uses 2 actions.
    """

    # Action space: 0 = Hold, Other = Swap
    def create_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(2)

    def take_action(self, action) -> bool:
        
        if action != 0:  # Swap
            position = self.positions[-1]
            if position > 0:
                amount = position * self.current_price
                fee = amount * self.transaction_fee_multiplier
                self.balances[-1] = amount - fee
                self.positions[-1] = 0
                self.fees.append(fee)
                return True
            
            amount = self.balances[-1] / self.current_price
            fee = amount * self.transaction_fee_multiplier
            self.positions[-1] = amount - fee
            self.balances[-1] = 0
            self.fees.append(fee * self.current_price)
            return True
        
        return False
    