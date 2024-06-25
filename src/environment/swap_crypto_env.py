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
    
    def get_mapped_action(self, action: int | None) -> int | None:
        return action

    def take_action(self, action) -> bool:
        
        if action != 0:  # Swap
            if self.position > 0:
                amount = self.position * self.current_price
                fee = amount * self.transaction_fee_multiplier
                self.balance = amount - fee
                self.position = 0
                self.last_transaction_fee = fee
                return True
            else:
                amount = self.balance / self.current_price
                fee = amount * self.transaction_fee_multiplier
                self.position = amount - fee
                self.balance = 0
                self.last_transaction_fee = fee * self.current_price
                return True
        self.last_transaction_fee = 0
        return False