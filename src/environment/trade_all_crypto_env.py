from src.environment.base_crypto_env import BaseCryptoEnv
from src.conf.env_config import EnvConfig
from gymnasium import spaces
import pandas as pd
import numpy as np

class TradeAllCryptoEnv(BaseCryptoEnv):
    
    # Action space: 0 = Hold, 1 = Buy(long), 2 = Sell/close-long. With allow_shorting, 3 = Short,
    # 4 = Cover. (BEWARE: by default the RL models return actions as [0..<n])
    def create_action_space(self) -> spaces.Discrete:
        if self.env_config.allow_shorting:
            return spaces.Discrete(5)
        if self.env_config.no_sell_action:
            return spaces.Discrete(2)
        return spaces.Discrete(3)

    def take_action(self, action) -> bool:
        position = self.positions[-1]
        if action == 1 and position == 0:  # open long
            balance = self.balances[-1]
            if balance > 0:
                deployed = balance * self._position_size()
                amount = deployed / self.current_price
                fee = amount * self.transaction_fee_multiplier
                self.positions[-1] = amount - fee
                self.balances[-1] = balance - deployed
                self.fees.append(fee * self.current_price)
                return True
        elif action == 2 and position > 0:  # close long
            amount = position * self.current_price
            fee = amount * self.transaction_fee_multiplier
            self.positions[-1] = 0
            self.balances[-1] = self.balances[-1] + amount - fee
            self.fees.append(fee)
            return True
        elif action == 3 and position == 0 and self.env_config.allow_shorting:  # open short
            balance = self.balances[-1]
            if balance > 0:
                size = min(self._position_size(), float(self.env_config.max_short_size))
                notional = balance * size
                amount = notional / self.current_price
                fee = notional * self.transaction_fee_multiplier
                self.positions[-1] = -amount
                self.balances[-1] = balance + notional - fee
                self.fees.append(fee)
                return True
        elif action == 4 and position < 0:  # cover short
            amount = -position
            cost = amount * self.current_price
            fee = cost * self.transaction_fee_multiplier
            self.positions[-1] = 0
            self.balances[-1] = self.balances[-1] - cost - fee
            self.fees.append(fee)
            return True

        return False