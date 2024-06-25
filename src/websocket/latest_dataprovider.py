import pandas as pd
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider

class LatestDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig):
        super(LatestDataProvider, self).__init__(config)

        self.latest_price = 1 
        self.latest_signal = [1,1,1,1] # We need to determine this beforehand
        self.raw_signals = []

    def get_timesteps(self) -> int:
        return 0
    
    def get_price(self, step: int) -> float:
        return self.latest_price

    def get_buy_sell_signal(self, step: int) -> int:
        return 0
    
    def get_values(self, step: int):
        print("get_values step: ", step)
        return self.latest_signal

    def store_price(self, price):
        self.latest_price = float(price)

    def store_klines(self, data):
        data["indicators"] = { "test": 0 }
        self.raw_signals.append(data)

        if len(self.raw_signals) > 2:
            self.raw_signals.pop(0)

        if len(self.raw_signals) > 1:
            df = pd.DataFrame(self.raw_signals)
            df = df.drop(columns=["tokenPair", "interval"])

            processed_df, _, _, _ = self.process_df(df, self.config.type, self.config.timestamp)
            self.latest_signal = processed_df.loc[len(processed_df) - 1].values
            return True
        return False
