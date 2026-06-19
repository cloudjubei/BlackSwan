import json
import pandas as pd
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider
from src.server.cache.ACache import PriceCache

class ServerDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig):
        super(ServerDataProvider, self).__init__(config)

        self.values = []
        self.prices = []
        self.timestamps = []

    def prepare_signal(self, signal_datas):

        values = []
        prices = []
        timestamps = []
        for signal_data in signal_datas:
            signal_json = json.dumps(signal_data)
            raw_df = pd.read_json(signal_json)

            df, ps, ts, buy_sells, rewards_buy_profitable, rewards_buy_drawdown = self.process_df(raw_df, self.config.type, self.config.timestamp, self.config.indicator, self.config.buyreward_percent, self.config.buyreward_maxwait)

            values.append(df.values)
            prices.append(ps)
            timestamps.append(ts)

        self.values = values
        self.prices = prices
        self.timestamps = timestamps

        return values

    def get_timesteps(self) -> int:
        return len(self.values)

    def get_price(self, step: int) -> float:
        return self.prices[step][-1]

    def get_signal_buy_sell(self, step: int) -> int:
        return 0

    def get_signal_buy_profitable(self, step: int) -> int:
        return 0

    def get_signal_buy_drawdown(self, step: int) -> int:
        return 0

    def get_values(self, step: int):
        return self.values
