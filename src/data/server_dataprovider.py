import json
import pandas as pd
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider
from src.server.cache.ACache import PriceCache

class ServerDataProvider(AbstractDataProvider):
    def __init__(self, config: DataConfig):
        super(ServerDataProvider, self).__init__(config)

    def prepare_signal(self, signal_datas):

        values = []
        for signal_data in signal_datas:
            signal_json = json.dumps(signal_data)  
            raw_df = pd.read_json(signal_json)
      
            df, prices, timestamps, buy_sells, rewards_buy_profitable, rewards_buy_drawdown = self.process_df(raw_df, self.config.type, self.config.timestamp, self.config.indicator, self.config.buyreward_percent, self.config.buyreward_maxwait)
      
            values.append(df.values)

        return values
