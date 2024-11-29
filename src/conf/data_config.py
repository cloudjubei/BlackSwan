from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    id: str
    train_data_paths: List[List[str]]
    test_data_paths: List[List[str]]
    lookback_window_size: int = 1
    type: str = "only_price_percent" # possible ["standard", "solo_price_percent", "only_price_percent", "all_percents"]
    indicator: str = "none"
    timestamp: str = "day_of_week" # possible ["none",  "expanded", "day_of_week"]
    buyreward_percent: float = 0.004
    buyreward_maxwait: int = 20
    fidelity_input: str = "1m" # possible ["1m", "1h", "1d"]
    fidelity_run: str = "1m"
    layers: List[str] = field(default_factory=[])  # possible ["1h"], ["1h", "4h", "1d", "7d"], ["1m", "15m", "1h", "4h", "1d", "7d"]
    fidelity_input_test: str = "1m"
    fidelity_run_test: str = "1m"
    layers_test: List[str] = field(default_factory=[])
   
data_2017_to_2023vs2024_only_price_percent = DataConfig(
    id= "data_2017_to_2023vs2024_only_price_percent",
    type="only_price_percent",
    train_data_paths= [
        [
            # "binance/BTCUSDT-1m-2017-8.json", 
            # "binance/BTCUSDT-1m-2017-9.json", "binance/BTCUSDT-1m-2017-10.json", "binance/BTCUSDT-1m-2017-11.json", "binance/BTCUSDT-1m-2017-12.json",               
            # "binance/BTCUSDT-1m-2018-1.json", "binance/BTCUSDT-1m-2018-2.json", "binance/BTCUSDT-1m-2018-3.json", "binance/BTCUSDT-1m-2018-4.json", "binance/BTCUSDT-1m-2018-5.json", "binance/BTCUSDT-1m-2018-6.json", "binance/BTCUSDT-1m-2018-7.json", "binance/BTCUSDT-1m-2018-8.json", "binance/BTCUSDT-1m-2018-9.json", "binance/BTCUSDT-1m-2018-10.json", "binance/BTCUSDT-1m-2018-11.json", "binance/BTCUSDT-1m-2018-12.json",               
            # "binance/BTCUSDT-1m-2019-1.json", "binance/BTCUSDT-1m-2019-2.json", "binance/BTCUSDT-1m-2019-3.json", "binance/BTCUSDT-1m-2019-4.json", "binance/BTCUSDT-1m-2019-5.json", "binance/BTCUSDT-1m-2019-6.json", "binance/BTCUSDT-1m-2019-7.json", "binance/BTCUSDT-1m-2019-8.json", "binance/BTCUSDT-1m-2019-9.json", "binance/BTCUSDT-1m-2019-10.json", "binance/BTCUSDT-1m-2019-11.json", "binance/BTCUSDT-1m-2019-12.json",               
            # "binance/BTCUSDT-1m-2020-1.json", "binance/BTCUSDT-1m-2020-2.json", "binance/BTCUSDT-1m-2020-3.json", "binance/BTCUSDT-1m-2020-4.json", "binance/BTCUSDT-1m-2020-5.json", "binance/BTCUSDT-1m-2020-6.json", "binance/BTCUSDT-1m-2020-7.json", "binance/BTCUSDT-1m-2020-8.json", "binance/BTCUSDT-1m-2020-9.json", "binance/BTCUSDT-1m-2020-10.json", "binance/BTCUSDT-1m-2020-11.json", "binance/BTCUSDT-1m-2020-12.json",
            # "binance/BTCUSDT-1m-2021-1.json", "binance/BTCUSDT-1m-2021-2.json", "binance/BTCUSDT-1m-2021-3.json", "binance/BTCUSDT-1m-2021-4.json", "binance/BTCUSDT-1m-2021-5.json", "binance/BTCUSDT-1m-2021-6.json", "binance/BTCUSDT-1m-2021-7.json", "binance/BTCUSDT-1m-2021-8.json", "binance/BTCUSDT-1m-2021-9.json", "binance/BTCUSDT-1m-2021-10.json", "binance/BTCUSDT-1m-2021-11.json", "binance/BTCUSDT-1m-2021-12.json",
            "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
            "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"
        ]
    ],
    test_data_paths= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
    ],
    fidelity_run= "1h",
    layers = ["1h", "1d"],

    layers_test = ["1h", "1d"]
)
data_2017_to_2023vs2024_only_price_percent_32 = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, layers=data_2017_to_2023vs2024_only_price_percent.layers, fidelity_run=data_2017_to_2023vs2024_only_price_percent.fidelity_run, layers_test=data_2017_to_2023vs2024_only_price_percent.layers_test, fidelity_run_test=data_2017_to_2023vs2024_only_price_percent.fidelity_run_test, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32)

data_2017_to_2023vs2024_only_price_percent_32_1m = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32,
    layers = ["1m", "1h", "1d"],
    layers_test = ["1m", "1h", "1d"]
)
data_2017_to_2023vs2024_only_price_percent_32_5m = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32,
    fidelity_run= "5m",
    layers = ["5m", "1h", "1d"],
    layers_test = ["5m", "1h", "1d"]
)
data_2017_to_2023vs2024_only_price_percent_32_at_5m = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32,
    fidelity_run= "5m",
    layers = ["1h", "1d"],
    layers_test = ["1h", "1d"]
)
data_2017_to_2023vs2024_only_price_percent_32_10m = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32,
    fidelity_run= "10m",
    layers = ["10m", "1h", "1d"],
    layers_test = ["10m", "1h", "1d"]
)
data_2017_to_2023vs2024_only_price_percent_32_at_10m = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32,
    fidelity_run= "10m",
    layers = ["1h", "1d"],
    layers_test = ["1h", "1d"]
)
data_2017_to_2023vs2024_only_price_percent_32_15m = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32,
    fidelity_run= "15m",
    layers = ["15m", "1h", "1d"],
    layers_test = ["15m", "1h", "1d"]
)
data_2017_to_2023vs2024_only_price_percent_32_at_15m = DataConfig(id=data_2017_to_2023vs2024_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_only_price_percent.train_data_paths, test_data_paths=data_2017_to_2023vs2024_only_price_percent.test_data_paths, type=data_2017_to_2023vs2024_only_price_percent.type, lookback_window_size=32,
    fidelity_run= "15m",
    layers = ["1h", "1d"],
    layers_test = ["1h", "1d"]
)

def get_datas_1h_1d():
    # return [data_2017_to_2023vs2024_only_price_percent]
    # return [data_2017_to_2023vs2024_only_price_percent_32]
    # return [data_2017_to_2023vs2024_only_price_percent_32_1m]
    return [data_2017_to_2023vs2024_only_price_percent_32_5m]
    # return [data_2017_to_2023vs2024_only_price_percent_32_at_5m]
    # return [data_2017_to_2023vs2024_only_price_percent_32_10m]
    # return [data_2017_to_2023vs2024_only_price_percent_32_at_10m]
    # return [data_2017_to_2023vs2024_only_price_percent_32_15m]
    # return [data_2017_to_2023vs2024_only_price_percent_32_at_15m]

def get_datas_1m_1h_1d():
    return [data_2017_to_2023vs2024_only_price_percent_32_1m]