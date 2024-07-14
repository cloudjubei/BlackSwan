from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    id: str
    train_data_paths: List[str]
    test_data_paths: List[str]
    lookback_window_size: int = 1
    flat_lookback: bool = True
    type: str = "only_price_percent" # possible ["standard", "solo_price_percent", "only_price_percent", "all_percents"]
    indicator: str = "none"
    timestamp: str = "day_of_week" # possible ["none",  "expanded", "day_of_week"]
    percent_buysell: float = 0.004
    fidelity: str = "1h" # possible ["1m", "1h", "1d"]
    layers: List[str] = field(default_factory=[])  # possible ["1h"], ["1h", "4h", "1d", "7d"], ["1m", "15m", "1h", "4h", "1d", "7d"]
    flat_layers: bool = True
    train_data_paths_extra: List[List[str]] | None = None
    test_data_paths_extra: List[List[str]] | None = None
   
data_2017_to_2023vs2024_1h = DataConfig(
    id= "2017_to_2023vs2024_1h",
    train_data_paths= ["binance/BTCUSDT-1h-2017-8.json", "binance/BTCUSDT-1h-2017-9.json", "binance/BTCUSDT-1h-2017-10.json", "binance/BTCUSDT-1h-2017-11.json", "binance/BTCUSDT-1h-2017-12.json",
                       "binance/BTCUSDT-1h-2018-1.json", "binance/BTCUSDT-1h-2018-2.json", "binance/BTCUSDT-1h-2018-3.json", "binance/BTCUSDT-1h-2018-4.json", "binance/BTCUSDT-1h-2018-5.json", "binance/BTCUSDT-1h-2018-6.json", "binance/BTCUSDT-1h-2018-7.json", "binance/BTCUSDT-1h-2018-8.json", "binance/BTCUSDT-1h-2018-9.json", "binance/BTCUSDT-1h-2018-10.json", "binance/BTCUSDT-1h-2018-11.json", "binance/BTCUSDT-1h-2018-12.json",
                       "binance/BTCUSDT-1h-2019-1.json", "binance/BTCUSDT-1h-2019-2.json", "binance/BTCUSDT-1h-2019-3.json", "binance/BTCUSDT-1h-2019-4.json", "binance/BTCUSDT-1h-2019-5.json", "binance/BTCUSDT-1h-2019-6.json", "binance/BTCUSDT-1h-2019-7.json", "binance/BTCUSDT-1h-2019-8.json", "binance/BTCUSDT-1h-2019-9.json", "binance/BTCUSDT-1h-2019-10.json", "binance/BTCUSDT-1h-2019-11.json", "binance/BTCUSDT-1h-2019-12.json",
                       "binance/BTCUSDT-1h-2020-1.json", "binance/BTCUSDT-1h-2020-2.json", "binance/BTCUSDT-1h-2020-3.json", "binance/BTCUSDT-1h-2020-4.json", "binance/BTCUSDT-1h-2020-5.json", "binance/BTCUSDT-1h-2020-6.json", "binance/BTCUSDT-1h-2020-7.json", "binance/BTCUSDT-1h-2020-8.json", "binance/BTCUSDT-1h-2020-9.json", "binance/BTCUSDT-1h-2020-10.json", "binance/BTCUSDT-1h-2020-11.json", "binance/BTCUSDT-1h-2020-12.json",
                       "binance/BTCUSDT-1h-2021-1.json", "binance/BTCUSDT-1h-2021-2.json", "binance/BTCUSDT-1h-2021-3.json", "binance/BTCUSDT-1h-2021-4.json", "binance/BTCUSDT-1h-2021-5.json", "binance/BTCUSDT-1h-2021-6.json", "binance/BTCUSDT-1h-2021-7.json", "binance/BTCUSDT-1h-2021-8.json", "binance/BTCUSDT-1h-2021-9.json", "binance/BTCUSDT-1h-2021-10.json", "binance/BTCUSDT-1h-2021-11.json", "binance/BTCUSDT-1h-2021-12.json",
                       "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json",
                       "binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"],
    test_data_paths= ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
    layers=["1h"]
)

data_downtrend_2021dec_2022dec_to_3 = DataConfig(
    id= "downtrend_2021dec_2022dec_to_3_1h",
    train_data_paths= ["binance/BTCUSDT-1h-2021-12.json", "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json"],
    test_data_paths= ["binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json"],
    layers=["1h"]
)

data_2023vs2024_1m = DataConfig(
    id= "2023vs2024_1m",
    train_data_paths= ["binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"],
    test_data_paths= ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
    fidelity = "1m",
    layers = ["1m"]
)
data_2020_to_2023vs2024_1m = DataConfig(
    id= "data_2020_to_2023vs2024_1m",
    train_data_paths= [
        "binance/BTCUSDT-1m-2020-1.json", "binance/BTCUSDT-1m-2020-2.json", "binance/BTCUSDT-1m-2020-3.json", "binance/BTCUSDT-1m-2020-4.json", "binance/BTCUSDT-1m-2020-5.json", "binance/BTCUSDT-1m-2020-6.json", "binance/BTCUSDT-1m-2020-7.json", "binance/BTCUSDT-1m-2020-8.json", "binance/BTCUSDT-1m-2020-9.json", "binance/BTCUSDT-1m-2020-10.json", "binance/BTCUSDT-1m-2020-11.json", "binance/BTCUSDT-1m-2020-12.json",               
        "binance/BTCUSDT-1m-2021-1.json", "binance/BTCUSDT-1m-2021-2.json", "binance/BTCUSDT-1m-2021-3.json", "binance/BTCUSDT-1m-2021-4.json", "binance/BTCUSDT-1m-2021-5.json", "binance/BTCUSDT-1m-2021-6.json", "binance/BTCUSDT-1m-2021-7.json", "binance/BTCUSDT-1m-2021-8.json", "binance/BTCUSDT-1m-2021-9.json", "binance/BTCUSDT-1m-2021-10.json", "binance/BTCUSDT-1m-2021-11.json", "binance/BTCUSDT-1m-2021-12.json",
        "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
        "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"],
    test_data_paths= ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
    fidelity = "1m",
    layers = ["1m"]
)

data_2017_to_2023vs2024_1m = DataConfig(
    id= "data_2017_to_2023vs2024_1m",
    train_data_paths= [
        "binance/BTCUSDT-1m-2017-8.json", "binance/BTCUSDT-1m-2017-9.json", "binance/BTCUSDT-1m-2017-10.json", "binance/BTCUSDT-1m-2017-11.json", "binance/BTCUSDT-1m-2017-12.json",               
        "binance/BTCUSDT-1m-2018-1.json", "binance/BTCUSDT-1m-2018-2.json", "binance/BTCUSDT-1m-2018-3.json", "binance/BTCUSDT-1m-2018-4.json", "binance/BTCUSDT-1m-2018-5.json", "binance/BTCUSDT-1m-2018-6.json", "binance/BTCUSDT-1m-2018-7.json", "binance/BTCUSDT-1m-2018-8.json", "binance/BTCUSDT-1m-2018-9.json", "binance/BTCUSDT-1m-2018-10.json", "binance/BTCUSDT-1m-2018-11.json", "binance/BTCUSDT-1m-2018-12.json",               
        "binance/BTCUSDT-1m-2019-1.json", "binance/BTCUSDT-1m-2019-2.json", "binance/BTCUSDT-1m-2019-3.json", "binance/BTCUSDT-1m-2019-4.json", "binance/BTCUSDT-1m-2019-5.json", "binance/BTCUSDT-1m-2019-6.json", "binance/BTCUSDT-1m-2019-7.json", "binance/BTCUSDT-1m-2019-8.json", "binance/BTCUSDT-1m-2019-9.json", "binance/BTCUSDT-1m-2019-10.json", "binance/BTCUSDT-1m-2019-11.json", "binance/BTCUSDT-1m-2019-12.json",               
        "binance/BTCUSDT-1m-2020-1.json", "binance/BTCUSDT-1m-2020-2.json", "binance/BTCUSDT-1m-2020-3.json", "binance/BTCUSDT-1m-2020-4.json", "binance/BTCUSDT-1m-2020-5.json", "binance/BTCUSDT-1m-2020-6.json", "binance/BTCUSDT-1m-2020-7.json", "binance/BTCUSDT-1m-2020-8.json", "binance/BTCUSDT-1m-2020-9.json", "binance/BTCUSDT-1m-2020-10.json", "binance/BTCUSDT-1m-2020-11.json", "binance/BTCUSDT-1m-2020-12.json",               
        "binance/BTCUSDT-1m-2021-1.json", "binance/BTCUSDT-1m-2021-2.json", "binance/BTCUSDT-1m-2021-3.json", "binance/BTCUSDT-1m-2021-4.json", "binance/BTCUSDT-1m-2021-5.json", "binance/BTCUSDT-1m-2021-6.json", "binance/BTCUSDT-1m-2021-7.json", "binance/BTCUSDT-1m-2021-8.json", "binance/BTCUSDT-1m-2021-9.json", "binance/BTCUSDT-1m-2021-10.json", "binance/BTCUSDT-1m-2021-11.json", "binance/BTCUSDT-1m-2021-12.json",
        "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
        "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"],
    test_data_paths= ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
    fidelity = "1m",
    layers = ["1m"]
)

data_2023vs2024_1m_1h_1d_only_price_percent = DataConfig(
    id= "2023vs2024_1m_1h_1d_only_price_percent",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        ["binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"],
        ["binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"],
        ["binance/BTCUSDT-1d-2023-1.json", "binance/BTCUSDT-1d-2023-2.json", "binance/BTCUSDT-1d-2023-3.json", "binance/BTCUSDT-1d-2023-4.json", "binance/BTCUSDT-1d-2023-5.json", "binance/BTCUSDT-1d-2023-6.json", "binance/BTCUSDT-1d-2023-7.json", "binance/BTCUSDT-1d-2023-8.json", "binance/BTCUSDT-1d-2023-9.json", "binance/BTCUSDT-1d-2023-10.json", "binance/BTCUSDT-1d-2023-11.json", "binance/BTCUSDT-1d-2023-12.json"]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
        ["binance/BTCUSDT-1d-2024-1.json", "binance/BTCUSDT-1d-2024-2.json", "binance/BTCUSDT-1d-2024-3.json", "binance/BTCUSDT-1d-2024-4.json"],
      ],
    fidelity = "1m",
    layers = ["1m", "1h", "1d"]
)
data_2022vs2024_1m_1h_1d_only_price_percent = DataConfig(
    id= "2022vs2024_1m_1h_1d_only_price_percent",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        [
            "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
            "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json",
            "binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1d-2022-1.json", "binance/BTCUSDT-1d-2022-2.json", "binance/BTCUSDT-1d-2022-3.json", "binance/BTCUSDT-1d-2022-4.json", "binance/BTCUSDT-1d-2022-5.json", "binance/BTCUSDT-1d-2022-6.json", "binance/BTCUSDT-1d-2022-7.json", "binance/BTCUSDT-1d-2022-8.json", "binance/BTCUSDT-1d-2022-9.json", "binance/BTCUSDT-1d-2022-10.json", "binance/BTCUSDT-1d-2022-11.json", "binance/BTCUSDT-1d-2022-12.json",
            "binance/BTCUSDT-1d-2023-1.json", "binance/BTCUSDT-1d-2023-2.json", "binance/BTCUSDT-1d-2023-3.json", "binance/BTCUSDT-1d-2023-4.json", "binance/BTCUSDT-1d-2023-5.json", "binance/BTCUSDT-1d-2023-6.json", "binance/BTCUSDT-1d-2023-7.json", "binance/BTCUSDT-1d-2023-8.json", "binance/BTCUSDT-1d-2023-9.json", "binance/BTCUSDT-1d-2023-10.json", "binance/BTCUSDT-1d-2023-11.json", "binance/BTCUSDT-1d-2023-12.json"
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
        ["binance/BTCUSDT-1d-2024-1.json", "binance/BTCUSDT-1d-2024-2.json", "binance/BTCUSDT-1d-2024-3.json", "binance/BTCUSDT-1d-2024-4.json"],
      ],
    fidelity = "1m",
    layers = ["1m", "1h", "1d"]
)
data_2022vs2024_1m_1h_1d_only_price_percent_32 = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=32)
data_2022vs2024_1m_1h_1d_only_price_percent_32_none = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, timestamp="none", lookback_window_size=32)


data_2021vs2024_1m_1h_1d_only_price_percent = DataConfig(
    id= "2021vs2024_1m_1h_1d_only_price_percent",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        [
            "binance/BTCUSDT-1m-2021-1.json", "binance/BTCUSDT-1m-2021-2.json", "binance/BTCUSDT-1m-2021-3.json", "binance/BTCUSDT-1m-2021-4.json", "binance/BTCUSDT-1m-2021-5.json", "binance/BTCUSDT-1m-2021-6.json", "binance/BTCUSDT-1m-2021-7.json", "binance/BTCUSDT-1m-2021-8.json", "binance/BTCUSDT-1m-2021-9.json", "binance/BTCUSDT-1m-2021-10.json", "binance/BTCUSDT-1m-2021-11.json", "binance/BTCUSDT-1m-2021-12.json",
            "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
            "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1h-2021-1.json", "binance/BTCUSDT-1h-2021-2.json", "binance/BTCUSDT-1h-2021-3.json", "binance/BTCUSDT-1h-2021-4.json", "binance/BTCUSDT-1h-2021-5.json", "binance/BTCUSDT-1h-2021-6.json", "binance/BTCUSDT-1h-2021-7.json", "binance/BTCUSDT-1h-2021-8.json", "binance/BTCUSDT-1h-2021-9.json", "binance/BTCUSDT-1h-2021-10.json", "binance/BTCUSDT-1h-2021-11.json", "binance/BTCUSDT-1h-2021-12.json",
            "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json",
            "binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1d-2021-1.json", "binance/BTCUSDT-1d-2021-2.json", "binance/BTCUSDT-1d-2021-3.json", "binance/BTCUSDT-1d-2021-4.json", "binance/BTCUSDT-1d-2021-5.json", "binance/BTCUSDT-1d-2021-6.json", "binance/BTCUSDT-1d-2021-7.json", "binance/BTCUSDT-1d-2021-8.json", "binance/BTCUSDT-1d-2021-9.json", "binance/BTCUSDT-1d-2021-10.json", "binance/BTCUSDT-1d-2021-11.json", "binance/BTCUSDT-1d-2021-12.json",
            "binance/BTCUSDT-1d-2022-1.json", "binance/BTCUSDT-1d-2022-2.json", "binance/BTCUSDT-1d-2022-3.json", "binance/BTCUSDT-1d-2022-4.json", "binance/BTCUSDT-1d-2022-5.json", "binance/BTCUSDT-1d-2022-6.json", "binance/BTCUSDT-1d-2022-7.json", "binance/BTCUSDT-1d-2022-8.json", "binance/BTCUSDT-1d-2022-9.json", "binance/BTCUSDT-1d-2022-10.json", "binance/BTCUSDT-1d-2022-11.json", "binance/BTCUSDT-1d-2022-12.json",
            "binance/BTCUSDT-1d-2023-1.json", "binance/BTCUSDT-1d-2023-2.json", "binance/BTCUSDT-1d-2023-3.json", "binance/BTCUSDT-1d-2023-4.json", "binance/BTCUSDT-1d-2023-5.json", "binance/BTCUSDT-1d-2023-6.json", "binance/BTCUSDT-1d-2023-7.json", "binance/BTCUSDT-1d-2023-8.json", "binance/BTCUSDT-1d-2023-9.json", "binance/BTCUSDT-1d-2023-10.json", "binance/BTCUSDT-1d-2023-11.json", "binance/BTCUSDT-1d-2023-12.json"
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
        ["binance/BTCUSDT-1d-2024-1.json", "binance/BTCUSDT-1d-2024-2.json", "binance/BTCUSDT-1d-2024-3.json", "binance/BTCUSDT-1d-2024-4.json"],
      ],
    fidelity = "1m",
    layers = ["1m", "1h", "1d"]
)
data_2021vs2024_1m_1h_1d_only_price_percent_32 = DataConfig(id=data_2021vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2021vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2021vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2021vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2021vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2021vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2021vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2021vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=32)

data_2020vs2024_1m_1h_1d_only_price_percent = DataConfig(
    id= "2020vs2024_1m_1h_1d_only_price_percent",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        [
            "binance/BTCUSDT-1m-2020-1.json", "binance/BTCUSDT-1m-2020-2.json", "binance/BTCUSDT-1m-2020-3.json", "binance/BTCUSDT-1m-2020-4.json", "binance/BTCUSDT-1m-2020-5.json", "binance/BTCUSDT-1m-2020-6.json", "binance/BTCUSDT-1m-2020-7.json", "binance/BTCUSDT-1m-2020-8.json", "binance/BTCUSDT-1m-2020-9.json", "binance/BTCUSDT-1m-2020-10.json", "binance/BTCUSDT-1m-2020-11.json", "binance/BTCUSDT-1m-2020-12.json",
            "binance/BTCUSDT-1m-2021-1.json", "binance/BTCUSDT-1m-2021-2.json", "binance/BTCUSDT-1m-2021-3.json", "binance/BTCUSDT-1m-2021-4.json", "binance/BTCUSDT-1m-2021-5.json", "binance/BTCUSDT-1m-2021-6.json", "binance/BTCUSDT-1m-2021-7.json", "binance/BTCUSDT-1m-2021-8.json", "binance/BTCUSDT-1m-2021-9.json", "binance/BTCUSDT-1m-2021-10.json", "binance/BTCUSDT-1m-2021-11.json", "binance/BTCUSDT-1m-2021-12.json",
            "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
            "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1h-2020-1.json", "binance/BTCUSDT-1h-2020-2.json", "binance/BTCUSDT-1h-2020-3.json", "binance/BTCUSDT-1h-2020-4.json", "binance/BTCUSDT-1h-2020-5.json", "binance/BTCUSDT-1h-2020-6.json", "binance/BTCUSDT-1h-2020-7.json", "binance/BTCUSDT-1h-2020-8.json", "binance/BTCUSDT-1h-2020-9.json", "binance/BTCUSDT-1h-2020-10.json", "binance/BTCUSDT-1h-2020-11.json", "binance/BTCUSDT-1h-2020-12.json",
            "binance/BTCUSDT-1h-2021-1.json", "binance/BTCUSDT-1h-2021-2.json", "binance/BTCUSDT-1h-2021-3.json", "binance/BTCUSDT-1h-2021-4.json", "binance/BTCUSDT-1h-2021-5.json", "binance/BTCUSDT-1h-2021-6.json", "binance/BTCUSDT-1h-2021-7.json", "binance/BTCUSDT-1h-2021-8.json", "binance/BTCUSDT-1h-2021-9.json", "binance/BTCUSDT-1h-2021-10.json", "binance/BTCUSDT-1h-2021-11.json", "binance/BTCUSDT-1h-2021-12.json",
            "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json",
            "binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1d-2020-1.json", "binance/BTCUSDT-1d-2020-2.json", "binance/BTCUSDT-1d-2020-3.json", "binance/BTCUSDT-1d-2020-4.json", "binance/BTCUSDT-1d-2020-5.json", "binance/BTCUSDT-1d-2020-6.json", "binance/BTCUSDT-1d-2020-7.json", "binance/BTCUSDT-1d-2020-8.json", "binance/BTCUSDT-1d-2020-9.json", "binance/BTCUSDT-1d-2020-10.json", "binance/BTCUSDT-1d-2020-11.json", "binance/BTCUSDT-1d-2020-12.json",
            "binance/BTCUSDT-1d-2021-1.json", "binance/BTCUSDT-1d-2021-2.json", "binance/BTCUSDT-1d-2021-3.json", "binance/BTCUSDT-1d-2021-4.json", "binance/BTCUSDT-1d-2021-5.json", "binance/BTCUSDT-1d-2021-6.json", "binance/BTCUSDT-1d-2021-7.json", "binance/BTCUSDT-1d-2021-8.json", "binance/BTCUSDT-1d-2021-9.json", "binance/BTCUSDT-1d-2021-10.json", "binance/BTCUSDT-1d-2021-11.json", "binance/BTCUSDT-1d-2021-12.json",
            "binance/BTCUSDT-1d-2022-1.json", "binance/BTCUSDT-1d-2022-2.json", "binance/BTCUSDT-1d-2022-3.json", "binance/BTCUSDT-1d-2022-4.json", "binance/BTCUSDT-1d-2022-5.json", "binance/BTCUSDT-1d-2022-6.json", "binance/BTCUSDT-1d-2022-7.json", "binance/BTCUSDT-1d-2022-8.json", "binance/BTCUSDT-1d-2022-9.json", "binance/BTCUSDT-1d-2022-10.json", "binance/BTCUSDT-1d-2022-11.json", "binance/BTCUSDT-1d-2022-12.json",
            "binance/BTCUSDT-1d-2023-1.json", "binance/BTCUSDT-1d-2023-2.json", "binance/BTCUSDT-1d-2023-3.json", "binance/BTCUSDT-1d-2023-4.json", "binance/BTCUSDT-1d-2023-5.json", "binance/BTCUSDT-1d-2023-6.json", "binance/BTCUSDT-1d-2023-7.json", "binance/BTCUSDT-1d-2023-8.json", "binance/BTCUSDT-1d-2023-9.json", "binance/BTCUSDT-1d-2023-10.json", "binance/BTCUSDT-1d-2023-11.json", "binance/BTCUSDT-1d-2023-12.json"
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
        ["binance/BTCUSDT-1d-2024-1.json", "binance/BTCUSDT-1d-2024-2.json", "binance/BTCUSDT-1d-2024-3.json", "binance/BTCUSDT-1d-2024-4.json"],
      ],
    fidelity = "1m",
    layers = ["1m", "1h", "1d"]
)
data_2020vs2024_1m_1h_1d_only_price_percent_32 = DataConfig(id=data_2020vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2020vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2020vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2020vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2020vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2020vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2020vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2020vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=32)

data_2017vs2024_1m_1h_1d_only_price_percent = DataConfig(
    id= "2017vs2024_1m_1h_1d_only_price_percent",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        [
            "binance/BTCUSDT-1m-2017-8.json", "binance/BTCUSDT-1m-2017-9.json", "binance/BTCUSDT-1m-2017-10.json", "binance/BTCUSDT-1m-2017-11.json", "binance/BTCUSDT-1m-2017-12.json",               
            "binance/BTCUSDT-1m-2018-1.json", "binance/BTCUSDT-1m-2018-2.json", "binance/BTCUSDT-1m-2018-3.json", "binance/BTCUSDT-1m-2018-4.json", "binance/BTCUSDT-1m-2018-5.json", "binance/BTCUSDT-1m-2018-6.json", "binance/BTCUSDT-1m-2018-7.json", "binance/BTCUSDT-1m-2018-8.json", "binance/BTCUSDT-1m-2018-9.json", "binance/BTCUSDT-1m-2018-10.json", "binance/BTCUSDT-1m-2018-11.json", "binance/BTCUSDT-1m-2018-12.json",               
            "binance/BTCUSDT-1m-2019-1.json", "binance/BTCUSDT-1m-2019-2.json", "binance/BTCUSDT-1m-2019-3.json", "binance/BTCUSDT-1m-2019-4.json", "binance/BTCUSDT-1m-2019-5.json", "binance/BTCUSDT-1m-2019-6.json", "binance/BTCUSDT-1m-2019-7.json", "binance/BTCUSDT-1m-2019-8.json", "binance/BTCUSDT-1m-2019-9.json", "binance/BTCUSDT-1m-2019-10.json", "binance/BTCUSDT-1m-2019-11.json", "binance/BTCUSDT-1m-2019-12.json",               
            "binance/BTCUSDT-1m-2020-1.json", "binance/BTCUSDT-1m-2020-2.json", "binance/BTCUSDT-1m-2020-3.json", "binance/BTCUSDT-1m-2020-4.json", "binance/BTCUSDT-1m-2020-5.json", "binance/BTCUSDT-1m-2020-6.json", "binance/BTCUSDT-1m-2020-7.json", "binance/BTCUSDT-1m-2020-8.json", "binance/BTCUSDT-1m-2020-9.json", "binance/BTCUSDT-1m-2020-10.json", "binance/BTCUSDT-1m-2020-11.json", "binance/BTCUSDT-1m-2020-12.json",
            "binance/BTCUSDT-1m-2021-1.json", "binance/BTCUSDT-1m-2021-2.json", "binance/BTCUSDT-1m-2021-3.json", "binance/BTCUSDT-1m-2021-4.json", "binance/BTCUSDT-1m-2021-5.json", "binance/BTCUSDT-1m-2021-6.json", "binance/BTCUSDT-1m-2021-7.json", "binance/BTCUSDT-1m-2021-8.json", "binance/BTCUSDT-1m-2021-9.json", "binance/BTCUSDT-1m-2021-10.json", "binance/BTCUSDT-1m-2021-11.json", "binance/BTCUSDT-1m-2021-12.json",
            "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
            "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1h-2017-8.json", "binance/BTCUSDT-1h-2017-9.json", "binance/BTCUSDT-1h-2017-10.json", "binance/BTCUSDT-1h-2017-11.json", "binance/BTCUSDT-1h-2017-12.json",               
            "binance/BTCUSDT-1h-2018-1.json", "binance/BTCUSDT-1h-2018-2.json", "binance/BTCUSDT-1h-2018-3.json", "binance/BTCUSDT-1h-2018-4.json", "binance/BTCUSDT-1h-2018-5.json", "binance/BTCUSDT-1h-2018-6.json", "binance/BTCUSDT-1h-2018-7.json", "binance/BTCUSDT-1h-2018-8.json", "binance/BTCUSDT-1h-2018-9.json", "binance/BTCUSDT-1h-2018-10.json", "binance/BTCUSDT-1h-2018-11.json", "binance/BTCUSDT-1h-2018-12.json",               
            "binance/BTCUSDT-1h-2019-1.json", "binance/BTCUSDT-1h-2019-2.json", "binance/BTCUSDT-1h-2019-3.json", "binance/BTCUSDT-1h-2019-4.json", "binance/BTCUSDT-1h-2019-5.json", "binance/BTCUSDT-1h-2019-6.json", "binance/BTCUSDT-1h-2019-7.json", "binance/BTCUSDT-1h-2019-8.json", "binance/BTCUSDT-1h-2019-9.json", "binance/BTCUSDT-1h-2019-10.json", "binance/BTCUSDT-1h-2019-11.json", "binance/BTCUSDT-1h-2019-12.json",               
            "binance/BTCUSDT-1h-2020-1.json", "binance/BTCUSDT-1h-2020-2.json", "binance/BTCUSDT-1h-2020-3.json", "binance/BTCUSDT-1h-2020-4.json", "binance/BTCUSDT-1h-2020-5.json", "binance/BTCUSDT-1h-2020-6.json", "binance/BTCUSDT-1h-2020-7.json", "binance/BTCUSDT-1h-2020-8.json", "binance/BTCUSDT-1h-2020-9.json", "binance/BTCUSDT-1h-2020-10.json", "binance/BTCUSDT-1h-2020-11.json", "binance/BTCUSDT-1h-2020-12.json",               
            "binance/BTCUSDT-1h-2021-1.json", "binance/BTCUSDT-1h-2021-2.json", "binance/BTCUSDT-1h-2021-3.json", "binance/BTCUSDT-1h-2021-4.json", "binance/BTCUSDT-1h-2021-5.json", "binance/BTCUSDT-1h-2021-6.json", "binance/BTCUSDT-1h-2021-7.json", "binance/BTCUSDT-1h-2021-8.json", "binance/BTCUSDT-1h-2021-9.json", "binance/BTCUSDT-1h-2021-10.json", "binance/BTCUSDT-1h-2021-11.json", "binance/BTCUSDT-1h-2021-12.json",
            "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json",
            "binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1d-2017-8.json", "binance/BTCUSDT-1d-2017-9.json", "binance/BTCUSDT-1d-2017-10.json", "binance/BTCUSDT-1d-2017-11.json", "binance/BTCUSDT-1d-2017-12.json",               
            "binance/BTCUSDT-1d-2018-1.json", "binance/BTCUSDT-1d-2018-2.json", "binance/BTCUSDT-1d-2018-3.json", "binance/BTCUSDT-1d-2018-4.json", "binance/BTCUSDT-1d-2018-5.json", "binance/BTCUSDT-1d-2018-6.json", "binance/BTCUSDT-1d-2018-7.json", "binance/BTCUSDT-1d-2018-8.json", "binance/BTCUSDT-1d-2018-9.json", "binance/BTCUSDT-1d-2018-10.json", "binance/BTCUSDT-1d-2018-11.json", "binance/BTCUSDT-1d-2018-12.json",               
            "binance/BTCUSDT-1d-2019-1.json", "binance/BTCUSDT-1d-2019-2.json", "binance/BTCUSDT-1d-2019-3.json", "binance/BTCUSDT-1d-2019-4.json", "binance/BTCUSDT-1d-2019-5.json", "binance/BTCUSDT-1d-2019-6.json", "binance/BTCUSDT-1d-2019-7.json", "binance/BTCUSDT-1d-2019-8.json", "binance/BTCUSDT-1d-2019-9.json", "binance/BTCUSDT-1d-2019-10.json", "binance/BTCUSDT-1d-2019-11.json", "binance/BTCUSDT-1d-2019-12.json",               
            "binance/BTCUSDT-1d-2020-1.json", "binance/BTCUSDT-1d-2020-2.json", "binance/BTCUSDT-1d-2020-3.json", "binance/BTCUSDT-1d-2020-4.json", "binance/BTCUSDT-1d-2020-5.json", "binance/BTCUSDT-1d-2020-6.json", "binance/BTCUSDT-1d-2020-7.json", "binance/BTCUSDT-1d-2020-8.json", "binance/BTCUSDT-1d-2020-9.json", "binance/BTCUSDT-1d-2020-10.json", "binance/BTCUSDT-1d-2020-11.json", "binance/BTCUSDT-1d-2020-12.json",               
            "binance/BTCUSDT-1d-2021-1.json", "binance/BTCUSDT-1d-2021-2.json", "binance/BTCUSDT-1d-2021-3.json", "binance/BTCUSDT-1d-2021-4.json", "binance/BTCUSDT-1d-2021-5.json", "binance/BTCUSDT-1d-2021-6.json", "binance/BTCUSDT-1d-2021-7.json", "binance/BTCUSDT-1d-2021-8.json", "binance/BTCUSDT-1d-2021-9.json", "binance/BTCUSDT-1d-2021-10.json", "binance/BTCUSDT-1d-2021-11.json", "binance/BTCUSDT-1d-2021-12.json",
            "binance/BTCUSDT-1d-2022-1.json", "binance/BTCUSDT-1d-2022-2.json", "binance/BTCUSDT-1d-2022-3.json", "binance/BTCUSDT-1d-2022-4.json", "binance/BTCUSDT-1d-2022-5.json", "binance/BTCUSDT-1d-2022-6.json", "binance/BTCUSDT-1d-2022-7.json", "binance/BTCUSDT-1d-2022-8.json", "binance/BTCUSDT-1d-2022-9.json", "binance/BTCUSDT-1d-2022-10.json", "binance/BTCUSDT-1d-2022-11.json", "binance/BTCUSDT-1d-2022-12.json",
            "binance/BTCUSDT-1d-2023-1.json", "binance/BTCUSDT-1d-2023-2.json", "binance/BTCUSDT-1d-2023-3.json", "binance/BTCUSDT-1d-2023-4.json", "binance/BTCUSDT-1d-2023-5.json", "binance/BTCUSDT-1d-2023-6.json", "binance/BTCUSDT-1d-2023-7.json", "binance/BTCUSDT-1d-2023-8.json", "binance/BTCUSDT-1d-2023-9.json", "binance/BTCUSDT-1d-2023-10.json", "binance/BTCUSDT-1d-2023-11.json", "binance/BTCUSDT-1d-2023-12.json"
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
        ["binance/BTCUSDT-1d-2024-1.json", "binance/BTCUSDT-1d-2024-2.json", "binance/BTCUSDT-1d-2024-3.json", "binance/BTCUSDT-1d-2024-4.json"],
    ],
    fidelity = "1m",
    layers = ["1m", "1h", "1d"]
)
data_2017vs2024_1m_1h_1d_only_price_percent_32 = DataConfig(id=data_2017vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2017vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2017vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2017vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=32)
data_2017vs2024_1m_1h_1d_only_price_percent_32_none = DataConfig(id=data_2017vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2017vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2017vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2017vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="none")


data_2017_to_2023vs2024_1h_1d_only_price_percent = DataConfig(
    id= "data_2017_to_2023vs2024_1h_1d",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        [
            "binance/BTCUSDT-1h-2017-8.json", "binance/BTCUSDT-1h-2017-9.json", "binance/BTCUSDT-1h-2017-10.json", "binance/BTCUSDT-1h-2017-11.json", "binance/BTCUSDT-1h-2017-12.json",               
            "binance/BTCUSDT-1h-2018-1.json", "binance/BTCUSDT-1h-2018-2.json", "binance/BTCUSDT-1h-2018-3.json", "binance/BTCUSDT-1h-2018-4.json", "binance/BTCUSDT-1h-2018-5.json", "binance/BTCUSDT-1h-2018-6.json", "binance/BTCUSDT-1h-2018-7.json", "binance/BTCUSDT-1h-2018-8.json", "binance/BTCUSDT-1h-2018-9.json", "binance/BTCUSDT-1h-2018-10.json", "binance/BTCUSDT-1h-2018-11.json", "binance/BTCUSDT-1h-2018-12.json",               
            "binance/BTCUSDT-1h-2019-1.json", "binance/BTCUSDT-1h-2019-2.json", "binance/BTCUSDT-1h-2019-3.json", "binance/BTCUSDT-1h-2019-4.json", "binance/BTCUSDT-1h-2019-5.json", "binance/BTCUSDT-1h-2019-6.json", "binance/BTCUSDT-1h-2019-7.json", "binance/BTCUSDT-1h-2019-8.json", "binance/BTCUSDT-1h-2019-9.json", "binance/BTCUSDT-1h-2019-10.json", "binance/BTCUSDT-1h-2019-11.json", "binance/BTCUSDT-1h-2019-12.json",               
            "binance/BTCUSDT-1h-2020-1.json", "binance/BTCUSDT-1h-2020-2.json", "binance/BTCUSDT-1h-2020-3.json", "binance/BTCUSDT-1h-2020-4.json", "binance/BTCUSDT-1h-2020-5.json", "binance/BTCUSDT-1h-2020-6.json", "binance/BTCUSDT-1h-2020-7.json", "binance/BTCUSDT-1h-2020-8.json", "binance/BTCUSDT-1h-2020-9.json", "binance/BTCUSDT-1h-2020-10.json", "binance/BTCUSDT-1h-2020-11.json", "binance/BTCUSDT-1h-2020-12.json",               
            "binance/BTCUSDT-1h-2021-1.json", "binance/BTCUSDT-1h-2021-2.json", "binance/BTCUSDT-1h-2021-3.json", "binance/BTCUSDT-1h-2021-4.json", "binance/BTCUSDT-1h-2021-5.json", "binance/BTCUSDT-1h-2021-6.json", "binance/BTCUSDT-1h-2021-7.json", "binance/BTCUSDT-1h-2021-8.json", "binance/BTCUSDT-1h-2021-9.json", "binance/BTCUSDT-1h-2021-10.json", "binance/BTCUSDT-1h-2021-11.json", "binance/BTCUSDT-1h-2021-12.json",
            "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json",
            "binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"
        ],
        [
            "binance/BTCUSDT-1d-2017-8.json", "binance/BTCUSDT-1d-2017-9.json", "binance/BTCUSDT-1d-2017-10.json", "binance/BTCUSDT-1d-2017-11.json", "binance/BTCUSDT-1d-2017-12.json",               
            "binance/BTCUSDT-1d-2018-1.json", "binance/BTCUSDT-1d-2018-2.json", "binance/BTCUSDT-1d-2018-3.json", "binance/BTCUSDT-1d-2018-4.json", "binance/BTCUSDT-1d-2018-5.json", "binance/BTCUSDT-1d-2018-6.json", "binance/BTCUSDT-1d-2018-7.json", "binance/BTCUSDT-1d-2018-8.json", "binance/BTCUSDT-1d-2018-9.json", "binance/BTCUSDT-1d-2018-10.json", "binance/BTCUSDT-1d-2018-11.json", "binance/BTCUSDT-1d-2018-12.json",               
            "binance/BTCUSDT-1d-2019-1.json", "binance/BTCUSDT-1d-2019-2.json", "binance/BTCUSDT-1d-2019-3.json", "binance/BTCUSDT-1d-2019-4.json", "binance/BTCUSDT-1d-2019-5.json", "binance/BTCUSDT-1d-2019-6.json", "binance/BTCUSDT-1d-2019-7.json", "binance/BTCUSDT-1d-2019-8.json", "binance/BTCUSDT-1d-2019-9.json", "binance/BTCUSDT-1d-2019-10.json", "binance/BTCUSDT-1d-2019-11.json", "binance/BTCUSDT-1d-2019-12.json",               
            "binance/BTCUSDT-1d-2020-1.json", "binance/BTCUSDT-1d-2020-2.json", "binance/BTCUSDT-1d-2020-3.json", "binance/BTCUSDT-1d-2020-4.json", "binance/BTCUSDT-1d-2020-5.json", "binance/BTCUSDT-1d-2020-6.json", "binance/BTCUSDT-1d-2020-7.json", "binance/BTCUSDT-1d-2020-8.json", "binance/BTCUSDT-1d-2020-9.json", "binance/BTCUSDT-1d-2020-10.json", "binance/BTCUSDT-1d-2020-11.json", "binance/BTCUSDT-1d-2020-12.json",               
            "binance/BTCUSDT-1d-2021-1.json", "binance/BTCUSDT-1d-2021-2.json", "binance/BTCUSDT-1d-2021-3.json", "binance/BTCUSDT-1d-2021-4.json", "binance/BTCUSDT-1d-2021-5.json", "binance/BTCUSDT-1d-2021-6.json", "binance/BTCUSDT-1d-2021-7.json", "binance/BTCUSDT-1d-2021-8.json", "binance/BTCUSDT-1d-2021-9.json", "binance/BTCUSDT-1d-2021-10.json", "binance/BTCUSDT-1d-2021-11.json", "binance/BTCUSDT-1d-2021-12.json",
            "binance/BTCUSDT-1d-2022-1.json", "binance/BTCUSDT-1d-2022-2.json", "binance/BTCUSDT-1d-2022-3.json", "binance/BTCUSDT-1d-2022-4.json", "binance/BTCUSDT-1d-2022-5.json", "binance/BTCUSDT-1d-2022-6.json", "binance/BTCUSDT-1d-2022-7.json", "binance/BTCUSDT-1d-2022-8.json", "binance/BTCUSDT-1d-2022-9.json", "binance/BTCUSDT-1d-2022-10.json", "binance/BTCUSDT-1d-2022-11.json", "binance/BTCUSDT-1d-2022-12.json",
            "binance/BTCUSDT-1d-2023-1.json", "binance/BTCUSDT-1d-2023-2.json", "binance/BTCUSDT-1d-2023-3.json", "binance/BTCUSDT-1d-2023-4.json", "binance/BTCUSDT-1d-2023-5.json", "binance/BTCUSDT-1d-2023-6.json", "binance/BTCUSDT-1d-2023-7.json", "binance/BTCUSDT-1d-2023-8.json", "binance/BTCUSDT-1d-2023-9.json", "binance/BTCUSDT-1d-2023-10.json", "binance/BTCUSDT-1d-2023-11.json", "binance/BTCUSDT-1d-2023-12.json"
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1h-2024-1.json", 
         "binance/BTCUSDT-1h-2024-2.json", 
        "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"
         ],
        ["binance/BTCUSDT-1d-2024-1.json", 
         "binance/BTCUSDT-1d-2024-2.json", 
        "binance/BTCUSDT-1d-2024-3.json", "binance/BTCUSDT-1d-2024-4.json"
         ],
    ],
    fidelity = "1h",
    layers = ["1h", "1d"]
)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32)
data_2017_to_2023vs2024_1h_1d_only_price_percent_sin_volume_32 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type="only_price_percent_sin_volume", lookback_window_size=32)

data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators1 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week", indicator='indicators1')
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators2 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week", indicator='indicators2')
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators3 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week", indicator='indicators3')
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators4 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week", indicator='indicators4')
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators5 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week", indicator='indicators5')
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators6 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week", indicator='indicators6')
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators7 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week", indicator='indicators7')


data_2017_to_2023vs2024_1h_1d_only_price_percent_32_ETH = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=[[item.replace("BTC", "ETH") for item in row] for row in data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra], test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=[[item.replace("BTC", "ETH") for item in row] for row in data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra], layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32)

data_2017_to_2023vs2024_1h_1d_only_price_percent_32_XRP = DataConfig(
    id= "data_2017_to_2023vs2024_1h_1d_only_price_percent_32_XRP",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        [
            "binance/XRPUSDT-1h-2018-5.json", "binance/XRPUSDT-1h-2018-6.json", "binance/XRPUSDT-1h-2018-7.json", "binance/XRPUSDT-1h-2018-8.json", "binance/XRPUSDT-1h-2018-9.json", "binance/XRPUSDT-1h-2018-10.json", "binance/XRPUSDT-1h-2018-11.json", "binance/XRPUSDT-1h-2018-12.json",               
            "binance/XRPUSDT-1h-2019-1.json", "binance/XRPUSDT-1h-2019-2.json", "binance/XRPUSDT-1h-2019-3.json", "binance/XRPUSDT-1h-2019-4.json", "binance/XRPUSDT-1h-2019-5.json", "binance/XRPUSDT-1h-2019-6.json", "binance/XRPUSDT-1h-2019-7.json", "binance/XRPUSDT-1h-2019-8.json", "binance/XRPUSDT-1h-2019-9.json", "binance/XRPUSDT-1h-2019-10.json", "binance/XRPUSDT-1h-2019-11.json", "binance/XRPUSDT-1h-2019-12.json",               
            "binance/XRPUSDT-1h-2020-1.json", "binance/XRPUSDT-1h-2020-2.json", "binance/XRPUSDT-1h-2020-3.json", "binance/XRPUSDT-1h-2020-4.json", "binance/XRPUSDT-1h-2020-5.json", "binance/XRPUSDT-1h-2020-6.json", "binance/XRPUSDT-1h-2020-7.json", "binance/XRPUSDT-1h-2020-8.json", "binance/XRPUSDT-1h-2020-9.json", "binance/XRPUSDT-1h-2020-10.json", "binance/XRPUSDT-1h-2020-11.json", "binance/XRPUSDT-1h-2020-12.json",               
            "binance/XRPUSDT-1h-2021-1.json", "binance/XRPUSDT-1h-2021-2.json", "binance/XRPUSDT-1h-2021-3.json", "binance/XRPUSDT-1h-2021-4.json", "binance/XRPUSDT-1h-2021-5.json", "binance/XRPUSDT-1h-2021-6.json", "binance/XRPUSDT-1h-2021-7.json", "binance/XRPUSDT-1h-2021-8.json", "binance/XRPUSDT-1h-2021-9.json", "binance/XRPUSDT-1h-2021-10.json", "binance/XRPUSDT-1h-2021-11.json", "binance/XRPUSDT-1h-2021-12.json",
            "binance/XRPUSDT-1h-2022-1.json", "binance/XRPUSDT-1h-2022-2.json", "binance/XRPUSDT-1h-2022-3.json", "binance/XRPUSDT-1h-2022-4.json", "binance/XRPUSDT-1h-2022-5.json", "binance/XRPUSDT-1h-2022-6.json", "binance/XRPUSDT-1h-2022-7.json", "binance/XRPUSDT-1h-2022-8.json", "binance/XRPUSDT-1h-2022-9.json", "binance/XRPUSDT-1h-2022-10.json", "binance/XRPUSDT-1h-2022-11.json", "binance/XRPUSDT-1h-2022-12.json",
            "binance/XRPUSDT-1h-2023-1.json", "binance/XRPUSDT-1h-2023-2.json", "binance/XRPUSDT-1h-2023-3.json", "binance/XRPUSDT-1h-2023-4.json", "binance/XRPUSDT-1h-2023-5.json", "binance/XRPUSDT-1h-2023-6.json", "binance/XRPUSDT-1h-2023-7.json", "binance/XRPUSDT-1h-2023-8.json", "binance/XRPUSDT-1h-2023-9.json", "binance/XRPUSDT-1h-2023-10.json", "binance/XRPUSDT-1h-2023-11.json", "binance/XRPUSDT-1h-2023-12.json"
        ],
        [          
            "binance/XRPUSDT-1d-2018-5.json", "binance/XRPUSDT-1d-2018-6.json", "binance/XRPUSDT-1d-2018-7.json", "binance/XRPUSDT-1d-2018-8.json", "binance/XRPUSDT-1d-2018-9.json", "binance/XRPUSDT-1d-2018-10.json", "binance/XRPUSDT-1d-2018-11.json", "binance/XRPUSDT-1d-2018-12.json",               
            "binance/XRPUSDT-1d-2019-1.json", "binance/XRPUSDT-1d-2019-2.json", "binance/XRPUSDT-1d-2019-3.json", "binance/XRPUSDT-1d-2019-4.json", "binance/XRPUSDT-1d-2019-5.json", "binance/XRPUSDT-1d-2019-6.json", "binance/XRPUSDT-1d-2019-7.json", "binance/XRPUSDT-1d-2019-8.json", "binance/XRPUSDT-1d-2019-9.json", "binance/XRPUSDT-1d-2019-10.json", "binance/XRPUSDT-1d-2019-11.json", "binance/XRPUSDT-1d-2019-12.json",               
            "binance/XRPUSDT-1d-2020-1.json", "binance/XRPUSDT-1d-2020-2.json", "binance/XRPUSDT-1d-2020-3.json", "binance/XRPUSDT-1d-2020-4.json", "binance/XRPUSDT-1d-2020-5.json", "binance/XRPUSDT-1d-2020-6.json", "binance/XRPUSDT-1d-2020-7.json", "binance/XRPUSDT-1d-2020-8.json", "binance/XRPUSDT-1d-2020-9.json", "binance/XRPUSDT-1d-2020-10.json", "binance/XRPUSDT-1d-2020-11.json", "binance/XRPUSDT-1d-2020-12.json",               
            "binance/XRPUSDT-1d-2021-1.json", "binance/XRPUSDT-1d-2021-2.json", "binance/XRPUSDT-1d-2021-3.json", "binance/XRPUSDT-1d-2021-4.json", "binance/XRPUSDT-1d-2021-5.json", "binance/XRPUSDT-1d-2021-6.json", "binance/XRPUSDT-1d-2021-7.json", "binance/XRPUSDT-1d-2021-8.json", "binance/XRPUSDT-1d-2021-9.json", "binance/XRPUSDT-1d-2021-10.json", "binance/XRPUSDT-1d-2021-11.json", "binance/XRPUSDT-1d-2021-12.json",
            "binance/XRPUSDT-1d-2022-1.json", "binance/XRPUSDT-1d-2022-2.json", "binance/XRPUSDT-1d-2022-3.json", "binance/XRPUSDT-1d-2022-4.json", "binance/XRPUSDT-1d-2022-5.json", "binance/XRPUSDT-1d-2022-6.json", "binance/XRPUSDT-1d-2022-7.json", "binance/XRPUSDT-1d-2022-8.json", "binance/XRPUSDT-1d-2022-9.json", "binance/XRPUSDT-1d-2022-10.json", "binance/XRPUSDT-1d-2022-11.json", "binance/XRPUSDT-1d-2022-12.json",
            "binance/XRPUSDT-1d-2023-1.json", "binance/XRPUSDT-1d-2023-2.json", "binance/XRPUSDT-1d-2023-3.json", "binance/XRPUSDT-1d-2023-4.json", "binance/XRPUSDT-1d-2023-5.json", "binance/XRPUSDT-1d-2023-6.json", "binance/XRPUSDT-1d-2023-7.json", "binance/XRPUSDT-1d-2023-8.json", "binance/XRPUSDT-1d-2023-9.json", "binance/XRPUSDT-1d-2023-10.json", "binance/XRPUSDT-1d-2023-11.json", "binance/XRPUSDT-1d-2023-12.json"
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/XRPUSDT-1h-2024-1.json", 
         "binance/XRPUSDT-1h-2024-2.json", 
        "binance/XRPUSDT-1h-2024-3.json", "binance/XRPUSDT-1h-2024-4.json"
         ],
        ["binance/XRPUSDT-1d-2024-1.json", 
         "binance/XRPUSDT-1d-2024-2.json", 
        "binance/XRPUSDT-1d-2024-3.json", "binance/XRPUSDT-1d-2024-4.json"
         ],
    ],
    fidelity = "1h",
    layers = ["1h", "1d"],
    lookback_window_size=32
)

data_2017_to_2023vs2024_1h_1d_only_price_percent_32_DOGE = DataConfig(
    id= "data_2017_to_2023vs2024_1h_1d_only_price_percent_32_DOGE",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        [
            "binance/DOGEUSDT-1h-2019-7.json", "binance/DOGEUSDT-1h-2019-8.json", "binance/DOGEUSDT-1h-2019-9.json", "binance/DOGEUSDT-1h-2019-10.json", "binance/DOGEUSDT-1h-2019-11.json", "binance/DOGEUSDT-1h-2019-12.json",               
            "binance/DOGEUSDT-1h-2020-1.json", "binance/DOGEUSDT-1h-2020-2.json", "binance/DOGEUSDT-1h-2020-3.json", "binance/DOGEUSDT-1h-2020-4.json", "binance/DOGEUSDT-1h-2020-5.json", "binance/DOGEUSDT-1h-2020-6.json", "binance/DOGEUSDT-1h-2020-7.json", "binance/DOGEUSDT-1h-2020-8.json", "binance/DOGEUSDT-1h-2020-9.json", "binance/DOGEUSDT-1h-2020-10.json", "binance/DOGEUSDT-1h-2020-11.json", "binance/DOGEUSDT-1h-2020-12.json",               
            "binance/DOGEUSDT-1h-2021-1.json", "binance/DOGEUSDT-1h-2021-2.json", "binance/DOGEUSDT-1h-2021-3.json", "binance/DOGEUSDT-1h-2021-4.json", "binance/DOGEUSDT-1h-2021-5.json", "binance/DOGEUSDT-1h-2021-6.json", "binance/DOGEUSDT-1h-2021-7.json", "binance/DOGEUSDT-1h-2021-8.json", "binance/DOGEUSDT-1h-2021-9.json", "binance/DOGEUSDT-1h-2021-10.json", "binance/DOGEUSDT-1h-2021-11.json", "binance/DOGEUSDT-1h-2021-12.json",
            "binance/DOGEUSDT-1h-2022-1.json", "binance/DOGEUSDT-1h-2022-2.json", "binance/DOGEUSDT-1h-2022-3.json", "binance/DOGEUSDT-1h-2022-4.json", "binance/DOGEUSDT-1h-2022-5.json", "binance/DOGEUSDT-1h-2022-6.json", "binance/DOGEUSDT-1h-2022-7.json", "binance/DOGEUSDT-1h-2022-8.json", "binance/DOGEUSDT-1h-2022-9.json", "binance/DOGEUSDT-1h-2022-10.json", "binance/DOGEUSDT-1h-2022-11.json", "binance/DOGEUSDT-1h-2022-12.json",
            "binance/DOGEUSDT-1h-2023-1.json", "binance/DOGEUSDT-1h-2023-2.json", "binance/DOGEUSDT-1h-2023-3.json", "binance/DOGEUSDT-1h-2023-4.json", "binance/DOGEUSDT-1h-2023-5.json", "binance/DOGEUSDT-1h-2023-6.json", "binance/DOGEUSDT-1h-2023-7.json", "binance/DOGEUSDT-1h-2023-8.json", "binance/DOGEUSDT-1h-2023-9.json", "binance/DOGEUSDT-1h-2023-10.json", "binance/DOGEUSDT-1h-2023-11.json", "binance/DOGEUSDT-1h-2023-12.json"
        ],
        [          
            "binance/DOGEUSDT-1d-2019-7.json", "binance/DOGEUSDT-1d-2019-8.json", "binance/DOGEUSDT-1d-2019-9.json", "binance/DOGEUSDT-1d-2019-10.json", "binance/DOGEUSDT-1d-2019-11.json", "binance/DOGEUSDT-1d-2019-12.json",               
            "binance/DOGEUSDT-1d-2020-1.json", "binance/DOGEUSDT-1d-2020-2.json", "binance/DOGEUSDT-1d-2020-3.json", "binance/DOGEUSDT-1d-2020-4.json", "binance/DOGEUSDT-1d-2020-5.json", "binance/DOGEUSDT-1d-2020-6.json", "binance/DOGEUSDT-1d-2020-7.json", "binance/DOGEUSDT-1d-2020-8.json", "binance/DOGEUSDT-1d-2020-9.json", "binance/DOGEUSDT-1d-2020-10.json", "binance/DOGEUSDT-1d-2020-11.json", "binance/DOGEUSDT-1d-2020-12.json",               
            "binance/DOGEUSDT-1d-2021-1.json", "binance/DOGEUSDT-1d-2021-2.json", "binance/DOGEUSDT-1d-2021-3.json", "binance/DOGEUSDT-1d-2021-4.json", "binance/DOGEUSDT-1d-2021-5.json", "binance/DOGEUSDT-1d-2021-6.json", "binance/DOGEUSDT-1d-2021-7.json", "binance/DOGEUSDT-1d-2021-8.json", "binance/DOGEUSDT-1d-2021-9.json", "binance/DOGEUSDT-1d-2021-10.json", "binance/DOGEUSDT-1d-2021-11.json", "binance/DOGEUSDT-1d-2021-12.json",
            "binance/DOGEUSDT-1d-2022-1.json", "binance/DOGEUSDT-1d-2022-2.json", "binance/DOGEUSDT-1d-2022-3.json", "binance/DOGEUSDT-1d-2022-4.json", "binance/DOGEUSDT-1d-2022-5.json", "binance/DOGEUSDT-1d-2022-6.json", "binance/DOGEUSDT-1d-2022-7.json", "binance/DOGEUSDT-1d-2022-8.json", "binance/DOGEUSDT-1d-2022-9.json", "binance/DOGEUSDT-1d-2022-10.json", "binance/DOGEUSDT-1d-2022-11.json", "binance/DOGEUSDT-1d-2022-12.json",
            "binance/DOGEUSDT-1d-2023-1.json", "binance/DOGEUSDT-1d-2023-2.json", "binance/DOGEUSDT-1d-2023-3.json", "binance/DOGEUSDT-1d-2023-4.json", "binance/DOGEUSDT-1d-2023-5.json", "binance/DOGEUSDT-1d-2023-6.json", "binance/DOGEUSDT-1d-2023-7.json", "binance/DOGEUSDT-1d-2023-8.json", "binance/DOGEUSDT-1d-2023-9.json", "binance/DOGEUSDT-1d-2023-10.json", "binance/DOGEUSDT-1d-2023-11.json", "binance/DOGEUSDT-1d-2023-12.json"
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/DOGEUSDT-1h-2024-1.json", 
         "binance/DOGEUSDT-1h-2024-2.json", 
        "binance/DOGEUSDT-1h-2024-3.json", "binance/DOGEUSDT-1h-2024-4.json"
         ],
        ["binance/DOGEUSDT-1d-2024-1.json", 
         "binance/DOGEUSDT-1d-2024-2.json", 
        "binance/DOGEUSDT-1d-2024-3.json", "binance/DOGEUSDT-1d-2024-4.json"
         ],
    ],
    fidelity = "1h",
    layers = ["1h", "1d"],
    lookback_window_size=32
)

data_2017_to_2023vs2024_1d_only_price_percent = DataConfig(
    id= "data_2017_to_2023vs2024_1d_only_price_percent",
    type="only_price_percent",
    train_data_paths= [
            "binance/BTCUSDT-1d-2017-8.json", "binance/BTCUSDT-1d-2017-9.json", "binance/BTCUSDT-1d-2017-10.json", "binance/BTCUSDT-1d-2017-11.json", "binance/BTCUSDT-1d-2017-12.json",               
            "binance/BTCUSDT-1d-2018-1.json", "binance/BTCUSDT-1d-2018-2.json", "binance/BTCUSDT-1d-2018-3.json", "binance/BTCUSDT-1d-2018-4.json", "binance/BTCUSDT-1d-2018-5.json", "binance/BTCUSDT-1d-2018-6.json", "binance/BTCUSDT-1d-2018-7.json", "binance/BTCUSDT-1d-2018-8.json", "binance/BTCUSDT-1d-2018-9.json", "binance/BTCUSDT-1d-2018-10.json", "binance/BTCUSDT-1d-2018-11.json", "binance/BTCUSDT-1d-2018-12.json",               
            "binance/BTCUSDT-1d-2019-1.json", "binance/BTCUSDT-1d-2019-2.json", "binance/BTCUSDT-1d-2019-3.json", "binance/BTCUSDT-1d-2019-4.json", "binance/BTCUSDT-1d-2019-5.json", "binance/BTCUSDT-1d-2019-6.json", "binance/BTCUSDT-1d-2019-7.json", "binance/BTCUSDT-1d-2019-8.json", "binance/BTCUSDT-1d-2019-9.json", "binance/BTCUSDT-1d-2019-10.json", "binance/BTCUSDT-1d-2019-11.json", "binance/BTCUSDT-1d-2019-12.json",               
            "binance/BTCUSDT-1d-2020-1.json", "binance/BTCUSDT-1d-2020-2.json", "binance/BTCUSDT-1d-2020-3.json", "binance/BTCUSDT-1d-2020-4.json", "binance/BTCUSDT-1d-2020-5.json", "binance/BTCUSDT-1d-2020-6.json", "binance/BTCUSDT-1d-2020-7.json", "binance/BTCUSDT-1d-2020-8.json", "binance/BTCUSDT-1d-2020-9.json", "binance/BTCUSDT-1d-2020-10.json", "binance/BTCUSDT-1d-2020-11.json", "binance/BTCUSDT-1d-2020-12.json",               
            "binance/BTCUSDT-1d-2021-1.json", "binance/BTCUSDT-1d-2021-2.json", "binance/BTCUSDT-1d-2021-3.json", "binance/BTCUSDT-1d-2021-4.json", "binance/BTCUSDT-1d-2021-5.json", "binance/BTCUSDT-1d-2021-6.json", "binance/BTCUSDT-1d-2021-7.json", "binance/BTCUSDT-1d-2021-8.json", "binance/BTCUSDT-1d-2021-9.json", "binance/BTCUSDT-1d-2021-10.json", "binance/BTCUSDT-1d-2021-11.json", "binance/BTCUSDT-1d-2021-12.json",
            "binance/BTCUSDT-1d-2022-1.json", "binance/BTCUSDT-1d-2022-2.json", "binance/BTCUSDT-1d-2022-3.json", "binance/BTCUSDT-1d-2022-4.json", "binance/BTCUSDT-1d-2022-5.json", "binance/BTCUSDT-1d-2022-6.json", "binance/BTCUSDT-1d-2022-7.json", "binance/BTCUSDT-1d-2022-8.json", "binance/BTCUSDT-1d-2022-9.json", "binance/BTCUSDT-1d-2022-10.json", "binance/BTCUSDT-1d-2022-11.json", "binance/BTCUSDT-1d-2022-12.json",
            "binance/BTCUSDT-1d-2023-1.json", "binance/BTCUSDT-1d-2023-2.json", "binance/BTCUSDT-1d-2023-3.json", "binance/BTCUSDT-1d-2023-4.json", "binance/BTCUSDT-1d-2023-5.json", "binance/BTCUSDT-1d-2023-6.json", "binance/BTCUSDT-1d-2023-7.json", "binance/BTCUSDT-1d-2023-8.json", "binance/BTCUSDT-1d-2023-9.json", "binance/BTCUSDT-1d-2023-10.json", "binance/BTCUSDT-1d-2023-11.json", "binance/BTCUSDT-1d-2023-12.json"
    ],
    test_data_paths= [
        "binance/BTCUSDT-1d-2024-1.json", 
         "binance/BTCUSDT-1d-2024-2.json", 
        "binance/BTCUSDT-1d-2024-3.json", "binance/BTCUSDT-1d-2024-4.json"
    ],
    fidelity = "1d",
    layers = ["1d"]
)
data_2017_to_2023vs2024_1d_only_price_percent_32 = DataConfig(id=data_2017_to_2023vs2024_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1d_only_price_percent.type, lookback_window_size=32)

# check if getting price_high/price_low as a percent difference from price not its previous step makes a difference
# check check price + vol, 
def get_datas_1h_1d():
    # return [data_2017_to_2023vs2024_1h_1d_only_price_percent] 
    # return [data_2017_to_2023vs2024_1h_1d_only_price_percent_32]
    return [data_2017_to_2023vs2024_1h_1d_only_price_percent_32, data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators1, data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators2, data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators3, data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators4, data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators5, data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators6, data_2017_to_2023vs2024_1h_1d_only_price_percent_32_indicators7]
def get_datas_downtrend():
    return [data_downtrend_2021dec_2022dec_to_3]

def get_datas_1m_1h_1d():
    # return [data_2017vs2024_1m_1h_1d_only_price_percent_32, data_2017vs2024_1m_1h_1d_only_price_percent_32_none]
    return [data_2022vs2024_1m_1h_1d_only_price_percent_32]
    # return [data_2022vs2024_1m_only_price_percent_32, data_2022vs2024_1m_1h_only_price_percent_32, data_2022vs2024_1m_1h_1d_only_price_percent_32]

    