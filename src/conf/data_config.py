from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    id: str
    train_data_paths: List[str]
    test_data_paths: List[str]
    lookback_window_size: int = 1
    flat_lookback: bool = False
    type: str = "standard" # possible ["standard", "solo_price_percent", "only_price_percent", "all_percents"]
    indicator: str = "none"
    timestamp: str = "none" # possible ["none",  "expanded", "day_of_week"]
    percent_buysell: float = 0.004
    fidelity: str = "1h" # possible ["1m", "1h", "1d"]
    layers: List[str] = field(default_factory=[])  # possible ["1h"], ["1h", "4h", "1d", "7d"], ["1m", "15m", "1h", "4h", "1d", "7d"]
    flat_layers: bool = False
    train_data_paths_extra: List[List[str]] | None = None
    test_data_paths_extra: List[List[str]] | None = None
    
data_1_to_1 = DataConfig(
    id= "1_to_1",
    train_data_paths= ["binance/BTCUSDT-1h-2023-1.json"],
    test_data_paths= ["binance/BTCUSDT-1h-2023-2.json"],
    layers= ["1h"]
)
data_1_to_1_solo_price_percent = DataConfig(id=data_1_to_1.id, train_data_paths=data_1_to_1.train_data_paths, test_data_paths=data_1_to_1.test_data_paths, layers=data_1_to_1.layers, type="solo_price_percent")
data_1_to_1_only_price_percent = DataConfig(id=data_1_to_1.id, train_data_paths=data_1_to_1.train_data_paths, test_data_paths=data_1_to_1.test_data_paths, layers=data_1_to_1.layers, type="only_price_percent")
data_1_to_1_all_percents = DataConfig(id=data_1_to_1.id, train_data_paths=data_1_to_1.train_data_paths, test_data_paths=data_1_to_1.test_data_paths, layers=data_1_to_1.layers, type="all_percents")
data_1_to_1_all_percents_60 = DataConfig(id=data_1_to_1.id, train_data_paths=data_1_to_1.train_data_paths, test_data_paths=data_1_to_1.test_data_paths, layers=data_1_to_1.layers, type="all_percents", lookback_window_size=60)
data_1_to_1_only_price_multi = DataConfig(id=data_1_to_1.id, train_data_paths=data_1_to_1.train_data_paths, test_data_paths=data_1_to_1.test_data_paths, layers= ["1h", "1d"], type="only_price")

data_3_to_1 = DataConfig(
    id= "2023_1-3vs4_1h",
    train_data_paths= ["binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json"],
    test_data_paths= ["binance/BTCUSDT-1h-2023-4.json"],
    layers=["1h"]
)
data_3_to_1_solo_price_percent = DataConfig(id=data_3_to_1.id, train_data_paths=data_3_to_1.train_data_paths, test_data_paths=data_3_to_1.test_data_paths, layers=data_3_to_1.layers, type="solo_price_percent")
data_3_to_1_only_price_percent = DataConfig(id=data_3_to_1.id, train_data_paths=data_3_to_1.train_data_paths, test_data_paths=data_3_to_1.test_data_paths, layers=data_3_to_1.layers, type="only_price_percent")
data_3_to_1_all_percents = DataConfig(id=data_3_to_1.id, train_data_paths=data_3_to_1.train_data_paths, test_data_paths=data_3_to_1.test_data_paths, layers=data_3_to_1.layers, type="all_percents")

data_2023vs2024_1h = DataConfig(
    id= "2023vs2024_1h",
    train_data_paths= ["binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"],
    test_data_paths= ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
    layers=["1h"]
)
data_2023vs2024_1h_solo_price_percent = DataConfig(id=data_2023vs2024_1h.id, train_data_paths=data_2023vs2024_1h.train_data_paths, test_data_paths=data_2023vs2024_1h.test_data_paths, layers=data_2023vs2024_1h.layers, type="solo_price_percent")
data_2023vs2024_1h_only_price_percent = DataConfig(id=data_2023vs2024_1h.id, train_data_paths=data_2023vs2024_1h.train_data_paths, test_data_paths=data_2023vs2024_1h.test_data_paths, layers=data_2023vs2024_1h.layers, type="only_price_percent")
data_2023vs2024_1h_all_percents = DataConfig(id=data_2023vs2024_1h.id, train_data_paths=data_2023vs2024_1h.train_data_paths, test_data_paths=data_2023vs2024_1h.test_data_paths, layers=data_2023vs2024_1h.layers, type="all_percents")
data_2023vs2024_1h_all_percents_without_candles = DataConfig(id=data_2023vs2024_1h.id, train_data_paths=data_2023vs2024_1h.train_data_paths, test_data_paths=data_2023vs2024_1h.test_data_paths, layers=data_2023vs2024_1h.layers, type="all_percents_without_candles")

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
data_2017_to_2023vs2024_1h_solo_price_percent = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="solo_price_percent")
data_2017_to_2023vs2024_1h_only_price_percent = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="only_price_percent")
data_2017_to_2023vs2024_1h_only_price_percent_sin_volume = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="only_price_percent_sin_volume")
data_2017_to_2023vs2024_1h_only_price_percent_expanded = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="only_price_percent", timestamp="expanded")
data_2017_to_2023vs2024_1h_only_price_percent_day_of_week = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="only_price_percent", timestamp="day_of_week")
data_2017_to_2023vs2024_1h_only_price_percent_expanded_32= DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="only_price_percent", timestamp="expanded", lookback_window_size=32)
data_2017_to_2023vs2024_1h_only_price_percent_32 = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="only_price_percent", lookback_window_size=32)
data_2017_to_2023vs2024_1h_all_percents = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="all_percents")
data_2017_to_2023vs2024_1h_all_percents_32 = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="all_percents", lookback_window_size=32)
data_2017_to_2023vs2024_1h_all_percents_without_candles = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="all_percents_without_candles")
data_2017_to_2023vs2024_1h_all_percents_without_candles_32 = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers=data_2017_to_2023vs2024_1h.layers, type="all_percents_without_candles", lookback_window_size=32)
data_2017_to_2023vs2024_1h_only_price_percent_multi = DataConfig(id=data_2017_to_2023vs2024_1h.id, train_data_paths=data_2017_to_2023vs2024_1h.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1h.test_data_paths, layers= ["1h", "1d"], type="only_price_percent")

data_downtrend_2021dec_2022dec_to_3 = DataConfig(
    id= "downtrend_2021dec_2022dec_to_3_1h",
    train_data_paths= ["binance/BTCUSDT-1h-2021-12.json", "binance/BTCUSDT-1h-2022-1.json", "binance/BTCUSDT-1h-2022-2.json", "binance/BTCUSDT-1h-2022-3.json", "binance/BTCUSDT-1h-2022-4.json", "binance/BTCUSDT-1h-2022-5.json", "binance/BTCUSDT-1h-2022-6.json", "binance/BTCUSDT-1h-2022-7.json", "binance/BTCUSDT-1h-2022-8.json", "binance/BTCUSDT-1h-2022-9.json", "binance/BTCUSDT-1h-2022-10.json", "binance/BTCUSDT-1h-2022-11.json", "binance/BTCUSDT-1h-2022-12.json"],
    test_data_paths= ["binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json"],
    layers=["1h"]
)

data_1_to_1_min = DataConfig(
    id= "1_to_1_1m",
    train_data_paths= ["binance/BTCUSDT-1m-2023-1.json"],
    test_data_paths= ["binance/BTCUSDT-1m-2023-2.json"],
    fidelity = "1m",
    layers = ["1m"]
)
data_3_to_1_m = DataConfig(
    id= "3_to_1_1m",
    train_data_paths= ["binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json"],
    test_data_paths= ["binance/BTCUSDT-1m-2023-4.json"],
    fidelity = "1m",
    layers = ["1m"]
)
data_3_to_1_m_solo_price = DataConfig(id=data_3_to_1_m.id, train_data_paths=data_3_to_1_m.train_data_paths, test_data_paths=data_3_to_1_m.test_data_paths, layers=data_3_to_1_m.layers, type="solo_price")
data_3_to_1_m_only_price = DataConfig(id=data_3_to_1_m.id, train_data_paths=data_3_to_1_m.train_data_paths, test_data_paths=data_3_to_1_m.test_data_paths, layers=data_3_to_1_m.layers, type="only_price")
data_3_to_1_m_solo_price_percent = DataConfig(id=data_3_to_1_m.id, train_data_paths=data_3_to_1_m.train_data_paths, test_data_paths=data_3_to_1_m.test_data_paths, layers=data_3_to_1_m.layers, type="solo_price_percent")
data_3_to_1_m_only_price_percent = DataConfig(id=data_3_to_1_m.id, train_data_paths=data_3_to_1_m.train_data_paths, test_data_paths=data_3_to_1_m.test_data_paths, layers=data_3_to_1_m.layers, type="only_price_percent")
data_3_to_1_m_all_percents = DataConfig(id=data_3_to_1_m.id, train_data_paths=data_3_to_1_m.train_data_paths, test_data_paths=data_3_to_1_m.test_data_paths, layers=data_3_to_1_m.layers, type="all_percents")
data_3_to_1_m_all_percents_without_candles = DataConfig(id=data_3_to_1_m.id, train_data_paths=data_3_to_1_m.train_data_paths, test_data_paths=data_3_to_1_m.test_data_paths, layers=data_3_to_1_m.layers, type="all_percents_without_candles")

data_2023vs2024_1m = DataConfig(
    id= "2023vs2024_1m",
    train_data_paths= ["binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"],
    test_data_paths= ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
    fidelity = "1m",
    layers = ["1m"]
)
data_2023vs2024_1m_solo_price = DataConfig(id=data_2023vs2024_1m.id, train_data_paths=data_2023vs2024_1m.train_data_paths, test_data_paths=data_2023vs2024_1m.test_data_paths, layers=data_2023vs2024_1m.layers, fidelity=data_2023vs2024_1m.fidelity, type="solo_price")
data_2023vs2024_1m_only_price = DataConfig(id=data_2023vs2024_1m.id, train_data_paths=data_2023vs2024_1m.train_data_paths, test_data_paths=data_2023vs2024_1m.test_data_paths, layers=data_2023vs2024_1m.layers, fidelity=data_2023vs2024_1m.fidelity, type="only_price")
data_2023vs2024_1m_solo_price_percent = DataConfig(id=data_2023vs2024_1m.id, train_data_paths=data_2023vs2024_1m.train_data_paths, test_data_paths=data_2023vs2024_1m.test_data_paths, layers=data_2023vs2024_1m.layers, fidelity=data_2023vs2024_1m.fidelity, type="solo_price_percent")
data_2023vs2024_1m_only_price_percent = DataConfig(id=data_2023vs2024_1m.id, train_data_paths=data_2023vs2024_1m.train_data_paths, test_data_paths=data_2023vs2024_1m.test_data_paths, layers=data_2023vs2024_1m.layers, fidelity=data_2023vs2024_1m.fidelity, type="only_price_percent")
data_2023vs2024_1m_only_price_percent_60 = DataConfig(id=data_2023vs2024_1m.id, train_data_paths=data_2023vs2024_1m.train_data_paths, test_data_paths=data_2023vs2024_1m.test_data_paths, layers=data_2023vs2024_1m.layers, fidelity=data_2023vs2024_1m.fidelity, type="only_price_percent", lookback_window_size=60)
data_2023vs2024_1m_all_percents = DataConfig(id=data_2023vs2024_1m.id, train_data_paths=data_2023vs2024_1m.train_data_paths, test_data_paths=data_2023vs2024_1m.test_data_paths, layers=data_2023vs2024_1m.layers, fidelity=data_2023vs2024_1m.fidelity, type="all_percents")
data_2023vs2024_1m_all_percents_without_candles = DataConfig(id=data_2023vs2024_1m.id, train_data_paths=data_2023vs2024_1m.train_data_paths, test_data_paths=data_2023vs2024_1m.test_data_paths, layers=data_2023vs2024_1m.layers, fidelity=data_2023vs2024_1m.fidelity, type="all_percents_without_candles")

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
data_2020_to_2023vs2024_1m_only_price_percent = DataConfig(id=data_2020_to_2023vs2024_1m.id, train_data_paths=data_2020_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2020_to_2023vs2024_1m.test_data_paths, layers=data_2020_to_2023vs2024_1m.layers, fidelity=data_2020_to_2023vs2024_1m.fidelity, type="only_price_percent")
data_2020_to_2023vs2024_1m_only_price_percent_expanded = DataConfig(id=data_2020_to_2023vs2024_1m.id, train_data_paths=data_2020_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2020_to_2023vs2024_1m.test_data_paths, layers=data_2020_to_2023vs2024_1m.layers, fidelity=data_2020_to_2023vs2024_1m.fidelity, type="only_price_percent", timestamp= "expanded")
data_2020_to_2023vs2024_1m_only_price_percent_day_of_week = DataConfig(id=data_2020_to_2023vs2024_1m.id, train_data_paths=data_2020_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2020_to_2023vs2024_1m.test_data_paths, layers=data_2020_to_2023vs2024_1m.layers, fidelity=data_2020_to_2023vs2024_1m.fidelity, type="only_price_percent", timestamp= "day_of_week")
data_2020_to_2023vs2024_1m_only_price_percent_60 = DataConfig(id=data_2020_to_2023vs2024_1m.id, train_data_paths=data_2020_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2020_to_2023vs2024_1m.test_data_paths, layers=data_2020_to_2023vs2024_1m.layers, fidelity=data_2020_to_2023vs2024_1m.fidelity, type="only_price_percent", lookback_window_size=60)
data_2020_to_2023vs2024_1m_only_price_percent_multi = DataConfig(id=data_2020_to_2023vs2024_1m.id, train_data_paths=data_2020_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2020_to_2023vs2024_1m.test_data_paths, layers= ["1m", "1h"], fidelity=data_2020_to_2023vs2024_1m.fidelity, type="only_price_percent")


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
data_2017_to_2023vs2024_1m_only_price_percent = DataConfig(id=data_2017_to_2023vs2024_1m.id, train_data_paths=data_2017_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1m.test_data_paths, layers=data_2017_to_2023vs2024_1m.layers, fidelity=data_2017_to_2023vs2024_1m.fidelity, type="only_price_percent")
data_2017_to_2023vs2024_1m_only_price_percent_expanded = DataConfig(id=data_2017_to_2023vs2024_1m.id, train_data_paths=data_2017_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1m.test_data_paths, layers=data_2017_to_2023vs2024_1m.layers, fidelity=data_2017_to_2023vs2024_1m.fidelity, type="only_price_percent", timestamp= "expanded")
data_2017_to_2023vs2024_1m_only_price_percent_day_of_week = DataConfig(id=data_2017_to_2023vs2024_1m.id, train_data_paths=data_2017_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1m.test_data_paths, layers=data_2017_to_2023vs2024_1m.layers, fidelity=data_2017_to_2023vs2024_1m.fidelity, type="only_price_percent", timestamp= "day_of_week")
data_2017_to_2023vs2024_1m_only_price_percent_60 = DataConfig(id=data_2017_to_2023vs2024_1m.id, train_data_paths=data_2017_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1m.test_data_paths, layers=data_2017_to_2023vs2024_1m.layers, fidelity=data_2017_to_2023vs2024_1m.fidelity, type="only_price_percent", lookback_window_size=60)
data_2017_to_2023vs2024_1m_only_price_percent_multi = DataConfig(id=data_2017_to_2023vs2024_1m.id, train_data_paths=data_2017_to_2023vs2024_1m.train_data_paths, test_data_paths=data_2017_to_2023vs2024_1m.test_data_paths, layers= ["1m", "1h"], fidelity=data_2017_to_2023vs2024_1m.fidelity, type="only_price_percent")

data_3_to_1_m_h = DataConfig(
    id= "3_to_1_1m1h",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        ["binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json"]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2023-4.json"]
    ],
    fidelity = "1m",
    layers = ["1m", "1h"]
)
data_2023vs2024_1m_1h_only_price_percent = DataConfig(
    id= "2023vs2024_1m_1h_only_price_percent",
    type="only_price_percent",
    train_data_paths= [],
    train_data_paths_extra= [
        ["binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"],
        ["binance/BTCUSDT-1h-2023-1.json", "binance/BTCUSDT-1h-2023-2.json", "binance/BTCUSDT-1h-2023-3.json", "binance/BTCUSDT-1h-2023-4.json", "binance/BTCUSDT-1h-2023-5.json", "binance/BTCUSDT-1h-2023-6.json", "binance/BTCUSDT-1h-2023-7.json", "binance/BTCUSDT-1h-2023-8.json", "binance/BTCUSDT-1h-2023-9.json", "binance/BTCUSDT-1h-2023-10.json", "binance/BTCUSDT-1h-2023-11.json", "binance/BTCUSDT-1h-2023-12.json"]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
      ],
    fidelity = "1m",
    layers = ["1m", "1h"]
)
data_2023vs2024_1m_1h_only_price_percent_lookback32 = DataConfig(id=data_2023vs2024_1m_1h_only_price_percent.id, train_data_paths=data_2023vs2024_1m_1h_only_price_percent.train_data_paths, train_data_paths_extra=data_2023vs2024_1m_1h_only_price_percent.train_data_paths_extra, test_data_paths=data_2023vs2024_1m_1h_only_price_percent.test_data_paths, test_data_paths_extra=data_2023vs2024_1m_1h_only_price_percent.test_data_paths_extra, layers=data_2023vs2024_1m_1h_only_price_percent.layers, fidelity=data_2023vs2024_1m_1h_only_price_percent.fidelity, type=data_2023vs2024_1m_1h_only_price_percent.type, lookback_window_size=32)
data_2023vs2024_1m_1h_only_price_percent_lookback32_day_of_week = DataConfig(id=data_2023vs2024_1m_1h_only_price_percent.id, train_data_paths=data_2023vs2024_1m_1h_only_price_percent.train_data_paths, train_data_paths_extra=data_2023vs2024_1m_1h_only_price_percent.train_data_paths_extra, test_data_paths=data_2023vs2024_1m_1h_only_price_percent.test_data_paths, test_data_paths_extra=data_2023vs2024_1m_1h_only_price_percent.test_data_paths_extra, layers=data_2023vs2024_1m_1h_only_price_percent.layers, fidelity=data_2023vs2024_1m_1h_only_price_percent.fidelity, type=data_2023vs2024_1m_1h_only_price_percent.type, lookback_window_size=32, timestamp="day_of_week")


data_2022vs2024_1m_only_price_percent = DataConfig(
    id= "2022vs2024_1m_1h_only_price_percent",
    type="only_price_percent",
    train_data_paths= [
            "binance/BTCUSDT-1m-2022-1.json", "binance/BTCUSDT-1m-2022-2.json", "binance/BTCUSDT-1m-2022-3.json", "binance/BTCUSDT-1m-2022-4.json", "binance/BTCUSDT-1m-2022-5.json", "binance/BTCUSDT-1m-2022-6.json", "binance/BTCUSDT-1m-2022-7.json", "binance/BTCUSDT-1m-2022-8.json", "binance/BTCUSDT-1m-2022-9.json", "binance/BTCUSDT-1m-2022-10.json", "binance/BTCUSDT-1m-2022-11.json", "binance/BTCUSDT-1m-2022-12.json",
            "binance/BTCUSDT-1m-2023-1.json", "binance/BTCUSDT-1m-2023-2.json", "binance/BTCUSDT-1m-2023-3.json", "binance/BTCUSDT-1m-2023-4.json", "binance/BTCUSDT-1m-2023-5.json", "binance/BTCUSDT-1m-2023-6.json", "binance/BTCUSDT-1m-2023-7.json", "binance/BTCUSDT-1m-2023-8.json", "binance/BTCUSDT-1m-2023-9.json", "binance/BTCUSDT-1m-2023-10.json", "binance/BTCUSDT-1m-2023-11.json", "binance/BTCUSDT-1m-2023-12.json"
    ],
    train_data_paths_extra= [],
    test_data_paths= [
        "binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"
    ],
    test_data_paths_extra= [],
    fidelity = "1m",
    layers = ["1m"]
)
data_2022vs2024_1m_only_price_percent_32 = DataConfig(id=data_2022vs2024_1m_only_price_percent.id, train_data_paths=data_2022vs2024_1m_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_only_price_percent.layers, fidelity=data_2022vs2024_1m_only_price_percent.fidelity, type=data_2022vs2024_1m_only_price_percent.type, lookback_window_size=32)

data_2022vs2024_1m_1h_only_price_percent = DataConfig(
    id= "2022vs2024_1m_1h_only_price_percent",
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
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
      ],
    fidelity = "1m",
    layers = ["1m", "1h"]
)
data_2022vs2024_1m_1h_only_price_percent_32 = DataConfig(id=data_2022vs2024_1m_1h_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_only_price_percent.type, lookback_window_size=32)

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
data_2022vs2024_1m_1h_1d_only_price_percent_day_of_week = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, timestamp="day_of_week")
data_2022vs2024_1m_1h_1d_only_price_percent_expanded = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, timestamp="expanded")
data_2022vs2024_1m_1h_1d_only_price_percent_32 = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=32)
data_2022vs2024_1m_1h_1d_only_price_percent_32_day_of_week = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, timestamp="day_of_week", lookback_window_size=32)
data_2022vs2024_1m_1h_1d_only_price_percent_32_expanded = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, timestamp="expanded", lookback_window_size=32)
data_2022vs2024_1m_1h_1d_only_price_percent_32_Flat = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent_32.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent_32.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent_32.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent_32.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent_32.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent_32.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent_32.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent_32.type, lookback_window_size=32, flat_lookback=True)
data_2022vs2024_1m_1h_1d_only_price_percent_32_FlatFlatlayers = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent_32.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent_32.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent_32.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent_32.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent_32.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent_32.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent_32.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent_32.type, lookback_window_size=32, flat_lookback=True, flat_layers=True)
data_2022vs2024_1m_1h_1d_only_price_percent_32_Flatlayers = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent_32.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent_32.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent_32.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent_32.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent_32.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent_32.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent_32.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent_32.type, lookback_window_size=32, flat_layers=True)

data_2022vs2024_1m_1h_1d_only_price_percent_7 = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=7)
data_2022vs2024_1m_1h_1d_only_price_percent_14 = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=14)
data_2022vs2024_1m_1h_1d_only_price_percent_64 = DataConfig(id=data_2022vs2024_1m_1h_1d_only_price_percent.id, train_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2022vs2024_1m_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2022vs2024_1m_1h_1d_only_price_percent.layers, fidelity=data_2022vs2024_1m_1h_1d_only_price_percent.fidelity, type=data_2022vs2024_1m_1h_1d_only_price_percent.type, lookback_window_size=64)

data_2020_to_2023vs2024_1m_1h_only_price_percent = DataConfig(
    id= "2020_to_2023vs2024_1m_1h_only_price_percent",
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
        ]
    ],
    test_data_paths= [],
    test_data_paths_extra= [
        ["binance/BTCUSDT-1m-2024-1.json", "binance/BTCUSDT-1m-2024-2.json", "binance/BTCUSDT-1m-2024-3.json", "binance/BTCUSDT-1m-2024-4.json"],
        ["binance/BTCUSDT-1h-2024-1.json", "binance/BTCUSDT-1h-2024-2.json", "binance/BTCUSDT-1h-2024-3.json", "binance/BTCUSDT-1h-2024-4.json"],
      ],
    fidelity = "1m",
    layers = ["1m", "1h"]
)

data_2017_to_2023vs2024_1h_1d_only_price_percent = DataConfig(
    id= "2017_to_2023vs2024_1h_1d_only_price_percent",
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
data_2017_to_2023vs2024_1h_1d_only_price_percent_expanded = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, timestamp="expanded")
data_2017_to_2023vs2024_1h_1d_only_price_percent_day_of_week = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, timestamp="day_of_week")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_solo = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type="solo_price_percent", lookback_window_size=32)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_soloprices = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type="only_price_percent_sin_volume", lookback_window_size=32)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_Flat = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, flat_lookback=True)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_FlatFlatlayers = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, flat_lookback=True, flat_layers=True)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_Flatlayers = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, flat_layers=True)
data_2017_to_2023vs2024_1h_1d_only_price_percent_expanded_32 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, timestamp="expanded", lookback_window_size=32)
data_2017_to_2023vs2024_1h_1d_only_price_percent_day_of_week_32 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, timestamp="day_of_week", lookback_window_size=32)

data_2017_to_2023vs2024_1h_1d_only_price_percent_32_rsi9 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="rsi9")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_rsi30 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="rsi30")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_williams30 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="williams30")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_sma8 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="sma8")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_sma30 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="sma30")

data_2017_to_2023vs2024_1h_1d_only_price_percent_32_sar001_01 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="sar001_01")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_sar002_02 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="sar002_02")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_sar005_03 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="sar005_03")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_macdLine = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="macdLine")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_macdSignal9 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="macdSignal9")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_macd9 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="macd9")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_bollinger20Mid = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="bollinger20Mid")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_bollinger20SD = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="bollinger20SD")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_bollinger20High = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="bollinger20High")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_bollinger20Low = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="bollinger20Low")

data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman5Mean = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman5Mean")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman5Momentum = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman5Momentum")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman14Mean = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman14Mean")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman14Momentum = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman14Momentum")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman20Mean = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman20Mean")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman20Momentum = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman20Momentum")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman30Mean = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman30Mean")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_kallman30Momentum = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="kallman30Momentum")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_lwti8 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="lwti8")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_lwti13 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="lwti13")
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_lwti30 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, indicator="lwti30")



data_2017_to_2023vs2024_1h_1d_only_price_percent_32_001 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.001)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_002 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.002)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_003 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.003)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_004 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.004)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_005 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.005)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_006 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.006)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_007 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.007)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_008 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.008)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_009 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.009)
data_2017_to_2023vs2024_1h_1d_only_price_percent_32_01 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=32, percent_buysell=0.01)


data_2017_to_2023vs2024_1h_1d_only_price_percent_2 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=2)
data_2017_to_2023vs2024_1h_1d_only_price_percent_7 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=7)
data_2017_to_2023vs2024_1h_1d_only_price_percent_14 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=14)
data_2017_to_2023vs2024_1h_1d_only_price_percent_64 = DataConfig(id=data_2017_to_2023vs2024_1h_1d_only_price_percent.id, train_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths, train_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.train_data_paths_extra, test_data_paths=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths, test_data_paths_extra=data_2017_to_2023vs2024_1h_1d_only_price_percent.test_data_paths_extra, layers=data_2017_to_2023vs2024_1h_1d_only_price_percent.layers, fidelity=data_2017_to_2023vs2024_1h_1d_only_price_percent.fidelity, type=data_2017_to_2023vs2024_1h_1d_only_price_percent.type, lookback_window_size=64)

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


def get_datas_simple():
    return [data_1_to_1_only_price_percent, data_1_to_1_only_price_multi]

def get_datas_short_min():
    return [data_1_to_1_min]

def get_datas_min():
    return [data_2023vs2024_1m_only_price_percent]

def get_datas_long_min():
    return [data_2020_to_2023vs2024_1m_only_price_percent]

def get_datas_vlong_min():
    return [data_2017_to_2023vs2024_1m_only_price_percent]

def get_datas_long():
    return [data_2023vs2024_1h_only_price_percent]

# check if getting price_high/price_low as a percent difference from price not its previous step makes a difference
# check check price + vol, 
def get_datas_vlong():
    # return [data_2017_to_2023vs2024_1h_1d_only_price_percent, data_2017_to_2023vs2024_1h_1d_only_price_percent_7, data_2017_to_2023vs2024_1h_1d_only_price_percent_14, data_2017_to_2023vs2024_1h_1d_only_price_percent_32]
    return [data_2017_to_2023vs2024_1h_1d_only_price_percent_32]

def get_datas_downtrend():
    return [data_downtrend_2021dec_2022dec_to_3]

def get_datas_min_hour():
    # return [data_2022vs2024_1m_1h_1d_only_price_percent_32]
    # return [data_2022vs2024_1m_1h_1d_only_price_percent_32_day_of_week]
    return [data_2023vs2024_1m_1h_only_price_percent_lookback32, data_2023vs2024_1m_1h_only_price_percent_lookback32_day_of_week]
    # return [data_2022vs2024_1m_1h_1d_only_price_percent_64, data_2022vs2024_1m_1h_1d_only_price_percent_32, data_2022vs2024_1m_1h_1d_only_price_percent_14, data_2022vs2024_1m_1h_1d_only_price_percent_7]
    # return [data_2022vs2024_1m_only_price_percent_32, data_2022vs2024_1m_1h_only_price_percent_32, data_2022vs2024_1m_1h_1d_only_price_percent_32]
    # return [data_2022vs2024_1m_1h_1d_only_price_percent_32, data_2022vs2024_1m_1h_1d_only_price_percent_32_Flat, data_2022vs2024_1m_1h_1d_only_price_percent_32_FlatFlatlayers, data_2022vs2024_1m_1h_1d_only_price_percent_32_Flatlayers]
    # return [data_2022vs2024_1m_1h_only_price_percent, data_2022vs2024_1m_1h_1d_only_price_percent_day_of_week, data_2022vs2024_1m_1h_1d_only_price_percent_expanded, data_2022vs2024_1m_1h_1d_only_price_percent_32, data_2022vs2024_1m_1h_1d_only_price_percent_32_day_of_week, data_2022vs2024_1m_1h_1d_only_price_percent_32_expanded]

    