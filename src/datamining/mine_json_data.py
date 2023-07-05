import json
import pandas as pd
import time

from datamining.binance_mine import getTimeSeriesData, getIntervalTime

def getSymbolData(folder: str, symbol: str):
    f = open(f"{folder}/symbols.json")
    data = json.load(f)
    return data[symbol]

def mineData(folder: str, name: str, symbol: str, interval: str = "1m"):
    symbol_data = getSymbolData(folder, symbol)
    start_time = symbol_data["startTime"]
    end_time = time.time() * 1000

    current_time = start_time
    window_time = getIntervalTime(interval)
    interval_time = getIntervalTime(interval, 1000)
    requests_per_min = 1000

    requests = 0
    results = []
    while current_time < end_time:
        new_end_time = current_time + interval_time
        result = getTimeSeriesData(symbol, interval, current_time, new_end_time - window_time)
        results += result
        current_time = new_end_time
        requests += 1
        if requests >= requests_per_min:
            requests = 0
            print(f"DONE {requests_per_min} reqs - sleeping at {time.time()}")
            time.sleep(61)


    f = open(f"{folder}/{name}_{symbol}_{interval}.json", "w")
    f.write(json.dumps(results))
    f.close()

    return results

def preparePDData(filename: str, timestamps_as_dates: bool = False):
    df = pd.read_json(filename, convert_dates= timestamps_as_dates)
    return df

# mineData("data/raw", "binance", "BTCUSDT", "1d")
# df = preparePDData("data/raw/binance_BTCUSDT_1d.json")
# print(f"df.value_counts: {df.count()}")
