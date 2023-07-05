import requests

API_URL = "https://api.binance.com/api/v3/klines"
API_URL_ALT = "https://data-api.binance.vision/api/v3/klines"
INTERVAL_1S = 1000
INTERVALS = {
    "1s": INTERVAL_1S,
    "1m": INTERVAL_1S * 60,
    "1h": INTERVAL_1S * 60 * 60,
    "1d": INTERVAL_1S * 60 * 60 * 24
}

def getIntervalTime(interval: str, multiplier: int = 1):
    return INTERVALS[interval] * multiplier

def getTimeSeriesData(symbol: str, interval: str, start_time: int, end_time: int):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }
    json = requests.get(API_URL, params=params).json()
    klines = processKlines(json)
    return klines

def processKlines(json):
    klines = []
    for vals in json:
        klines.append({
            "timestamp_open": vals[0],
            "price_open": vals[1],
            "price_high": vals[2],
            "price_low": vals[3],
            "price_close": vals[4],
            "volume": vals[5],
            "timestamp_close": vals[6],
            "asset_volume_quote": vals[7],
            "trades_number": vals[8],
            "asset_volume_taker_base": vals[9],
            "asset_volume_taker_quote": vals[10]
        })
    return klines