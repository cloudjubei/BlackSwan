from fastapi import APIRouter, Depends, HTTPException, WebSocket, status
from fastapi.responses import JSONResponse

import os
import json
import time
import random
import datetime
import traceback
import pandas as pd
from src.datamining.binance_mine import getTimeSeriesData, getIntervalTime
from src.indicators.prepare_indicators import prepareAllIndicators


router = APIRouter(
    prefix="/backtesting",
    tags=["backtesting"],
    responses={404: {"description": "Not found"}},
)


@router.get("/assets")
async def get_assets_with_times():
    with open("data/raw/symbols.json") as f:
        data = json.load(f)
        data = [dict(name=k, start_time=v["startTime"]) for k, v in data.items()]
    return data


@router.put("/assets/{name}")
async def add_asset_with_time(name: str):
    try:
        with open("data/raw/symbols.json", "r") as f:
            data = json.load(f)
        if name in data and "startTime" in data[name]:
            return HTTPException(499, detail="Already exists")

        beginning = 0 # 01.01.1970
        now = int(time.time() * 1000)
        result = getTimeSeriesData(name, "1s", beginning, now, 1)
        timestamp = result[0]["timestamp_open"]        
        if name not in data:
            data[name] = {}
        data[name]["startTime"] = timestamp
        with open("data/raw/symbols.json", "w") as f:
            json.dump(data, f, indent=4)
        return dict(name=name, start_time=timestamp)
    except Exception as ex:
        return HTTPException(499, detail=f'Not available in binance: {ex}')


@router.websocket("/run")
async def start_executions(websocket: WebSocket):
    n = 10
    statuses = ['running'] + ['waiting'] * (n + 1)
    statuses = [dict(name='Downloading (0%)', stage='running')] + \
                [dict(name=f'Running test {i+1}', stage='waiting') for i in range(n)] + \
                [dict(name='Analyzing results', stage='waiting')]
    # receive params
    await websocket.accept()
    params = await websocket.receive_text()
    params = json.loads(params)
    await websocket.send_text(json.dumps(dict(statuses=statuses)))
    # download
    df = await execution_step_download(params, statuses, websocket)
    # execute
    res = await execution_step_run(params, statuses, websocket, df, n)
    # combine and save to file
    await execution_step_save(params, statuses, websocket, res)
    # disconnect
    time.sleep(1)
    await websocket.send_text(json.dumps(dict(redirect=True)))
    await websocket.close()


async def execution_step_download(params, statuses, websocket):
    try:
        async for progress, df in download(params['asset'], params['interval'], params['timeframe_min'], params['timeframe_max']):
            statuses[0] = dict(name=f'Downloading ({progress}%)', stage='running')
            await websocket.send_text(json.dumps(dict(statuses=statuses)))

        statuses[0] = dict(name='Downloading (100%)', stage='pass')
        await websocket.send_text(json.dumps(dict(statuses=statuses)))
        return df
    except:
        traceback.print_exc()
        statuses[0]['stage'] = 'fail'
        for i in range(1, len(statuses)):
            statuses[i]['stage'] = 'skip'
        await websocket.send_text(json.dumps(dict(statuses=statuses)))
        await websocket.close()
        return None


async def execution_step_run(params, statuses, websocket, df, n):
    results = []
    for i in range(n):
        try:
            async for progress, res in run(df, params['action_probability'], params['code'], params['fee_fixed'], params['fee_minimum'], params['fee_percent']):
                statuses[i+1] = dict(name=f'Running test {i+1} ({progress}%)', stage='running')
                await websocket.send_text(json.dumps(dict(statuses=statuses)))
            results.append(res)

            statuses[i+1] = dict(name=f'Running test {i+1} (100%)', stage='pass')
        except:
            traceback.print_exc()
            statuses[i+1]['stage'] = 'fail'
        finally:
            await websocket.send_text(json.dumps(dict(statuses=statuses)))
    return results


async def execution_step_save(params, statuses, websocket, res):
    try:
        if len(res) == 0: raise Exception()
        await save(res, params)
        statuses[-1]['stage'] = 'pass'
    except:
        statuses[-1]['stage'] = 'fail'
    finally:
        await websocket.send_text(json.dumps(dict(statuses=statuses)))


async def download(asset: str, interval: str, timeframe_min: int, timeframe_max: int) -> pd.DataFrame:
    # download, prepare and return df from selected timerange
    interval_ms = getIntervalTime(interval)
    asset_start_time = getTimeSeriesData(asset, interval, 0, int(time.time() * 1000), 1)[0]['timestamp_open']
    
    def _nearest_timeframe(timeframe, typ):
        n = (timeframe - asset_start_time) / interval_ms
        add = 0 if typ == "lower" else 1
        if n == int(n): return timeframe
        else: return asset_start_time + interval_ms * (int(n) + add)
    
    nearest_correct_timefarme_min = _nearest_timeframe(timeframe_min, 'higher')
    nearest_correct_timefarme_max = _nearest_timeframe(timeframe_max, 'lower')
    times_to_download = []

    filename = f'data/raw/binance_{asset}_{interval}.csv'
    if not os.path.exists(filename):
        times_to_download = [(nearest_correct_timefarme_min, nearest_correct_timefarme_max)]
        df = pd.DataFrame()
    else:
        df = pd.read_csv(filename)
        times_to_download = []
        prev = nearest_correct_timefarme_min
        part_start = nearest_correct_timefarme_min
        times = range(nearest_correct_timefarme_min, nearest_correct_timefarme_max+1, interval_ms)
        times = [t for t in times if not df['timestamp_open'].eq(t).any()]
        for t in times:
            if t == prev + interval_ms:
                prev = t
            else:
                if prev != part_start:
                    times_to_download.append((part_start, prev))
                part_start = t
                prev = t
        if prev != part_start:
            times_to_download.append((part_start, prev))
    
    all_requests = 0
    requests = 0
    for start, end in times_to_download:
        all_requests += int(((end - start) / interval_ms) / 1000) + 1

    for start, end in times_to_download:
        print(start, end)
        part_df = pd.DataFrame()
        n = int((end - start) / interval_ms)
        for i in range(0, n, 1000):
            part_start = start + interval_ms * i - 1
            part_end = start + interval_ms * (i + 1000) + 1
            res = getTimeSeriesData(asset, interval, part_start, part_end, 1000)
            res = pd.DataFrame(res)
            for c in ['price_open', 'price_high', 'price_low', 'price_close', 'volume', 'asset_volume_quote', 'asset_volume_taker_base', 'asset_volume_taker_quote']:
                res[c] = res[c].astype(float)
            part_df = pd.concat([part_df, res]).reset_index(drop=True)
            time.sleep(0.3) # 200 requests per minute
        # fill the spaces in part_df
        for t in range(start, end+1, interval_ms):
            if not part_df['timestamp_open'].eq(t).any():
                empty = pd.DataFrame([{'timestamp_open': t}])
                part_df = pd.concat([part_df, empty]).reset_index(drop=True)
        # create additional params
        part_df = prepareAllIndicators(part_df, 1)
        # update web if available
        yield (int(100 * (requests + 1) / all_requests), None)
        requests += 1

        df = pd.concat([df, part_df]).reset_index(drop=True)
    df = df.sort_values(by='timestamp_open')
    
    df.to_csv(filename, index=False)
    yield (100, df[
        (df['timestamp_open'] > nearest_correct_timefarme_min) &
        (df['timestamp_open'] < nearest_correct_timefarme_max) &
        (df['price_open'].notna())])


async def run(df: pd.DataFrame, action_probability: list[int], code: str, fee_fixed: float = None, fee_minimum: float = None, fee_percent: int = None) -> dict:
    # run and return results
    buys = [] # timestamp, price_per_one, price, amount, fee
    sells = [] # timestamp, price_per_one, price, amount, fee
    balance_assets = 0
    balance_account = 1_000_000
    skip_action_rows = 0
    # TODO buy/sell strategies (amount to buy each time)
    
    for i, row in df.iterrows():
        # update web if available
        yield (int(100 * i / len(df)), None)

        if skip_action_rows > 0:
            skip_action_rows -= 1
            continue
        context = row.to_dict()
        if code is not None: exec(code, context)
        action = context.get('action')
        if action == 'buy' and balance_account > 0:
            fee = 0
            if fee_fixed is not None:
                fee = fee_fixed
            if fee_percent is not None and fee_minimum is not None:
                fee = (fee_percent / 100) * balance_account
                if fee < fee_minimum: fee = fee_minimum
            cash = balance_account - fee
            if cash < 0: break
            prices = list(df.loc[i:i+5, 'price_close'])
            prices = prices + (5 - len(prices)) * [prices][-1]
            action_i = random.choices(range(5), action_probability)[0]
            price_per_one = prices[action_i]
            skip_action_rows = action_i - 1
            amount = cash / price_per_one
            buys.append((row.timestamp_close, price_per_one, cash, amount, fee))
            balance_account = 0
            balance_assets = amount
            continue
        
        if action == 'sell' and balance_assets > 0:
            prices = list(df.loc[i:i+5, 'price_close'])
            prices = prices + (5 - len(prices)) * [prices][-1]
            action_i = random.choices(range(5), action_probability)[0]
            price_per_one = prices[action_i]
            skip_action_rows = action_i - 1
            cash = balance_assets * price_per_one
            if fee_fixed is not None:
                fee = fee_fixed
            if fee_percent is not None and fee_minimum is not None:
                fee = (fee_percent / 100) * cash
                if fee < fee_minimum: fee = fee_minimum
            cash -= fee
            sells.append((row.timestamp_close, price_per_one, cash, balance_assets, fee))
            balance_account = cash
            balance_assets = 0
            continue
    
    if balance_assets > 0:
        price_per_one = df.iloc[-1]['price_close']
        cash = balance_assets * price_per_one
        fee = 0
        if fee_fixed is not None:
            fee = fee_fixed
        if fee_percent is not None and fee_minimum is not None:
            fee = (fee_percent / 100) * cash
            if fee < fee_minimum: fee = fee_minimum
        cash -= fee
        balance_account = cash
        sells.append((row.timestamp_close, price_per_one, cash, balance_assets, fee))
    
    # for b in buys: print('buy', b)
    # for b in sells: print('sell', b)

    fees = [(b[0], b[4]) for b in buys] + [(s[0], s[4]) for s in sells]
    fees = sorted(fees, key=lambda x: x[0])
    fees = [f[1] for f in fees]

    data = {
        'n_buys': len(buys),
        'n_sells': len(sells),
        'buys_timestamps': [b[0] for b in buys],
        'sells_timestamps': [s[0] for s in sells],
        'fees': fees,
        'return': (balance_account - 1_000_000) / 1_000_000
    }
    
    # hold return
    balance_assets = 0
    balance_account = 1_000_000
    fee = 0
    if fee_fixed is not None:
        fee = fee_fixed
    if fee_percent is not None and fee_minimum is not None:
        fee = (fee_percent / 100) * balance_account
        if fee < fee_minimum: fee = fee_minimum
    cash = balance_account - fee
    price_per_one = df.iloc[0]['price_close']
    balance_assets = cash / price_per_one

    price_per_one = df.iloc[-1]['price_close']
    cash = balance_assets * price_per_one
    fee = 0
    if fee_fixed is not None:
        fee = fee_fixed
    if fee_percent is not None and fee_minimum is not None:
        fee = (fee_percent / 100) * cash
        if fee < fee_minimum: fee = fee_minimum
    cash -= fee
    hold_return = (cash - 1_000_000) / 1_000_000
    data['return_hold'] = hold_return

    # print('operations', data['n_buys'], 'return', f"{data['return']*100:.2f}%", 'hold', f"{data['return_hold']*100:.2f}%")

    yield (100, data)


async def save(res: list[dict], params: dict):
    df = pd.DataFrame([{**params, **r} for r in res])
    df['timestamp'] = int(time.time() * 1000)
    filename = 'data/results/history.csv'
    if not os.path.exists(filename):
        os.system('mkdir -p data/results')
        with open(filename, 'w+') as f:
            df.to_csv(f, header=True, index=False)
    else:
        with open(filename, 'a') as f:
            df.to_csv(f, header=False, index=False)


@router.get("/history")
async def get_history_params():
    try:
        df = pd.read_csv('data/results/history.csv')
        df = df.fillna('')
        same = df[['name', 'asset', 'interval', 'timeframe_min', 'timeframe_max', 'fee_fixed', 'fee_minimum', 'fee_percent', 'action_probability', 'code', 'timestamp']]
        same = df.groupby('timestamp').agg(lambda items: min(items)).reset_index()
        avg = df[['return', 'return_hold', 'timestamp']].groupby('timestamp').agg(lambda items: sum(items)/len(items))
        data = zip(same.to_dict(orient='records'), avg.to_dict(orient='records'))
        data = [{**s, **a} for (s, a) in data][::-1]
        return data
    except:
        traceback.print_exc()
        return []


@router.get("/history/{timestamp}")
async def get_specific_history_param(timestamp: int):
    try:
        df = pd.read_csv('data/results/history.csv')
        df = df.fillna('')
        df = df[df['timestamp'] == timestamp]
        same = df[['name', 'asset', 'interval', 'timeframe_min', 'timeframe_max', 'fee_fixed', 'fee_minimum', 'fee_percent', 'action_probability', 'code', 'timestamp']]
        same = same.groupby('timestamp').agg(lambda items: min(items))
        variants = df[['return', 'return_hold', 'n_buys', 'n_sells', 'buys_timestamps', 'sells_timestamps', 'fees']]
        response = same.iloc[0].to_dict()
        response['variants'] = variants.to_dict(orient='records')

        asset = response['asset']
        interval = response['interval']
        prices = pd.read_csv(f'data/raw/binance_{asset}_{interval}.csv')
        prices = prices[(prices['timestamp_open'] >= response['timeframe_min']) &
            (prices['timestamp_open'] <= response['timeframe_max']) &
            (prices['price_close'].notna())]
        prices['timestamp'] = prices['timestamp_close']
        prices['price'] = prices['price_close']
        prices = prices[['timestamp', 'price']]
        response['price'] = prices.to_dict(orient='records')
        return response
    except:
        traceback.print_exc()
        return {}


@router.delete("/history/{timestamp}")
async def remove_specific_history_param(timestamp: int):
    try:
        df = pd.read_csv('data/results/history.csv')
        df = df[df['timestamp'] != timestamp]
        with open('data/results/history.csv', 'w+') as f:
            df.to_csv(f, header=True, index=False)
        return dict(successfull=True)
    except:
        traceback.print_exc()
        return dict(successfull=False)
