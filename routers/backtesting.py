from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

import json
import time
import datetime
import traceback

from src.datamining.binance_mine import getTimeSeriesData

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
