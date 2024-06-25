
from typing import List

import numpy as np
import pandas as pd
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider
from src.data.multitimeline_dataprovider import MultiTimelineDataProvider
from src.data.single_dataprovider import SingleDataProvider
import mplfinance as mpf

def create_provider(config: DataConfig, paths: List[str], extra_paths: List[List[str]] | None = None) -> AbstractDataProvider:
    if len(config.layers) == 1:
        return SingleDataProvider(config, paths)
    elif len(config.layers) > 1:
        return MultiTimelineDataProvider(config, extra_paths)
    raise ValueError("{config.type} - env not supported")
