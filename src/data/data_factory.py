
from typing import List

import numpy as np
import pandas as pd
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider
from src.data.multitimeline_dataprovider import MultiTimelineDataProvider
from src.data.single_dataprovider import SingleDataProvider

def create_provider(config: DataConfig, paths: List[List[str]], fidelity_input: str, fidelity_run: str, layers: List[str], buyreward_maxwait: float, buyreward_percent: float) -> AbstractDataProvider:
    if len(config.layers) == 1:
        return SingleDataProvider(config, paths[0])
    elif len(config.layers) > 1:
        return MultiTimelineDataProvider(config, paths, fidelity_input, fidelity_run, layers, buyreward_maxwait, buyreward_percent)
    raise ValueError("{config.type} - env not supported")
