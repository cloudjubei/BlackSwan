
from typing import List

import numpy as np
import pandas as pd
from src.conf.data_config import DataConfig
from src.data.abstract_dataprovider import AbstractDataProvider
from src.data.multitimeline_dataprovider import MultiTimelineDataProvider
from src.data.single_dataprovider import SingleDataProvider

def create_provider(config: DataConfig, paths: List[List[str]], fidelity_input: str, fidelity_run: str, layers: List[str], buyreward_maxwait: float, buyreward_percent: float) -> AbstractDataProvider:
    if len(config.layers) < 1:
        raise ValueError(f"{config.type} - env not supported")
    # The fast single-timeline path only fits a lone layer that IS the loaded base AND is stepped at the
    # base cadence (no resampling, no day-by-day stepping). A single COARSER layer (e.g. '1d' at a 1h step)
    # OR a base layer at a COARSER step (e.g. '1h' at a 1d step, divider_run > 1) needs the multi-timeline
    # provider — same machinery as a multi-layer stack with one layer.
    if len(config.layers) == 1 and config.layers[0] == fidelity_input and fidelity_run == fidelity_input:
        return SingleDataProvider(config, paths[0])
    return MultiTimelineDataProvider(config, paths, fidelity_input, fidelity_run, layers, buyreward_maxwait, buyreward_percent)
